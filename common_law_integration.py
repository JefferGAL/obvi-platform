#!/usr/bin/env python3
"""
Common Law Search Integration Components
Integration with existing trademark search system
"""

import logging
import json
import re
import csv
import io
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from pathlib import Path

from common_law_analyzer import CommonLawResult

logger = logging.getLogger(__name__)

# =============================================================================
# SESSION-BASED TEMPORARY DATA MANAGEMENT
# =============================================================================

class SessionBasedCommonLawStorage:
    """Manages temporary common law search data using session-based JSON files"""
    
    def __init__(self, temp_directory: str = "./temp"):
        self.temp_directory = Path(temp_directory)
        self.temp_directory.mkdir(exist_ok=True)
        self.active_sessions = {}  # Track active session data files
        
    def generate_storage_id(self, search_term: str, session_id: str) -> str:
        """Generate unique storage ID for search results"""
        # Sanitize search term for filename
        sanitized_term = re.sub(r'[^\w\-_\.]', '_', search_term.lower())
        
        # Check for existing files with same search term for this session
        base_id = f"{sanitized_term}_{session_id}"
        counter = 1
        
        while (self.temp_directory / f"{base_id}_{counter}.json").exists():
            counter += 1
            
        return f"{base_id}_{counter}"
    
    async def store_search_results(
        self, 
        search_term: str, 
        session_id: str, 
        results: Dict[str, Any]
    ) -> str:
        """Store common law search results in temporary JSON file"""
        try:
            storage_id = self.generate_storage_id(search_term, session_id)
            file_path = self.temp_directory / f"{storage_id}.json"
            
            # Prepare data with metadata
            data = {
                "search_term": search_term,
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
                "results": results,
                "access_count": 0
            }
            
            # Write to JSON file with secure permissions
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Set restrictive file permissions (owner read/write only)
            file_path.chmod(0o600)
            
            # Track in active sessions
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = []
            self.active_sessions[session_id].append({
                "storage_id": storage_id,
                "file_path": str(file_path),
                "search_term": search_term,
                "created_at": datetime.now(timezone.utc)
            })
            
            logger.info(f"✅ Common law search results stored: {storage_id}")
            return storage_id
            
        except Exception as e:
            logger.error(f"❌ Failed to store common law search results: {str(e)}")
            raise
    
    async def retrieve_search_results(self, storage_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve common law search results from temporary storage"""
        try:
            file_path = self.temp_directory / f"{storage_id}.json"
            
            if not file_path.exists():
                logger.warning(f"⚠️ Common law data file not found: {storage_id}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Verify session ownership
            if data.get("session_id") != session_id:
                logger.warning(f"⚠️ Session mismatch for common law data: {storage_id}")
                return None
            
            # Check expiration
            expires_at = datetime.fromisoformat(data["expires_at"].replace('Z', '+00:00'))
            if datetime.now(timezone.utc) > expires_at:
                logger.warning(f"⚠️ Common law data expired: {storage_id}")
                await self.delete_storage_file(storage_id)
                return None
            
            # Update access count
            data["access_count"] += 1
            data["last_accessed"] = datetime.now(timezone.utc).isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            return data["results"]
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve common law search results: {str(e)}")
            return None
    
    async def delete_storage_file(self, storage_id: str) -> bool:
        """Delete a specific storage file"""
        try:
            file_path = self.temp_directory / f"{storage_id}.json"
            if file_path.exists():
                file_path.unlink()
                logger.info(f"✅ Deleted common law data file: {storage_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete common law data file {storage_id}: {str(e)}")
            return False
    
    async def cleanup_session_data(self, session_id: str) -> int:
        """Clean up all data files for a specific session"""
        cleaned_count = 0
        
        try:
            if session_id in self.active_sessions:
                session_files = self.active_sessions[session_id]
                
                for file_info in session_files:
                    if await self.delete_storage_file(file_info["storage_id"]):
                        cleaned_count += 1
                
                # Remove from active sessions tracking
                del self.active_sessions[session_id]
                
                logger.info(f"✅ Cleaned up {cleaned_count} common law data files for session: {session_id}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup session data for {session_id}: {str(e)}")
            return cleaned_count
    
    async def cleanup_expired_data(self) -> int:
        """Clean up all expired data files"""
        cleaned_count = 0
        current_time = datetime.now(timezone.utc)
        
        try:
            # Scan all JSON files in temp directory
            for file_path in self.temp_directory.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    expires_at = datetime.fromisoformat(data["expires_at"].replace('Z', '+00:00'))
                    
                    if current_time > expires_at:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"🗑️ Cleaned expired file: {file_path.name}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error processing file {file_path}: {str(e)}")
                    # If we can't read the file, delete it as it's likely corrupted
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except:
                        pass
            
            logger.info(f"✅ Cleaned up {cleaned_count} expired common law data files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup expired data: {str(e)}")
            return cleaned_count
    
    async def get_session_storage_info(self, session_id: str) -> List[Dict[str, Any]]:
        """Get information about stored data for a session"""
        if session_id not in self.active_sessions:
            return []
        
        storage_info = []
        for file_info in self.active_sessions[session_id]:
            try:
                file_path = Path(file_info["file_path"])
                if file_path.exists():
                    stat = file_path.stat()
                    storage_info.append({
                        "storage_id": file_info["storage_id"],
                        "search_term": file_info["search_term"],
                        "created_at": file_info["created_at"].isoformat(),
                        "file_size": stat.st_size,
                        "downloadable": True
                    })
            except Exception as e:
                logger.warning(f"⚠️ Error getting info for {file_info['storage_id']}: {str(e)}")
        
        return storage_info

# Global instance for session-based storage
_session_storage = None

def get_session_storage() -> SessionBasedCommonLawStorage:
    """Get global session storage instance"""
    global _session_storage
    if _session_storage is None:
        _session_storage = SessionBasedCommonLawStorage()
    return _session_storage

# =============================================================================
# API AND REPORT INTEGRATION
# =============================================================================

class CommonLawAPIIntegration:
    """Integration with FastAPI endpoints"""
    
    def __init__(self, db_manager):
        # db_manager is no longer used for common law data storage but is kept for consistency
        # with the main app's initialization
        pass
    
    @staticmethod
    def format_for_api(search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format search results for API response"""
        
        formatted = {
            "summary": {
                "total_sources_checked": search_results.get('total_sources_checked', 0),
                "overall_risk": search_results.get('risk_summary', {}).get('overall_risk', 'unknown')
            },
            "categories": {}
        }
        
        results_by_category = search_results.get('results_by_category', {})
        
        for category, results_list in results_by_category.items():
            formatted_results = []
            
            for result in results_list:
                formatted_result = asdict(result)
                formatted_results.append(formatted_result)
            
            formatted["categories"][category] = {
                "count": len(formatted_results),
                "results": formatted_results
            }
        
        return formatted

# =============================================================================
# CSV EXPORT
# =============================================================================
def export_common_law_report_csv(results: Dict[str, Any], trademark: str) -> str:
    """
    Generates a CSV report from common law search results.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Define CSV headers
    headers = [
        "Search_Trademark", "Source_Type", "Source_Name", "URL", "Status",
        "Owner_Info", "Summary", "Risk_Level", "Confidence_Score"
    ]
    writer.writerow(headers)
    
    # Write data rows
    for category_results in results.get('results_by_category', {}).values():
        for result in category_results:
            if isinstance(result, CommonLawResult):
                writer.writerow([
                    trademark,
                    result.source_type,
                    result.source_name,
                    result.url,
                    result.status,
                    result.owner_info,
                    result.summary,
                    result.risk_level,
                    f"{result.confidence_score:.2f}"
                ])
                
    return output.getvalue()
    
# =============================================================================
# REPORT INTEGRATION
# =============================================================================

class CommonLawReportIntegration:
    """Integration with trademark report generator"""
    
    @staticmethod
    def get_disclaimer() -> str:
        """Returns a static legal disclaimer for common law reports."""
        return """
**IMPORTANT LEGAL DISCLAIMER: COMMON LAW SEARCH**

This report summarizes common law findings for selected owners, but it is **not** a comprehensive clearance search and does **not** constitute legal advice.

**LIMITATIONS:**
- Automated searches may miss relevant information.
- State-level and local business registrations are not fully covered.
- This report is not a substitute for professional legal counsel.

You should consult with a qualified trademark attorney to conduct a full, professional clearance search and receive legal guidance.
"""

    @staticmethod
    def generate_common_law_section(search_results: Dict[str, Any]) -> str:
        """Generate markdown section for common law search results"""
        
        if not search_results or not search_results.get('success'):
            return "\n## Common Law Search\n\n**Status:** Search not performed or failed\n"
        
        common_law_results = search_results.get('common_law_results', {})
        risk_assessment = search_results.get('risk_assessment', {})
        recommendations = search_results.get('recommendations', [])
        
        markdown = "\n---\n\n## Common Law Search Results\n\n"
        
        # Summary section
        summary = common_law_results.get('summary', {})
        markdown += f"**Search Summary:**\n"
        markdown += f"- **Sources Checked:** {summary.get('total_sources_checked', 0)}\n"
        markdown += f"- **Overall Risk Level:** {summary.get('overall_risk', 'Unknown').upper()}\n\n"
        
        # Risk assessment
        if risk_assessment:
            markdown += "**Risk Breakdown:**\n"
            markdown += f"- High Risk Conflicts: {risk_assessment.get('high_risk_conflicts', 0)}\n"
            markdown += f"- Medium Risk Conflicts: {risk_assessment.get('medium_risk_conflicts', 0)}\n"
            markdown += f"- Low Risk Conflicts: {risk_assessment.get('low_risk_conflicts', 0)}\n\n"
        
        # Detailed results by category
        categories = common_law_results.get('categories', {})
        
        for category, category_data in categories.items():
            if category_data.get('count', 0) > 0:
                markdown += f"### {category.title()} Search Results\n\n"
                
                for result in category_data.get('results', []):
                    markdown += f"**{result.get('source', 'Unknown Source')}**\n"
                    markdown += f"- Status: {result.get('status', 'Unknown')}\n"
                    markdown += f"- Risk Level: {result.get('risk_level', 'Unknown').upper()}\n"
                    markdown += f"- Confidence: {result.get('confidence_score', 0):.1%}\n"
                    
                    if result.get('business_info'):
                        markdown += f"- Business: {result.get('business_info')}\n"
                    
                    if result.get('url'):
                        markdown += f"- URL: {result.get('url')}\n"
                    
                    markdown += "\n"
        
        # Recommendations
        if recommendations:
            markdown += "### Common Law Recommendations\n\n"
            for i, rec in enumerate(recommendations, 1):
                markdown += f"{i}. {rec}\n"
            markdown += "\n"
        
        # Important notice
        markdown += "**Important Note:** Common law searches are not comprehensive and may not capture all unregistered trademark rights. Professional legal counsel should review these results and conduct additional research as needed.\n\n"
        
        return markdown
        
def initialize_common_law_system(db_manager):
    """Initialize the common law search system - this function is now a placeholder as db is removed"""
    logger.info("Common law system initialized (no database required)")

if __name__ == "__main__":
    # Test the integration
    async def test_integration():
        # This would test with your actual database manager
        print("Common law integration components ready")
    
    asyncio.run(test_integration())