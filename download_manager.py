#!/usr/bin/env python3
"""
Download Manager Module
Export search results in multiple formats (CSV, JSON, TXT)
"""

import csv
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from io import StringIO

logger = logging.getLogger(__name__)

class DownloadManager:
    """Manage export of search results in various formats"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'txt', 'variations']
    
    def export_results_csv(self, results: Dict[str, Any], trademark: str) -> str:
        """Exports the main search results to a CSV string, including scores and risk."""
        output = StringIO()
        writer = csv.writer(output)
        
        # ANNOTATION: The header is expanded to include the missing score and risk columns.
        header = [
            'Serial Number', 'Mark', 'Owner', 'Status', 'Classes',
            'Risk Level', 'Overall Score', 'Phonetic Score', 'Visual Score', 'Conceptual Score'
        ]
        writer.writerow(header)
        
        for match in results.get('matches', []):
            # ANNOTATION: The similarity_scores dictionary is now accessed.
            scores = match.get('similarity_scores', {})
            
            # ANNOTATION: The row is populated with the new score and risk data.
            row = [
                match.get('serial_number', ''),
                match.get('mark_identification', ''),
                match.get('owner', ''),
                match.get('status_code', ''),
                ', '.join(match.get('nice_classes', [])),
                match.get('risk_level', 'unknown'),
                f"{(scores.get('overall', 0) * 100):.1f}%",
                f"{(scores.get('phonetic', 0) * 100):.1f}%",
                f"{(scores.get('visual', 0) * 100):.1f}%",
                f"{(scores.get('conceptual', 0) * 100):.1f}%"
            ]
            writer.writerow(row)
        
        return output.getvalue()

    def export_results_json(self, results: Dict[str, Any], trademark: str, variations: Dict[str, List[str]] = None) -> str:
        """Exports the full search results object to a JSON string."""
        # ANNOTATION: No changes are needed here, as the 'results' object already
        # contains the complete data including similarity_scores and risk_level.
        export_data = {
            "search_metadata": {
                "query_trademark": results.get('query_trademark', 'N/A'),
                "total_matches": results.get('total_matches', 0),
                "search_mode": results.get('search_mode', 'N/A'),
                "export_timestamp": datetime.now().isoformat()
            },
            "matches": results.get('matches', [])
        }
        if variations:
           export_data["variations_analysis"] = variations
        return json.dumps(export_data, indent=2, ensure_ascii=False)

    def export_results_txt(self, results: Dict[str, Any], trademark: str) -> str:
        """Exports the main search results to a detailed plain text report."""
        output = StringIO()
        
        output.write(f"TRADEMARK SEARCH REPORT\n")
        output.write("=" * 50 + "\n\n")
        output.write(f"Query: {results.get('query_trademark', 'N/A')}\n")
        output.write(f"Total Matches Found: {results.get('total_matches', 0)}\n")
        output.write(f"Search Mode: {results.get('search_mode', 'N/A').upper()}\n")
        output.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for i, match in enumerate(results.get('matches', []), 1):
            scores = match.get('similarity_scores', {})
            output.write(f"--- Match #{i} ---\n")
            output.write(f"  Mark: {match.get('mark_identification', 'N/A')}\n")
            output.write(f"  Owner: {match.get('owner', 'N/A')}\n")
            output.write(f"  Serial: {match.get('serial_number', 'N/A')}\n")
            output.write(f"  Status: {match.get('status_code', 'N/A')}\n")
            output.write(f"  Classes: {', '.join(match.get('nice_classes', []))}\n")
            output.write(f"  Risk Level: {match.get('risk_level', 'unknown').upper()}\n")
            # ANNOTATION: Adds the detailed score breakdown to the text report.
            output.write("  Similarity Scores:\n")
            output.write(f"    - Overall: {(scores.get('overall', 0) * 100):.1f}%\n")
            output.write(f"    - Phonetic: {(scores.get('phonetic', 0) * 100):.1f}%\n")
            output.write(f"    - Visual: {(scores.get('visual', 0) * 100):.1f}%\n")
            output.write(f"    - Conceptual: {(scores.get('conceptual', 0) * 100):.1f}%\n\n")

        return output.getvalue()
        

    def export_common_law_report(self, results: Dict[str, Any], trademark: str) -> str:
        """Exports the hierarchical common law investigation results to a text file."""
        output = StringIO()
        
        output.write("COMMON LAW INVESTIGATION REPORT\n")
        output.write("=" * 50 + "\n\n")
        output.write(f"Initial Query Trademark: {trademark}\n")
        output.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if not results:
            output.write("No investigation results were provided.\n")
            return output.getvalue()

        for mark, owners in results.items():
            output.write(f"--- MARK: {mark} ---\n")
            for owner, data in owners.items():
                output.write(f"\n  OWNER: {owner}\n")
                output.write(f"  {'-' * (len(owner) + 8)}\n")
                
                if data.get('common_law_findings'):
                    for finding in data['common_law_findings']:
                        output.write(f"    - Source Type: {finding.get('source_type', 'N/A').replace('_', ' ').title()}\n")
                        output.write(f"      Summary: {finding.get('summary', 'N/A')}\n")
                        if finding.get('url'):
                            output.write(f"      URL: {finding.get('url')}\n")
                        if finding.get('mark_found_on_site'):
                            output.write(f"      NOTE: The original mark was found on this owner's website.\n")
                else:
                    output.write("    - No common law findings for this owner.\n")
            output.write("\n")

        output.write("END OF REPORT\n")
        return output.getvalue()

    def export_variations_list(self, variations: Dict[str, List[str]], trademark: str) -> str:
        """Export variations list in copy-paste friendly format"""
        
        output = StringIO()
        
        # Header
        output.write(f"Trademark Variations for: {trademark}\n")
        output.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output.write("=" * 50 + "\n\n")
        
        # All variations in one list
        all_variations = []
        for var_type, var_list in variations.items():
            if not var_type.startswith('_') and isinstance(var_list, list):
                all_variations.extend(var_list)
        
        # Remove duplicates and sort
        unique_variations = sorted(list(set(all_variations)))
        
        output.write(f"All Variations ({len(unique_variations)} total):\n")
        output.write("-" * 30 + "\n")
        for i, variation in enumerate(unique_variations, 1):
            output.write(f"{i:3d}. {variation}\n")
        
        output.write("\n" + "=" * 50 + "\n")
        output.write("Categorized Variations:\n\n")
        
        # Categorized variations
        category_names = {
            'phonetic': 'Phonetic (Sound-alike)',
            'visual': 'Visual (Appearance)',
            'morphological': 'Morphological (Word Forms)',
            'conceptual': 'Conceptual (Meaning-based)',
            'leet_speak': 'Leet Speak (Character Substitution)'
        }
        
        for var_type, var_list in variations.items():
            if var_type.startswith('_') or not isinstance(var_list, list):
                continue
            
            category_name = category_names.get(var_type, var_type.title())
            output.write(f"{category_name} ({len(var_list)} variations):\n")
            output.write("-" * (len(category_name) + 20) + "\n")
            
            for i, variation in enumerate(var_list, 1):
                output.write(f"  {i:2d}. {variation}\n")
            output.write("\n")
        
        return output.getvalue()
    
    def export_detailed_report(self, results: Dict[str, Any], trademark: str) -> str:
        """Export detailed text report"""
        
        output = StringIO()
        
        # Header
        output.write("TRADEMARK SEARCH DETAILED REPORT\n")
        output.write("=" * 50 + "\n\n")
        
        # Search Information
        output.write("SEARCH INFORMATION\n")
        output.write("-" * 20 + "\n")
        output.write(f"Query Trademark: {trademark}\n")
        output.write(f"Search Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output.write(f"Total Matches Found: {len(results.get('matches', []))}\n")
        output.write(f"Search Execution Time: {results.get('execution_time_ms', 0):.2f} ms\n")
        output.write(f"Search ID: {results.get('search_id', 'N/A')}\n\n")
        
        # Search Parameters
        search_params = results.get('search_parameters', {})
        if search_params:
            output.write("SEARCH PARAMETERS\n")
            output.write("-" * 20 + "\n")
            for key, value in search_params.items():
                if key != 'trademark':
                    output.write(f"{key.replace('_', ' ').title()}: {value}\n")
            output.write("\n")
        
        # Risk Assessment
        risk_assessment = results.get('risk_assessment', {})
        if risk_assessment:
            output.write("RISK ASSESSMENT\n")
            output.write("-" * 20 + "\n")
            output.write(f"Overall Risk Level: {risk_assessment.get('overall_risk', 'Unknown').upper()}\n")
            output.write(f"Risk Score: {risk_assessment.get('risk_score', 0):.1%}\n")
            output.write(f"High Risk Matches: {risk_assessment.get('high_risk_matches', 0)}\n")
            output.write(f"Summary: {risk_assessment.get('summary', 'No assessment available')}\n\n")
            
            # Recommendations
            recommendations = risk_assessment.get('recommendations', [])
            if recommendations:
                output.write("RECOMMENDATIONS\n")
                output.write("-" * 20 + "\n")
                for i, rec in enumerate(recommendations, 1):
                    output.write(f"{i}. {rec}\n")
                output.write("\n")
        
        # Performance Metrics
        performance = results.get('performance_metrics', {})
        if performance:
            output.write("PERFORMANCE METRICS\n")
            output.write("-" * 20 + "\n")
            output.write(f"Total Duration: {performance.get('total_duration', 0):.2f} seconds\n")
            output.write(f"Records Processed: {performance.get('total_records_processed', 0):,}\n")
            output.write(f"Processing Rate: {performance.get('processing_rate_per_second', 0):.0f} rec/sec\n")
            
            stages = performance.get('stages', [])
            if stages:
                output.write("\nStage Breakdown:\n")
                for stage in stages:
                    output.write(f"  {stage.get('stage_name', 'Unknown')}: {stage.get('duration_ms', 0):.0f}ms\n")
            output.write("\n")
        
        # Detailed Matches
        matches = results.get('matches', [])
        if matches:
            output.write("DETAILED MATCH RESULTS\n")
            output.write("-" * 30 + "\n\n")
            
            for i, match in enumerate(matches, 1):
                output.write(f"MATCH #{i}\n")
                output.write("=" * 15 + "\n")
                output.write(f"Mark: {match.get('mark_identification', 'N/A')}\n")
                output.write(f"Serial Number: {match.get('serial_number', 'N/A')}\n")
                output.write(f"Owner: {match.get('owner', 'N/A')}\n")
                output.write(f"Registration Number: {match.get('registration_number', 'N/A')}\n")
                output.write(f"Status: {match.get('status_code', 'N/A')}\n")
                output.write(f"Filing Date: {match.get('filing_date', 'N/A')}\n")
                output.write(f"NICE Classes: {', '.join(match.get('nice_classes', []))}\n")
                
                # Similarity Scores
                similarity = match.get('similarity_scores', {})
                output.write(f"\nSimilarity Scores:\n")
                output.write(f"  Phonetic: {similarity.get('phonetic', 0):.1%}\n")
                output.write(f"  Visual: {similarity.get('visual', 0):.1%}\n")
                output.write(f"  Conceptual: {similarity.get('conceptual', 0):.1%}\n")
                output.write(f"  Overall: {similarity.get('overall', 0):.1%}\n")
                
                output.write(f"\nRisk Level: {match.get('risk_level', 'Unknown').upper()}\n")
                output.write(f"Match Type: {match.get('match_type', 'Unknown')}\n")
                output.write(f"Confidence: {match.get('confidence_score', 0):.1%}\n")
                
                # Match Details
                match_details = match.get('match_details', {})
                if match_details:
                    output.write(f"\nMatch Details:\n")
                    for detail_type, detail_value in match_details.items():
                        output.write(f"  {detail_type.title()}: {detail_value}\n")
                
                output.write("\n" + "-" * 30 + "\n\n")
        
        # Footer
        output.write("END OF REPORT\n")
        output.write(f"Generated by Obvi Trademark Search API v2.0.0\n")
        output.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return output.getvalue()
    
    def export_performance_summary(self, results: Dict[str, Any], trademark: str) -> str:
        """Export performance-focused summary"""
        
        output = StringIO()
        
        # Header
        output.write("PERFORMANCE ANALYSIS SUMMARY\n")
        output.write("=" * 40 + "\n\n")
        
        output.write(f"Trademark: {trademark}\n")
        output.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Key Performance Metrics
        performance = results.get('performance_metrics', {})
        output.write("KEY METRICS\n")
        output.write("-" * 15 + "\n")
        output.write(f"Total Search Time: {performance.get('total_duration', 0):.2f} seconds\n")
        output.write(f"Records Analyzed: {performance.get('total_records_processed', 0):,}\n")
        output.write(f"Matches Found: {len(results.get('matches', []))}\n")
        output.write(f"Processing Rate: {performance.get('processing_rate_per_second', 0):.0f} records/second\n")
        output.write(f"Efficiency Rating: {performance.get('efficiency_rating', 'Unknown')}\n\n")
        
        # Stage Performance
        stages = performance.get('stages', [])
        if stages:
            output.write("STAGE PERFORMANCE\n")
            output.write("-" * 20 + "\n")
            for stage in stages:
                duration = stage.get('duration_ms', 0) / 1000
                processed = stage.get('records_processed', 0)
                found = stage.get('records_found', 0)
                
                output.write(f"{stage.get('stage_name', 'Unknown')}:\n")
                output.write(f"  Duration: {duration:.2f}s\n")
                output.write(f"  Processed: {processed:,} records\n")
                output.write(f"  Found: {found:,} matches\n")
                output.write(f"  Rate: {processed/max(duration, 0.001):.0f} rec/sec\n\n")
        
        # Optimization Suggestions
        suggestions = performance.get('optimization_suggestions', [])
        if suggestions:
            output.write("OPTIMIZATION SUGGESTIONS\n")
            output.write("-" * 25 + "\n")
            for i, suggestion in enumerate(suggestions, 1):
                output.write(f"{i}. {suggestion}\n")
            output.write("\n")
        
        # Search Statistics
        search_stats = results.get('search_statistics', {})
        if search_stats:
            output.write("SEARCH STATISTICS\n")
            output.write("-" * 20 + "\n")
            output.write(f"Variations Generated: {search_stats.get('total_variations_generated', 0)}\n")
            output.write(f"Database Candidates: {search_stats.get('database_candidates_found', 0)}\n")
            output.write(f"Final Matches: {search_stats.get('final_matches', 0)}\n")
            output.write(f"Search Classes: {', '.join(map(str, search_stats.get('search_classes', [])))}\n")
        
        return output.getvalue()
    
    def create_download_response(self, content: str, filename: str, content_type: str) -> Dict[str, Any]:
        """Create a download response with proper headers"""
        
        return {
            'content': content,
            'filename': filename,
            'content_type': content_type,
            'headers': {
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Type': content_type,
                'Content-Length': str(len(content.encode('utf-8')))
            }
        }
    
    def generate_filename(self, trademark: str, format_type: str, timestamp: bool = True) -> str:
        """Generate appropriate filename for download"""
        
        # Clean trademark name for filename
        clean_trademark = "".join(c for c in trademark if c.isalnum() or c in (' ', '-', '_')).strip()
        clean_trademark = clean_trademark.replace(' ', '_')
        
        # Add timestamp if requested
        if timestamp:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = f"{clean_trademark}_search_{timestamp_str}"
        else:
            base_name = f"{clean_trademark}_search"
        
        # Add appropriate extension
        extensions = {
            'csv': '.csv',
            'json': '.json',
            'txt': '.txt',
            'variations': '_variations.txt',
            'report': '_report.txt',
            'performance': '_performance.txt'
        }
        
        extension = extensions.get(format_type, '.txt')
        return base_name + extension
    
    def get_content_type(self, format_type: str) -> str:
        """Get appropriate content type for format"""
        
        content_types = {
            'csv': 'text/csv',
            'json': 'application/json',
            'txt': 'text/plain',
            'variations': 'text/plain',
            'report': 'text/plain',
            'performance': 'text/plain'
        }
        
        return content_types.get(format_type, 'text/plain')
    
    def export_format(self, results: Dict[str, Any], trademark: str, format_type: str, **kwargs) -> Dict[str, Any]:
        """Main export method that routes to appropriate format handler"""
        
        if format_type not in self.supported_formats + ['report', 'performance']:
            raise ValueError(f"Unsupported format: {format_type}")
        
        try:
            # Route to appropriate export method
            if format_type == 'csv':
                content = self.export_results_csv(results, trademark)
            elif format_type == 'json':
                content = self.export_results_json(results, trademark)
            elif format_type == 'variations':
                variations = kwargs.get('variations', {})
                content = self.export_variations_list(variations, trademark)
            elif format_type == 'report':
                content = self.export_detailed_report(results, trademark)
            elif format_type == 'performance':
                content = self.export_performance_summary(results, trademark)
            else:  # txt format or fallback
                content = self.export_detailed_report(results, trademark)
            
            # Generate filename and response
            filename = self.generate_filename(trademark, format_type)
            content_type = self.get_content_type(format_type)
            
            return self.create_download_response(content, filename, content_type)
            
        except Exception as e:
            logger.error(f"Export failed for format {format_type}: {str(e)}")
            raise
    
    def get_export_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about exportable data"""
        
        matches = results.get('matches', [])
        variations = results.get('variations_tested', {})
        
        stats = {
            'total_matches': len(matches),
            'high_risk_matches': len([m for m in matches if m.get('risk_level') in ['high', 'very_high']]),
            'total_variations': sum(len(v) for v in variations.values() if isinstance(v, list)),
            'variation_categories': len([k for k in variations.keys() if not k.startswith('_')]),
            'has_performance_data': 'performance_metrics' in results,
            'has_risk_assessment': 'risk_assessment' in results,
            'search_date': datetime.now().isoformat(),
            'exportable_formats': self.supported_formats + ['report', 'performance']
        }
        
        return stats

# Global instance for import convenience
download_manager = DownloadManager()