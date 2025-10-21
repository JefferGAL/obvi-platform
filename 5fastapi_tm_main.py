#!/usr/bin/env python3
"""
Enhanced FastAPI Trademark Search Application
Refactored for improved organization, readability, and maintainability.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import logging
import re
import io
import json
import csv
import os
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager
import time
import webbrowser
import threading

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Request, Query, Security, BackgroundTasks, Response
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Internal imports (modules within the same project)
from document_converter import DocumentConverter
from database_manager import DatabaseManager
from word_expansion_engine import WordExpansionEngine
from trademark_analyzer import TrademarkAnalyzer

try:
    from common_law_integration import initialize_common_law_system, export_common_law_report_csv, infer_industry_from_classes
    COMMON_LAW_AVAILABLE = True
except ImportError:
    COMMON_LAW_AVAILABLE = False
    logging.warning("Common Law integration system not available")

# Authentication imports
try:
   from auth_manager import AuthManager, get_current_user_info
   from separate_auth_db import UserStoreForSeparateDB
   from config_app_config import get_config
   from pydantic import BaseModel
   AUTH_AVAILABLE = True
except ImportError:
   AUTH_AVAILABLE = False
   logging.warning("Authentication system not available")

# Report generator imports
try:
   from trademark_report_generator import TrademarkReportGenerator, SearchParameters
   REPORT_GENERATOR_AVAILABLE = True
except ImportError:
   REPORT_GENERATOR_AVAILABLE = False
   logging.warning("TrademarkReportGenerator not available")

# Questionnaire imports
try:
    from llm_questionnaire import TrademarkQuestionnaire, QUESTIONNAIRE_QUESTIONS, wayfinding_questionnaire
    QUESTIONNAIRE_AVAILABLE = True
except ImportError:
    QUESTIONNAIRE_AVAILABLE = False
    logging.warning("TrademarkQuestionnaire not available")

from uspto_analytics import get_global_analytics

# =============================================================================
# LOGGING AND GLOBAL CONFIGURATION
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for search results
search_results_cache: Dict[str, Dict[str, Any]] = {}
CACHE_EXPIRY_HOURS = 24

# Global instances (initialized during application lifespan)
auth_manager = None
app_config = None
db_manager = None
word_expansion_engine = None
trademark_analyzer = None
document_converter = None
download_handler = None


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class LoginRequest(BaseModel):
   username: str
   password: str


# =============================================================================
# HANDLER CLASSES
# =============================================================================

class FilenameGenerator:
   """Generates consistent filenames for downloads"""
   
   def create_filename(self, trademark: str, format_type: str, report_type: str = "search") -> str:
       """Generate filename in format: clean_mark-[basic/enhanced]-[timestamp].[extension]"""
       
       # Clean trademark name for filename
       clean_trademark = re.sub(r'[^\w\s-]', '', trademark).strip()
       clean_trademark = re.sub(r'[-\s]+', '_', clean_trademark).lower()
       
       # Generate timestamp
       timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
       
       # Extension mapping
       extensions = {
           'csv': '.csv',
           'json': '.json',
           'txt': '.txt',
           'variations': '.txt',
           'report_md': '.md',
           'report_docx': '.docx',
           'common_law_csv': '.csv'
       }
       
       extension = extensions.get(format_type, '.txt')
       
       # Build filename: clean_mark-[report_type]-[timestamp].extension
       components = [clean_trademark, report_type, timestamp]
       components = [comp for comp in components if comp]
       
       return "_".join(components) + extension

class DownloadHandler:
   """Handles all download operations with multi-format document support"""
   
   def __init__(self, filename_generator: FilenameGenerator, document_converter: DocumentConverter):
       self.filename_generator = filename_generator
       self.document_converter = document_converter
       
       if REPORT_GENERATOR_AVAILABLE:
           self.report_generator = TrademarkReportGenerator()
       else:
           self.report_generator = None
           logger.warning("TrademarkReportGenerator not available")
   
   def create_streaming_response(self, content: Union[str, bytes], filename: str, media_type: str) -> StreamingResponse:
       """Create a StreamingResponse for file downloads"""
       if isinstance(content, str):
           content_bytes = content.encode('utf-8')
       else:
           content_bytes = content
           
       return StreamingResponse(
           io.BytesIO(content_bytes),
           media_type=media_type,
           headers={"Content-Disposition": f"attachment; filename={filename}"}
       )
   
   def generate_report_if_needed(self, cached_data: Dict, search_id: str, search_mode: str) -> Dict[str, Any]:
       """Generate report using existing TrademarkReportGenerator if not available"""
       if not self.report_generator:
           raise HTTPException(status_code=503, detail="Report generator not available")
   
       response_data = cached_data['response']
       variations_data = cached_data.get('variations', {})
       
       try:
           # Create SearchParameters object from response data
           search_params = SearchParameters(
               trademark=response_data.get('query_trademark', 'Unknown'),
               questionnaire_responses={},
               ai_recommended_classes=[],
               ai_reasoning="",
               coordination_applied=[],
               search_variations=variations_data,
               thresholds=response_data.get('search_parameters', {}).get('thresholds', {}),
               search_mode=search_mode,
               analysis_method=search_mode,
               selected_classes=response_data.get('search_parameters', {}).get('classes', [])
           )
           
           report_data = self.report_generator.generate_report(search_params, response_data)
           
           # Store report in cache
           cached_data['report_data'] = report_data
           return report_data
       
       except Exception as e:
           logger.error(f"Report generation error: {str(e)}")
           raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

   def handle_csv_download(self, response_data: Dict, trademark: str, search_mode: str, include_scores: bool = True) -> StreamingResponse:
       """Handle CSV format downloads"""
       try:
           filename = self.filename_generator.create_filename(trademark, 'csv', search_mode)
           content = self._generate_csv_content(response_data, include_scores)
           return self.create_streaming_response(content, filename, "text/csv")
       except Exception as e:
           logger.error(f"CSV download error: {str(e)}")
           raise HTTPException(status_code=500, detail=f"CSV generation failed: {str(e)}")
   
   def handle_json_download(self, response_data: Dict, trademark: str, search_mode: str, variations_data: Dict = None) -> StreamingResponse:
       """Handle JSON format downloads"""
       try:
           filename = self.filename_generator.create_filename(trademark, 'json', search_mode)
           content = self._generate_json_content(response_data, variations_data)
           return self.create_streaming_response(content, filename, "application/json")
       except Exception as e:
           logger.error(f"JSON download error: {str(e)}")
           raise HTTPException(status_code=500, detail=f"JSON generation failed: {str(e)}")
   
   def handle_txt_download(self, response_data: Dict, trademark: str, search_mode: str, include_scores: bool = True) -> StreamingResponse:
       """Handle TXT format downloads"""
       try:
           filename = self.filename_generator.create_filename(trademark, 'txt', search_mode)
           content = self._generate_txt_content(response_data, include_scores)
           return self.create_streaming_response(content, filename, "text/plain")
       except Exception as e:
           logger.error(f"TXT download error: {str(e)}")
           raise HTTPException(status_code=500, detail=f"TXT generation failed: {str(e)}")
   
   def handle_variations_download(self, response_data: Dict, variations_data: Dict, trademark: str, search_mode: str) -> StreamingResponse:
       """Handle word variations downloads"""
       try:
           filename = self.filename_generator.create_filename(trademark, 'variations', search_mode)
           content = self._generate_variations_content(variations_data, trademark)
           return self.create_streaming_response(content, filename, "text/plain")
       except Exception as e:
           logger.error(f"Variations download error: {str(e)}")
           raise HTTPException(status_code=500, detail=f"Variations generation failed: {str(e)}")
   
   def handle_report_download(self, report_data: Dict, response_data: Dict, trademark: str,
                            format_type: str, search_mode: str) -> StreamingResponse:
       """Handle multi-format report downloads using existing report generator"""
       try:
           markdown_content = report_data.get('report_content', '')
           if not markdown_content:
               raise ValueError("No markdown content found in report data")
           
           if format_type == "report_md":
               filename = self.filename_generator.create_filename(trademark, 'report_md', search_mode)
               return self.create_streaming_response(markdown_content, filename, "text/markdown")
           
           elif format_type == "report_docx":
               filename = self.filename_generator.create_filename(trademark, 'report_docx', search_mode)
               docx_content = self.document_converter.convert_markdown_to_docx(markdown_content, filename)
               return self.create_streaming_response(docx_content, filename, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                     
           else:
               raise ValueError(f"Unsupported report format: {format_type}")
               
       except Exception as e:
           logger.error(f"Report download error ({format_type}): {str(e)}")
           raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
   
   def handle_common_law_csv_download(self, results: Dict, trademark: str) -> StreamingResponse:
       """
       Generates and returns a downloadable CSV report of common law findings.
       """
       try:
           filename = self.filename_generator.create_filename(trademark=trademark, format_type='common_law_csv', report_type='raw-common')
           content = export_common_law_report_csv(results, trademark)
           return self.create_streaming_response(content, filename, "text/csv")
       except Exception as e:
           logger.error(f"Common law CSV download failed: {str(e)}")
           raise HTTPException(status_code=500, detail=f"Common law CSV generation failed: {str(e)}")
       
   def _generate_csv_content(self, response_data: Dict, include_scores: bool = True) -> str:
       """Generate CSV content from search results"""
       output = io.StringIO()
       writer = csv.writer(output)
       
       # Write header
       header = ['Serial Number', 'Mark', 'Owner', 'Status', 'Classes']
       if include_scores:
           header.extend(['Phonetic Score', 'Visual Score', 'Overall Score', 'Risk Level'])
       
       writer.writerow(header)
       
       # Write matches
       for match in response_data.get('matches', []):
           row = [
               match.get('serial_number', ''),
               match.get('mark_identification', ''),
               match.get('owner', ''),
               match.get('status_code', ''),
               ', '.join(map(str, match.get('nice_classes', [])))
           ]
           
           if include_scores:
               scores = match.get('similarity_scores', {})
               row.extend([
                   f"{(scores.get('phonetic', 0) * 100):.1f}%",
                   f"{(scores.get('visual', 0) * 100):.1f}%",
                   f"{(scores.get('overall', 0) * 100):.1f}%",
                   match.get('risk_level', 'unknown')
               ])
           
           writer.writerow(row)
       
       return output.getvalue()
   
   def _generate_json_content(self, response_data: Dict, variations_data: Dict = None) -> str:
       """Generate JSON content from search results, including variations if provided"""
       export_data = {
           "search_metadata": {
               "query_trademark": response_data.get('query_trademark', 'N/A'),
               "total_matches": response_data.get('total_matches', 0),
               "execution_time_ms": response_data.get('execution_time_ms', 0),
               "search_mode": response_data.get('search_mode', 'N/A'),
               "search_parameters": response_data.get('search_parameters', {}),
               "export_timestamp": datetime.now().isoformat()
           },
           "risk_assessment": response_data.get('risk_assessment', {}),
           "matches": response_data.get('matches', [])
       }
       if variations_data:
           export_data["variations_analysis"] = variations_data
       return json.dumps(export_data, indent=2, ensure_ascii=False)
   
   def _generate_txt_content(self, response_data: Dict, include_scores: bool = True) -> str:
       """Generate plain text content from search results"""
       output = io.StringIO()
       output.write(f"TRADEMARK SEARCH RESULTS\n========================\n\n")
       output.write(f"Query: {response_data.get('query_trademark', 'N/A')}\n")
       output.write(f"Total Matches: {response_data.get('total_matches', 0)}\n")
       output.write(f"Execution Time: {response_data.get('execution_time_ms', 0):.1f}ms\n\n")
       
       for i, match in enumerate(response_data.get('matches', []), 1):
           output.write(f"MATCH {i}\n---------\n")
           output.write(f"Mark: {match.get('mark_identification', 'N/A')}\n")
           output.write(f"Serial: {match.get('serial_number', 'N/A')}\n")
           output.write(f"Owner: {match.get('owner', 'N/A')}\n")
           output.write(f"Status: {match.get('status_code', 'N/A')}\n")
           output.write(f"Classes: {', '.join(map(str, match.get('nice_classes', [])))}\n")
           output.write(f"Risk Level: {match.get('risk_level', 'unknown').upper()}\n")
           
           if include_scores and match.get('similarity_scores'):
               scores = match['similarity_scores']
               output.write(f"Phonetic Score: {(scores.get('phonetic', 0) * 100):.1f}%\n")
               output.write(f"Visual Score: {(scores.get('visual', 0) * 100):.1f}%\n")
               output.write(f"Overall Score: {(scores.get('overall', 0) * 100):.1f}%\n")
           
           output.write("\n")
       return output.getvalue()
   
   def _generate_variations_content(self, variations_data: Dict, trademark: str) -> str:
       """Generate variations content"""
       output = io.StringIO()
       output.write(f"TRADEMARK VARIATIONS FOR: {trademark.upper()}\n{'=' * (25 + len(trademark))}\n\n")
       
       for category, variations in variations_data.items():
           if isinstance(variations, list) and variations:
               output.write(f"{category.upper().replace('_', ' ')}\n{'-' * len(category)}\n")
               for variation in variations:
                   output.write(f"  {variation}\n")
               output.write("\n")
       
       total_variations = sum(len(v) for v in variations_data.values() if isinstance(v, list))
       output.write(f"SUMMARY\n-------\n")
       output.write(f"Total variations generated: {total_variations}\n")
       output.write(f"Categories analyzed: {len([k for k, v in variations_data.items() if isinstance(v, list) and v])}\n")
       
       return output.getvalue()


# =============================================================================
# LIFESPAN AND DEPENDENCY INJECTION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
   """Manage application lifespan (startup and shutdown events)"""
   global auth_manager, app_config, db_manager, word_expansion_engine, trademark_analyzer, document_converter, download_handler
   
   logger.info("Starting Enhanced FastAPI Trademark Search System...")
   
   try:
       app_config = get_config()
       logger.info("Configuration loaded")
       
       if AUTH_AVAILABLE:
           auth_manager = AuthManager()
           user_store = UserStoreForSeparateDB()
           auth_manager.user_store = user_store
           await auth_manager.initialize()
           logger.info("Authentication system initialized")

       db_manager = DatabaseManager()
       await db_manager.initialize()
       logger.info("Database manager initialized")
       
       word_expansion_engine = WordExpansionEngine()
       logger.info("Word expansion engine initialized")
       
       trademark_analyzer = TrademarkAnalyzer(db_manager)
       logger.info("Trademark analyzer initialized")
       
       document_converter = DocumentConverter()
       filename_generator = FilenameGenerator()
       download_handler = DownloadHandler(filename_generator, document_converter)
       logger.info("Download handler and document converter initialized")
       
       if COMMON_LAW_AVAILABLE:
           initialize_common_law_system(None)
           logger.info("Common Law system initialized")

       logger.info("FastAPI Trademark Search System fully initialized")
       
       yield
       
   except Exception as e:
       logger.error(f"Failed to initialize application: {str(e)}")
       logger.exception("Full initialization error:")
       raise
   finally:
       logger.info("Shutting down Enhanced Trademark Search API...")
       if 'db_manager' in locals() and db_manager:
           await db_manager.close()
       if 'auth_manager' in locals() and auth_manager:
           await auth_manager.close()
       logger.info("Shutdown complete")

# =============================================================================
# DEPENDENCY FUNCTIONS
# =============================================================================

def validate_search_access(search_id: str, user_info: Dict[str, Any]) -> Dict[str, Any]:
   """Validate user access to search results"""
   if search_id not in search_results_cache:
       logger.error(f"Search ID not found in cache: {search_id}")
       raise HTTPException(status_code=404, detail="Search results not found or expired")
   
   cached_data = search_results_cache[search_id]
   
   if cached_data.get('user_id') != user_info.get('user_id'):
       logger.warning(f"Unauthorized access attempt to search {search_id} by user {user_info.get('username')}")
       raise HTTPException(status_code=403, detail="Access denied: search results belong to another user")
   
   if 'timestamp' in cached_data:
       cache_time = datetime.fromisoformat(cached_data['timestamp'])
       if datetime.now() - cache_time > timedelta(hours=CACHE_EXPIRY_HOURS):
           logger.info(f"Expired search results accessed: {search_id}")
           del search_results_cache[search_id]
           raise HTTPException(status_code=410, detail="Search results have expired")
   
   return cached_data


# =============================================================================
# FASTAPI APP INSTANCE AND MIDDLEWARE
# =============================================================================

app = FastAPI(
   title="Trademark Intelligence Platform",
   description="Comprehensive trademark business intelligence with multi-perspective analysis and multi-format reports",
   version="2.0.0",
   lifespan=lifespan
)

app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# =============================================================================
# API ROUTES
# =============================================================================

# -- UI and System Endpoints --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the new landing page"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(current_dir, "landing.html")
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Landing Page Not Found</h1>", status_code=404)

@app.get("/search", response_class=HTMLResponse)
async def search_page():
    """Serve the main application search interface"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(current_dir, "obvi_platform.html")
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>UI Not Found</h1>", status_code=404)

@app.get("/api")
async def api_status():
    """API status endpoint"""
    return {"status": "active", "version": "2.0.0"}

@app.get("/nice-classes")
async def get_nice_classes():
   """Get NICE class definitions from nice_classes.py"""
   try:
       from nice_classes import NICE_CLASS_COORDINATION
       classes_data = {}
       coordinated_mapping = {}
       for class_id, class_info in NICE_CLASS_COORDINATION.items():
           classes_data[class_id] = {
               "number": int(class_id),
               "type": class_info["type"],
               "description": class_info["description"],
               "title": class_info["title"],
               "coordinated": class_info.get("coordinated", []),
               "forcedCoordination": class_info.get("forcedCoordination", [])
           }
           if class_info.get("coordinated"):
               coordinated_mapping[class_id] = class_info["coordinated"]
       return {"classes": classes_data, "coordinated": coordinated_mapping}
   except ImportError as e:
       logger.error(f"Failed to import NICE classes: {str(e)}")
       raise HTTPException(status_code=500, detail="NICE classes data not available")

@app.get("/capabilities")
async def system_capabilities():
   """Get system capabilities and supported features"""
   return {
       "document_formats": {"markdown": True, "docx": document_converter.pandoc_available},
       "report_generator_available": REPORT_GENERATOR_AVAILABLE,
       "conversion_engine": "pypandoc" if document_converter.pandoc_available else "not_available",
       "supported_download_formats": ["csv", "json", "txt", "variations", "report_md"] + 
                                     (["report_docx"] if document_converter.pandoc_available else []),
       "filename_pattern": "trademark_[clean_trademark]_[type]_[timestamp].[ext]"
   }


# -- Authentication and User Endpoints ----------------------------------------

@app.post("/auth/login")
async def login(login_request: LoginRequest, response: Response):
   """Authenticate user and set a session cookie."""
   if not auth_manager:
       raise HTTPException(status_code=503, detail="Authentication service not available")
   
   try:
       auth_result = await auth_manager.authenticate_user(login_request.username, login_request.password)
       if not auth_result['success']:
           raise HTTPException(status_code=401, detail=auth_result['message'])
       
       token_result = await auth_manager.create_access_token(auth_result['user'])
       token = token_result['access_token']

       response.set_cookie(
           key="session_token",
           value=token,
           httponly=True,
           secure=app_config.environment != 'development',
           samesite="lax",
           max_age=token_result['expires_in']
       )
       return {"user_info": auth_result['user']}
   except HTTPException:
       raise
   except Exception as e:
       logger.error(f"Login error: {str(e)}")
       raise HTTPException(status_code=500, detail="Login failed")

@app.post("/auth/logout")
async def logout(response: Response, user_info: Dict[str, Any] = Depends(get_current_user_info)):
    """Logs out the user by clearing the session cookie."""
    try:
        session_id_from_token = user_info.get('token_jti')
        if session_id_from_token and auth_manager:
            await auth_manager.logout_user(session_id_from_token)
        response.delete_cookie("session_token")
        return {"success": True, "message": "Logged out successfully"}
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(status_code=500, detail="Logout failed")

@app.get("/questionnaire/questions", dependencies=[Depends(get_current_user_info)])
async def get_questionnaire_questions():
    """Returns the structured questions for the business context questionnaire"""
    try:
        if not QUESTIONNAIRE_AVAILABLE:
            raise ImportError("Questionnaire module not available")
        return QUESTIONNAIRE_QUESTIONS
    except Exception as e:
        logger.error(f"Unexpected error getting questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve questions: {str(e)}")

# --- Wayfinding Endpoints ---

@app.post("/wayfinding/start")
async def start_wayfinding_conversation(request: Request, user_info: Dict[str, Any] = Depends(get_current_user_info)):
    """Start a new wayfinding conversation"""
    try:
        data = await request.json()
        initial_context = data.get('initial_context', {})
        search_id = data.get('search_id')
        first_question = wayfinding_questionnaire.start_conversation(search_id, initial_context)
        logger.info(f"Wayfinding conversation started for search {search_id}")
        return {"question": first_question, "session_id": search_id, "success": True}
    except Exception as e:
        logger.error(f"Error starting wayfinding conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")

@app.post("/wayfinding/next")
async def wayfinding_next_question(request: Request, user_info: Dict[str, Any] = Depends(get_current_user_info)):
    """Process user answer and get next question or final results"""
    try:
        data = await request.json()
        answer = data.get('answer', '').strip()
        search_id = data.get('search_id')
        if not answer:
            raise HTTPException(status_code=400, detail="Answer is required")
        
        termination_phrases = ['stop', 'end', 'enough', 'generate report', 'stop the questionnaire']
        if any(phrase in answer.lower() for phrase in termination_phrases):
            final_summary = {
                "goods_services_description": "User requested early termination.",
                "trade_channels": "Not specified.", 
                "target_purchasers": "Not specified.",
                "termination_reason": "user_requested"
            }
            wayfinding_questionnaire.end_conversation(search_id)
            return {"question": json.dumps(final_summary), "conversation_ended": True, "termination_type": "forced"}
        
        next_question, injection_detected = wayfinding_questionnaire.post_answer_and_get_next_question(search_id, answer)
        if injection_detected:
            raise HTTPException(status_code=400, detail="Invalid input detected")
        
        conversation_ended = False
        try:
            json.loads(next_question)
            conversation_ended = True
            wayfinding_questionnaire.end_conversation(search_id)
        except json.JSONDecodeError:
            pass
        
        return {"question": next_question, "conversation_ended": conversation_ended, "termination_type": "natural" if conversation_ended else "continue"}
    except Exception as e:
        logger.error(f"Error in wayfinding next question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process answer: {str(e)}")

@app.post("/wayfinding/end")
async def end_wayfinding_conversation(request: Request, user_info: Dict[str, Any] = Depends(get_current_user_info)):
    """Explicitly end a wayfinding conversation and clean up resources"""
    try:
        data = await request.json()
        search_id = data.get('search_id')
        reason = data.get('reason', 'unknown')
        wayfinding_questionnaire.end_conversation(search_id)
        logger.info(f"Wayfinding conversation ended for search {search_id}, reason: {reason}")
        return {"success": True, "message": "Conversation ended successfully", "reason": reason}
    except Exception as e:
        logger.error(f"Error ending wayfinding conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to end conversation: {str(e)}")

# --- Questionnaire Analysis ---

@app.post("/questionnaire/analyze")
async def analyze_questionnaire(request: Dict[str, Any], user_info: Dict[str, Any] = Depends(get_current_user_info)):
    """Analyze questionnaire responses"""
    try:
        from llm_questionnaire import TrademarkQuestionnaire
        from nice_classes import NICE_CLASS_COORDINATION
        
        manager = TrademarkQuestionnaire()
        search_context = request.get('search_context', 'clearance')
        analysis = manager.process_responses(
            request.get('responses', {}),
            search_context=search_context,
            nice_class_data=NICE_CLASS_COORDINATION
        )
        return {
            'success': not analysis.get('error', False),
            'recommendations': analysis,
            'debug_info': {'processing_method': analysis.get('method_used', 'unknown')}
        }
    except Exception as e:
        logger.error(f"Questionnaire analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Questionnaire analysis failed: {str(e)}")


# -- Search Endpoints ---------------------------------------------------------

@app.post("/search/trademark")
async def trademark_search(request: Dict[str, Any], user_info: Dict[str, Any] = Depends(get_current_user_info)):
    """Performs the initial USPTO database search and caches the results."""
    if not trademark_analyzer:
        raise HTTPException(status_code=503, detail="Trademark Analyzer not initialized.")

    try:
        trademark = request.get('trademark', '').strip()
        if not trademark:
            raise HTTPException(status_code=400, detail="Trademark is required")
        
        classes = request.get('classes', [])
        is_all_classes_search = "all_classes" in classes
        if is_all_classes_search:
            classes = []
        if not classes and not is_all_classes_search:
            raise HTTPException(status_code=400, detail="At least one NICE class is required")
        
        results_data = await trademark_analyzer.perform_database_search(
            trademark=trademark,
            classes=classes,
            thresholds=request.get('thresholds', {"phonetic": 0.7, "visual": 0.76, "conceptual": 0.5}),
            search_mode=request.get('search_mode', 'basic'),
            use_variations=request.get('use_variations', True),
            enable_slang_search=request.get('enable_slang_search', False),
            max_results=request.get('max_results', 100)
        )
        
        search_id = f"search_{int(datetime.now().timestamp())}_{user_info.get('user_id', 'anon')}"
        search_results_cache[search_id] = {
            'response': results_data,
            'variations': trademark_analyzer.get_last_generated_variations(),
            'timestamp': datetime.now().isoformat(),
            'user_id': user_info.get('user_id'),
            'search_mode': request.get('search_mode', 'basic')
        }
        results_data['search_id'] = search_id
        return results_data
    except Exception as e:
        logger.error(f"Initial search failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Initial search failed: {str(e)}")

@app.post("/api/common-law/investigate")
async def common_law_investigation(request: Dict[str, Any], user_info: Dict[str, Any] = Depends(get_current_user_info)):
    """Performs the owner-centric, selection-based common law investigation."""
    if not trademark_analyzer:
        raise HTTPException(status_code=503, detail="Trademark Analyzer not initialized.")
    
    selected_marks = request.get('selected_marks')
    if not selected_marks or not isinstance(selected_marks, list):
        raise HTTPException(status_code=400, detail="A list of 'selected_marks' is required.")
    
    questionnaire_responses = request.get('questionnaire_responses')
    if questionnaire_responses is None:
        logger.info("Questionnaire not provided. Inferring context.")
        search_id = request.get('search_id')
        if not search_id or search_id not in search_results_cache:
            raise HTTPException(status_code=400, detail="A 'search_id' is required for inferred context.")
        
        cached_data = search_results_cache[search_id]
        selected_classes = cached_data.get('response', {}).get('search_parameters', {}).get('classes', [])
        inferred_industry = infer_industry_from_classes(selected_classes)
        questionnaire_responses = {
            "core_offering": f"Assumed industry: {inferred_industry}.",
            "target_market": "Not provided.", "brand_identity": "Not provided."
        }

    try:
        investigation_results = await trademark_analyzer.investigate_selected_marks(selected_marks, questionnaire_responses)
        
        search_id = request.get('search_id')
        if search_id and search_id in search_results_cache:
            search_results_cache[search_id]['common_law_results'] = investigation_results
            final_search_id = search_id
        else:
            final_search_id = f"common_law_{int(datetime.now().timestamp())}_{user_info.get('user_id', 'anon')}"
            search_results_cache[final_search_id] = {
                'response': {}, 'common_law_results': investigation_results, 'timestamp': datetime.now().isoformat(),
                'user_id': user_info.get('user_id'), 'search_mode': 'common_law_only'
            }

        return {
            "success": True,
            "investigation_results": investigation_results.get('investigation_results'),
            "overall_risk_summary": investigation_results.get('overall_risk_summary'),
            "search_id": final_search_id
        }
    except Exception as e:
        logger.error(f"Common law investigation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Investigation failed: {str(e)}")

@app.get("/api/common-law/download")
async def download_common_law_report(search_id: str = Query(...), user_info: Dict[str, Any] = Depends(get_current_user_info)):
    """Generates and returns a downloadable CSV report of common law findings."""
    cached_data = validate_search_access(search_id, user_info)
    results = cached_data.get('common_law_results')
    if not results or not results.get('investigation_results'):
        raise HTTPException(status_code=404, detail="Common law results not found or are empty.")
    try:
        first_mark = next(iter(results.get('investigation_results', {})), 'common_law_search')
        return download_handler.handle_common_law_csv_download(results, first_mark)
    except Exception as e:
        logger.error(f"Common law report download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/api/analytics/global", dependencies=[Depends(get_current_user_info)])
async def global_analytics_data():
    """Provides aggregated analytics data for the entire USPTO database."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database Manager not initialized.")
    try:
        analytics_results = await get_global_analytics(db_manager)
        if "error" in analytics_results:
            raise HTTPException(status_code=500, detail=analytics_results["error"])
        return analytics_results
    except Exception as e:
        logger.error(f"Global analytics endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve global analytics: {e}")

@app.get("/search/{search_id}/status")
async def get_search_status(search_id: str, user_info: Dict[str, Any] = Depends(get_current_user_info)):
    """Get search status and progress"""
    if search_id in search_results_cache:
        cached_data = search_results_cache[search_id]
        return {
            "search_id": search_id, "status": "completed", "timestamp": cached_data.get('timestamp'),
            "results_count": len(cached_data.get('response', {}).get('matches', []))
        }
    else:
        raise HTTPException(status_code=404, detail="Search not found")

# -- Download Endpoints -------------------------------------------------------

@app.get("/download/{search_id}")
async def download_search_results(
   search_id: str,
   format: str = Query("csv"),
   user_info: Dict[str, Any] = Depends(get_current_user_info)
):
   """Download search results in various formats."""
   logger.info(f"Download request: search_id={search_id}, format={format}, user={user_info.get('username')}")
   
   valid_formats = ["csv", "json", "txt", "variations", "report_md", "report_docx"]
   if format not in valid_formats:
       raise HTTPException(status_code=400, detail=f"Invalid format. Must be one of: {', '.join(valid_formats)}")
   
   if format == "report_docx" and not document_converter.pandoc_available:
       raise HTTPException(status_code=503, detail="DOCX conversion not available.")
   
   try:
       cached_data = validate_search_access(search_id, user_info)
       response_data = cached_data['response']
       variations_data = cached_data.get('variations', {})
       report_data = cached_data.get('report_data', {})
       trademark = response_data.get('query_trademark', 'trademark_search')
       search_mode = cached_data.get('search_mode', 'basic')
       
       if format == "csv":
           return download_handler.handle_csv_download(response_data, trademark, search_mode)
       elif format == "json":
           return download_handler.handle_json_download(response_data, trademark, search_mode, variations_data)
       elif format == "txt":
           return download_handler.handle_txt_download(response_data, trademark, search_mode)
       elif format == "variations":
           if not variations_data:
               raise HTTPException(status_code=404, detail="No variations data for this search.")
           return download_handler.handle_variations_download(response_data, variations_data, trademark, search_mode)
       elif format in ["report_md", "report_docx"]:
           if not report_data:
               report_data = download_handler.generate_report_if_needed(cached_data, search_id, search_mode)
           return download_handler.handle_report_download(report_data, response_data, trademark, format, search_mode)
       
   except Exception as e:
       logger.error(f"Download error for search {search_id}: {str(e)}", exc_info=True)
       raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

# -- Cache and Health Endpoints -----------------------------------------------

@app.get("/health")
async def health_check():
   """Health check endpoint with system capabilities"""
   return {
       "status": "healthy", "timestamp": datetime.now().isoformat(), "cache_size": len(search_results_cache),
       "document_conversion_available": document_converter.pandoc_available, "report_generator_available": REPORT_GENERATOR_AVAILABLE
   }

@app.get("/cache/stats", dependencies=[Depends(get_current_user_info)])
async def cache_statistics():
   """Get cache statistics"""
   return {"total_searches": len(search_results_cache), "cache_keys": list(search_results_cache.keys())}

@app.delete("/cache/{search_id}", dependencies=[Depends(get_current_user_info)])
async def clear_search_cache(search_id: str):
   """Clear specific search from cache"""
   if search_id in search_results_cache:
       del search_results_cache[search_id]
       return {"message": f"Cache cleared for search {search_id}"}
   else:
       raise HTTPException(status_code=404, detail="Search not found in cache")

@app.get("/formats")
async def get_supported_formats():
   """Get all supported download formats with descriptions"""
   base_formats = ["csv", "json", "txt", "variations", "report_md"]
   if document_converter.pandoc_available:
       base_formats.append("report_docx")
   return {"supported_formats": base_formats}


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
   """Custom HTTP exception handler with detailed logging"""
   logger.error(f"HTTP {exc.status_code}: {exc.detail} - Path: {request.url.path}")
   return JSONResponse(status_code=exc.status_code, content={"error": True, "message": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
   """General exception handler for unhandled errors"""
   logger.error(f"Unhandled exception: {str(exc)} - Path: {request.url.path}", exc_info=True)
   return JSONResponse(status_code=500, content={"error": True, "message": "Internal server error."})

# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
   logger.info("Starting Trademark Intelligence Platform...")
   
   def open_browser():
       time.sleep(2)
       webbrowser.open("http://localhost:8000")
   
   threading.Thread(target=open_browser, daemon=True).start()
   
   uvicorn.run(
       "5fastapi_tm_main:app", host="0.0.0.0", port=8000,
       reload=True, log_level="info"
   )