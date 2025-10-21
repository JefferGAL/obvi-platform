"""
Streamlined LLM Integration for Trademark and Patent Analysis Platform
Focused on search analysis, conflict assessment, and IP expertise
"""
import logging
import requests
import json
import ollama
import time
import re

import asyncio

from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Any


logger = logging.getLogger(__name__)

class OllamaClient:
    """LLM client for trademark and patent analysis"""
    
    def __init__(self, api_url="http://localhost:11434/api"):
        self.api_url = api_url
        self.model = self._select_preferred_model()
        self.ip_expertise = {
            "trademark": {
                "role": "a trademark law expert",
                "instructions": """
                - Refer to trademark classes as "NICE Classes" where appropriate
                - Distinguish between common law rights and federal registration
                - Reference USPTO's TESS database for trademark searches
                - Consider likelihood of confusion factors: similarity of marks, relatedness of goods/services, channels of trade
                - Assess mark strength (generic, descriptive, suggestive, arbitrary, fanciful)
                """
            },
            "patent": {
                "role": "a patent law expert",
                "instructions": """
                - Distinguish between utility, design, and plant patents
                - Reference MPEP guidelines for patent examination
                - Explain prior art significance to patentability
                - Consider novelty, non-obviousness, and utility requirements
                """
            },
            "ip_general": {
                "role": "an intellectual property expert",
                "instructions": """
                - Distinguish between trademarks, patents, and copyrights
                - Explain different IP protection mechanisms
                - Consider international IP considerations
                - Assess portfolio strategy and risk management
                """
            }
        }

    def _select_preferred_model(self) -> str:
        """Select best available model for IP analysis"""
        try:
            models = self.list_models()
            preferred = ["gemma2", "llama3", "mistral", "gemma", "llava"]
            
            for model in preferred:
                if any(model in available.lower() for available in models):
                    return next(available for available in models if model in available.lower())
            
            return models[0] if models else "llama2"
        except:
            return "llama2"
    
    def check_connection(self) -> bool:
        """Check if Ollama is accessible"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except:
            return []
    
    def query(self, prompt: str, temperature: float = 0.3) -> str:
        """Query the LLM with IP-focused prompt"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            response = requests.post(f"{self.api_url}/generate", json=payload, timeout=45)
            
            if response.status_code == 200:
                return response.json().get('response', 'No response received')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return f"Error: API request failed"
            
        except Exception as e:
            logger.error(f"Error querying Ollama: {str(e)}")
            return f"Error: {str(e)}"

#    def analyze_trademark_questionnaire(self, responses: Dict[str, Any], business_classes_map: Dict[str, Any], nice_class_data: Dict = None) -> Dict[str, Any]:
    def analyze_trademark_questionnaire(self, responses: Dict[str, Any], business_classes_map: Dict[str, Any], search_context: str = "clearance", nice_class_data: Dict = None) -> Dict[str, Any]:
        """Analyze trademark questionnaire for business type and search recommendations"""
        
        trademark = responses.get('trademark_name', '')
        business_desc = responses.get('core_offering', '')
        brand_identity = responses.get('brand_identity', '')
        target_market = responses.get('target_market', '')
        expansion_plans = responses.get('expansion_plans', '')
        
        # Build NICE class descriptions for the prompt
        class_info = ""
        if nice_class_data:
            class_info = "\n".join([
                f"Class {num}: {info['title']} - {info['description'][:150]}"
                for num, info in sorted(nice_class_data.items())[:20]  # First 20 to avoid token limits
            ])
        
        business_types_str = ", ".join(business_classes_map.keys())
        
        # NEW: Define context-specific instructions for the AI based on the search type.
        if search_context == 'knockout':
            class_instruction = "Identify ONLY the most critical and directly competing NICE classes (1-2 maximum). Do not include related or coordinated classes."
        else: # clearance
            class_instruction = "Identify a comprehensive list of all relevant NICE classes (up to 5). Include related and coordinated classes that a professional would consider for a full clearance search."

        prompt = f"""
        You are an AI trademark assistant analyzing a business for NICE classification.

        Business Name: "{trademark}"
        Business Description: {business_desc}
        Brand Identity: {brand_identity}
        Target Market: {target_market}
        Expansion Plans: {expansion_plans}

        Available NICE Classes (International Trademark Classification):
        {class_info}

        Based ONLY on the business description and the provided NICE Class descriptions, perform the following tasks:
        1.  {class_instruction}
        2.  From this list ({business_types_str}), select the single most relevant business type.
        3.  Provide a brief, one-sentence reasoning for your class selection.

        Respond with ONLY a JSON object in the following format. Do not use example values; generate them from your analysis.
        {{
            "business_type": "[the single best business type you selected]",
            "suggested_nice_classes": [array_of_integer_class_numbers],
            "search_mode": "enhanced",
            "phonetic_threshold": 0.7,
            "visual_threshold": 0.7,
            "conceptual_threshold": 0.6,
            "risk_level": "medium",
            "reasoning": "[your one-sentence justification for the class selection]"
        }}
        """
        
        response = self.query(prompt, temperature=0.2)
        return self._parse_json_response(response)

    def extract_key_sentences(self, web_text: str, max_length: int = 15000) -> str:
        """Performs the extractive step of the hybrid summary."""
        if not web_text:
            return ""
            
        prompt = f"""
        You are an expert data extraction bot. From the following website text, extract the 3 to 7 most important sentences that describe the company's purpose, products, or services. Ignore boilerplate language, navigation, and footers. Respond ONLY with the extracted sentences, each on a new line.

        WEBSITE TEXT:
        "{web_text[:max_length]}"
        """
        return self.query(prompt, temperature=0.0)

    def generate_abstractive_summary(self, key_sentences: str) -> str:
        """Performs the abstractive step of the hybrid summary."""
        if not key_sentences:
            return "No key sentences provided for summary."
            
        prompt = f"""
        As an expert analyst, write a concise, one to two-sentence abstractive summary based on the following key sentences from a website. Synthesize the information into a fluent description of the business's primary purpose. If the sentences indicate the domain is parked or for sale, state that.

        KEY SENTENCES:
        "{key_sentences}"
        """
        summary = self.query(prompt, temperature=0.3)
        return summary.strip().replace('"', '')

    # ANNOTATION: This is the new, context-aware analysis method.
    def analyze_website_for_conflict(self, web_text: str, questionnaire: Dict[str, Any], max_length: int = 5000) -> Dict[str, Any]:
        """
        Analyzes website text in the context of a user's business to determine risk.
        """
        if not web_text or len(web_text.strip()) < 50:
            return {
                "summary": "Page has minimal text content (e.g., parked domain or under construction).",
                "risk_level": "low",
                "analyst_note": "The domain appears to be inactive or parked, posing a low immediate risk."
            }

        core_offering = questionnaire.get('core_offering', 'a business')
        target_market = questionnaire.get('target_market', 'the general public')

        prompt = f"""
        You are an expert trademark paralegal. Your task is to analyze scraped website text for a potential trademark conflict.

        CONTEXT:
        My client's business is about: "{core_offering}"
        My client's target market is: "{target_market}"

        INSTRUCTIONS:
        Analyze the following WEBSITE TEXT. Based *only* on the text provided, determine if the business described operates in a similar or related industry to my client's.
        
        Respond with ONLY a JSON object in the following format. Do not include any other text or explanations.
        {{
          "summary": "A concise, one-sentence summary of the website's business purpose.",
          "risk_level": "Choose one: 'high', 'medium', or 'low'. 'high' for identical industries, 'medium' for related industries, 'low' for completely unrelated industries.",
          "analyst_note": "A brief, one-sentence explanation for your risk assessment, comparing the website's business to my client's context."
        }}

        WEBSITE TEXT:
        "{web_text[:max_length]}"
        """
        
        response_str = self.query(prompt, temperature=0.1)
        analysis = self._parse_json_response(response_str)

        # Fallback if JSON parsing fails or returns an invalid structure
        if not analysis or "risk_level" not in analysis:
            return {
                "summary": self.generate_abstractive_summary(web_text[:1000]),
                "risk_level": "unknown",
                "analyst_note": "Could not perform contextual risk analysis. Manual review required."
            }
        return analysis
    
    def generate_ai_analyst_summary(self, risk_data: Dict[str, Any]) -> str:
        """
        Takes structured risk data and generates a fluent, narrative summary for reports.
        """
        mark_text = risk_data.get('mark_identification', 'the conflicting mark')
        risk_level = risk_data.get('risk_level', 'unknown').upper()
        status_impact = risk_data.get('risk_analysis', {}).get('status_impact', '')
        threshold_breach = risk_data.get('risk_analysis', {}).get('threshold_breach', '')
        
        prompt = f"""
        As a trademark paralegal, write a concise, one-paragraph summary of a potential trademark conflict for a report.
        Based on the following structured data, explain the key risk factors in a professional and objective tone.

        DATA:
        - Conflicting Mark: "{mark_text}"
        - Final Risk Level: {risk_level}
        - Status Analysis: "{status_impact}"
        - Similarity Analysis: "{threshold_breach}"

        Synthesize this data into a fluent summary.
        """
        summary = self.query(prompt, temperature=0.3)
        return summary.strip()
    # ANNOTATION: This is the new, powerful summarization and verification method.
    # It takes raw webpage text and extracts structured data for analysis, which is
    # the core of the new "Deep Content Analysis & Verification" phase.
    def verify_and_summarize_content(self, web_text: str, target_business: str, max_length: int = 8000) -> Dict[str, Any]:
        """
        Analyzes webpage text to verify the company name and summarize its purpose.
        Falls back to basic HTML parsing if the LLM fails.
        """
        if not web_text or len(web_text.strip()) < 100:
            return {
                "identified_company": "N/A",
                "summary": "Page has minimal text content.",
                "is_match": False
            }

        prompt = f"""
        You are an expert business analyst. Analyze the following text from a webpage.
        My target company is "{target_business}".

        Perform two tasks:
        1. Identify the primary company name mentioned or described in the text.
        2. Provide a concise, one-sentence summary of the business's main purpose or product.

        Respond with ONLY a JSON object in the following format. Do not include any other text.
        {{
          "identified_company": "The company name you found in the text",
          "summary": "A one-sentence summary of the business."
        }}

        WEBSITE TEXT:
        "{web_text[:max_length]}"
        """
        
        analysis = {}
        try:
            response_str = self.query(prompt, temperature=0.1)
            analysis = self._parse_json_response(response_str)
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}. Proceeding with fallback.")
            analysis = {}

        # ANNOTATION: If the LLM analysis failed or returned an invalid structure, use the fallback.
        if not analysis or "identified_company" not in analysis:
            logger.warning(f"Using Beautiful Soup fallback for '{target_business}'.")
            analysis = self._fallback_summary_with_soup(web_text, target_business)
        
        # Determine if the identified company name is a plausible match for our target
        found_company = analysis.get("identified_company", "").lower()
        target_company = target_business.lower()
        
        is_match = (target_company in found_company or 
                    found_company in target_company or
                    target_company.replace('corp', '').strip() in found_company)

        analysis['is_match'] = is_match
        return analysis

    # ANNOTATION: This is the new fallback function using Beautiful Soup.
    def _fallback_summary_with_soup(self, web_text: str, target_business: str) -> Dict[str, any]:
        """
        Parses HTML with Beautiful Soup to extract basic info when the LLM fails.
        """
        try:
            soup = BeautifulSoup(web_text, 'html.parser')
            
            title = soup.title.string.strip() if soup.title and soup.title.string else "No title found."
            
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            description = desc_tag['content'].strip() if desc_tag and desc_tag.get('content') else ""

            # Manual check for company name in title
            identified_company = title if target_business.lower() in title.lower() else "Could not identify company name."
            
            summary_parts = [title]
            if description:
                summary_parts.append(description)
                
            # Add the user-requested note about manual date verification.
            manual_search_note = "NOTE: Perform a manual domain age check to determine the precise date of first use for relevant content."
            summary_parts.append(manual_search_note)
            
            return {
                "identified_company": identified_company,
                "summary": " | ".join(summary_parts),
            }
        except Exception as e:
            logger.error(f"Beautiful Soup fallback failed: {e}")
            return {
                "identified_company": "Analysis Failed",
                "summary": "Fallback HTML parsing also failed. Manual review required.",
            }

    # ANNOTATION: This is the new, required method for handling multi-message conversations.
    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Sends a list of messages to the Ollama chat endpoint for a conversational response.
        """
        if not self.model:
            raise ConnectionError("LLM model not available. Please check your Ollama installation.")
        
        try:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, so we can't use asyncio.run()
                # Instead, use the synchronous ollama.chat directly
                response = ollama.chat(model=self.model, messages=messages)
                return response
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                response = asyncio.run(asyncio.to_thread(
                    ollama.chat,
                    model=self.model,
                    messages=messages
                ))
                return response
        except Exception as e:
            print(f"Failed to get chat response from LLM: {e}")
            # Return a structured error message
            return {
                'message': {
                    'content': 'Error communicating with the language model.'
                }
            }

    def _parse_json_response(self, response: str) -> Dict[str, Any]:

        """Parse JSON from LLM response with fallback"""
        try:
            # Enhanced regex to find JSON even with leading/trailing text
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {} # Return empty dict on no match
        except Exception as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            return {}
    
    def _create_fallback_analysis(self) -> Dict[str, Any]:
        """Create fallback analysis when LLM parsing fails"""
        return {
            "suggested_nice_classes": [35, 42],
            "search_mode": "enhanced",
            "phonetic_threshold": 0.7,
            "visual_threshold": 0.7,
            "conceptual_threshold": 0.6,
            "risk_level": "medium",
            "reasoning": "Fallback analysis applied - manual review recommended for optimal class selection"
        }

class TrademarkAnalysisEngine:
    """Main engine for trademark analysis using LLM"""
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.llm = ollama_client or OllamaClient()
        self.is_available = self.llm.check_connection() if self.llm else False
    
#    def analyze_questionnaire(self, responses: Dict[str, Any], business_classes_map: Dict[str, Any], nice_class_data: Dict = None) -> Dict[str, Any]:
    def analyze_questionnaire(self, responses: Dict[str, Any], business_classes_map: Dict[str, Any], search_context: str = "clearance", nice_class_data: Dict = None) -> Dict[str, Any]:
        """Analyze questionnaire responses"""
        if not self.is_available:
            return self._rule_based_fallback(responses, business_classes_map, nice_class_data)
                
        try:
            return self.llm.analyze_trademark_questionnaire(responses, business_classes_map, nice_class_data)
        except Exception as e:
            logger.error(f"LLM questionnaire analysis failed: {str(e)}")
            return self._rule_based_fallback(responses, business_classes_map, nice_class_data)
    
    def analyze_conflicts(self, trademark: str, search_results: List[Dict]) -> str:
        """Analyze search results for conflicts"""
        if not self.is_available:
            return self._basic_conflict_analysis(trademark, search_results)
        
        try:
            return self.llm.analyze_search_results(trademark, search_results)
        except Exception as e:
            logger.error(f"LLM conflict analysis failed: {str(e)}")
            return self._basic_conflict_analysis(trademark, search_results)
    
    # Rationale: The fallback method now takes the business_classes_map as an argument.
    # It finds the most relevant business type from the map and uses the corresponding
    # classes for its recommendation, making it more intelligent than a simple hardcoded
    # fallback. It also uses the same helper functions as the main process_responses method,
    # reducing code duplication.
    #def _rule_based_fallback(self, responses: Dict[str, Any], business_classes_map: Dict[str, Any]) -> Dict[str, Any]:
    def _rule_based_fallback(self, responses: Dict[str, Any], business_classes_map: Dict[str, Any], nice_class_data: Dict = None) -> Dict[str, Any]:
        """Rule-based analysis when LLM unavailable"""
        
        trademark = responses.get('trademark_name', responses.get('business_info', {}).get('trademark_name', ''))
        business_desc = responses.get('business_info', {}).get('core_offering', '').lower()
        brand_identity = responses.get('business_info', {}).get('brand_identity', '').lower()
        
        best_match_type = 'general_business'
        max_matches = 0
        combined_text = f"{business_desc} {brand_identity}".lower()
        
        for b_type, b_classes in business_classes_map.items():
            keywords = b_type.split('_')
            match_count = sum(1 for keyword in keywords if keyword in combined_text)
            if match_count > max_matches:
                max_matches = match_count
                best_match_type = b_type
                
        suggested_classes = business_classes_map.get(best_match_type, [35, 42])
        
        search_mode = self._determine_search_mode(trademark, brand_identity)
        thresholds = self._calculate_thresholds(trademark, business_desc)
        risk_level = self._assess_initial_risk(trademark, business_desc, suggested_classes)
        
        return {
            "business_type": best_match_type,
            "suggested_nice_classes": suggested_classes,
            "search_mode": search_mode,
            "phonetic_threshold": thresholds['phonetic'],
            "visual_thresholds": thresholds['visual'],
            "conceptual_thresholds": thresholds['conceptual'],
            "risk_level": risk_level,
            "reasoning": f"Intelligent fallback analysis for '{trademark}' based on detected business type: {best_match_type}. LLM unavailable."
        }
    
    def _basic_conflict_analysis(self, trademark: str, results: List[Dict]) -> str:
        """Basic conflict analysis without LLM"""
        
        if not results:
            return f"No similar trademarks found for '{trademark}'. This suggests lower conflict risk, but comprehensive clearance should include common law searches and professional review."
        
        high_risk = [r for r in results if r.get('similarity_scores', {}).get('overall', 0) > 0.8]
        live_marks = [r for r in results if 'REGISTERED' in r.get('status_code', '').upper()]
        
        analysis = f"TRADEMARK CONFLICT ANALYSIS FOR '{trademark}':\n\n"
        analysis += f"Found {len(results)} similar marks in database.\n"
        analysis += f"High similarity matches: {len(high_risk)}\n"
        analysis += f"Live/registered marks: {len(live_marks)}\n\n"
        
        if high_risk:
            analysis += "HIGH RISK CONFLICTS:\n"
            for mark in high_risk[:3]:
                analysis += f"- '{mark.get('mark_identification')}' (Status: {mark.get('status_code')})\n"
        
        analysis += "\n TL/DR: Professional trademark clearance advised given the similar marks found."
        
        return analysis
