#!/usr/bin/env python3
"""
Owner-Centric Common Law Trademark Investigator
Performs targeted searches based on owner names extracted from primary search results.
"""

import asyncio
import aiohttp
import re
import logging
import time
import random
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from collections import defaultdict

import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from llm_integration import OllamaClient


logger = logging.getLogger(__name__)

@dataclass
class CommonLawSearchConfig:
    """Configuration for common law searches"""
    request_delay_min: float = 1.0
    request_delay_max: float = 3.0
    concurrent_requests: int = 3
    domain_tlds: List[str] = field(default_factory=lambda: [
        ".com", ".net", ".org", ".co", ".us", ".io", ".biz", ".info"
    ])
    social_platforms: List[str] = field(default_factory=lambda: [
        "facebook.com", "instagram.com", "twitter.com", "linkedin.com",
        "youtube.com", "tiktok.com", "pinterest.com"
    ])
    business_directories: List[str] = field(default_factory=lambda: [
        "yellowpages.com", "yelp.com", "bbb.org", "owler.com",
        "glassdoor.com", "dnb.com", "maps.google.com"
    ])

@dataclass
class CommonLawResult:
    """A single common law finding for a specific investigation target."""
    source_type: str
    source_name: str
    finding: str
    url: Optional[str] = None
    summary: Optional[str] = None
    mark_found_on_site: bool = False
    status: str = "found" # Can be 'found', 'not_found', 'error'
    owner_info: Optional[str] = None
    risk_level: str = "unknown"
    confidence_score: float = 0.0
    # ANNOTATION: New field to hold the contextual analysis note.
    analyst_note: Optional[str] = None

class CommonLawAnalyzer:
    """
    Performs an intelligent, multi-stage common law investigation that includes
    website verification and social media corroboration.
    """
    
    def __init__(self, config: Optional[CommonLawSearchConfig] = None):
        load_dotenv(dotenv_path='2.env')
        self.config = config or CommonLawSearchConfig()
        self.session = None
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        ]
        self.llm_client = OllamaClient()
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.search_engine_id = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")
        # ANNOTATION: List of social media sites for targeted corroboration searches.
        self.social_sites = [
            "linkedin.com", "twitter.com", "facebook.com", "instagram.com",
            "pinterest.com", "etsy.com", "blueskyweb.xyz", "truthsocial.com"
        ]
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.config.concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def comprehensive_search(
        self, investigation_targets: List[Dict[str, str]], questionnaire_responses: Dict[str, Any]
    ) -> Dict[str, List[CommonLawResult]]:
        """
        Main entry point for the owner-centric investigation.
        """
        logger.info(f"Initiating INTELLIGENT common law investigation for {len(investigation_targets)} targets.")
        
        final_results_by_key = {}
        
        # NEW: Ensure every target has an entry in the final results dict from the start. 10142025
        for target in investigation_targets:
            key = f"{target['mark']}|{target['owner'].strip().upper()}"
            if key not in final_results_by_key:
                final_results_by_key[key] = []
        #/////

        for target in investigation_targets:
            owner_name = target['owner'].strip().upper()
            mark_text = target['mark']
            
            verified_candidates = await self._intelligent_investigation(owner_name, mark_text)
            
            findings = []
#            for candidate in verified_candidates:
#                findings.append(CommonLawResult(
#                    source_type="verified_website",
#                    source_name=urlparse(candidate['link']).netloc,
#                    finding=candidate.get('ai_summary', 'AI summary not available.'),
#                    url=candidate.get('link'),
#                    summary=candidate.get('ai_summary'),
#                    status="found",
#                    owner_info=candidate.get('ai_identified_company', owner_name),
#                    risk_level="high" if candidate.get('ai_company_match') else "medium",
#                    confidence_score=candidate.get('final_score', 0) / 200.0,
#                    analyst_note=f"Social Media Presence: {', '.join(candidate.get('social_media_found', ['None']))}"
#                ))

            for candidate in verified_candidates:
                # Add the primary website finding
                findings.append(CommonLawResult(
                    source_type="verified_website",
                    source_name=urlparse(candidate.get('link', '')).netloc,
                    finding=candidate.get('ai_summary', 'AI summary not available.'),
                    url=candidate.get('link'),
                    status="found" if candidate.get('ai_company_match') else "uncertain",
                    risk_level="high" if candidate.get('ai_company_match') else "medium"
                ))
                # Add all social media findings (both 'found' and 'not_found')
                findings.extend(candidate.get('social_media_findings', []))
                # NEW: Add all business directory findings to the final report.
                findings.extend(candidate.get('business_directory_findings', []))
            result_key = f"{mark_text}|{owner_name}"
            final_results_by_key[result_key] = findings
        
        return final_results_by_key
        
    async def _intelligent_investigation(self, business_name: str, trademark: str) -> List[Dict]:
        """
        Executes the full multi-step investigation workflow.
        """
        # --- PHASE 1 & 2: DISCOVERY AND INITIAL RANKING ---
        ranked_candidates = await self._discover_and_rank_leads(business_name, trademark)
        if not ranked_candidates: return []

        # --- PHASE 3: DEEP CONTENT ANALYSIS ---
        top_candidates = ranked_candidates[:7]
        analysis_tasks = [self._analyze_candidate(candidate, business_name) for candidate in top_candidates]
        analyzed_candidates = await asyncio.gather(*analysis_tasks)

        # --- PHASE 4: SOCIAL MEDIA and BUSINESS DIRECTORY CORROBORATION (NEW) ---
        corroboration_tasks = []
        for c in analyzed_candidates:
            if c.get('ai_company_match'):
                corroboration_tasks.append(self._find_social_media_presence(c, trademark))
                # NEW: Add the business directory search to the asynchronous tasks
                corroboration_tasks.append(self._find_business_directory_presence(c, trademark))
        
        if corroboration_tasks:
            await asyncio.gather(*corroboration_tasks)

        # --- PHASE 5: FINAL RE-RANKING ---
        final_ranked = self._rerank_based_on_analysis(analyzed_candidates)
        return final_ranked[:3]

    async def _discover_and_rank_leads(self, business_name: str, trademark: str) -> List[Dict]:
        """Combines discovery and initial ranking steps."""
        discovery_queries = [
            f'"{business_name}"', f'"{trademark}" official website',
            f'site:linkedin.com/company/ "{trademark}" OR "{business_name}"',
        ]
        all_results, seen_links = [], set()
        for query in discovery_queries:
            for result in self._perform_google_search(query):
                if result.get('link') and result['link'] not in seen_links:
                    all_results.append(result)
                    seen_links.add(result['link'])
        
        return self._score_and_rank_candidates(all_results, business_name, trademark)

    def _score_and_rank_candidates(self, results: List[Dict], business_name: str, trademark: str) -> List[Dict]:
        """Performs the initial, heuristic-based ranking."""
        scored_results = []
        clean_biz_name = business_name.lower().replace('corp','').strip()
        clean_trademark = trademark.lower()
        BLACKLISTED_DOMAINS = ['bizprofile.net', 'importinfo.com', 'bpiworld.org', 'greenpeople.org']

        for result in results:
            score = 0
            link = result.get('link','').lower()
            title = result.get('title','').lower()
            domain = urlparse(link).netloc.replace('www.','')
            
            if any(blacklisted in domain for blacklisted in BLACKLISTED_DOMAINS):
                score = -999
            else:
                if domain.split('.')[0] == clean_trademark: score += 200
                if business_name.lower() in title: score += 60
            
            result['initial_score'] = score
            scored_results.append(result)

        return sorted([r for r in scored_results if r['initial_score'] > 0], key=lambda x: x['initial_score'], reverse=True)

    async def _analyze_candidate(self, candidate: Dict, target_business: str) -> Dict:
        """Fetches website content and gets an AI summary."""
        url = candidate.get('link')
        html_content = await self._fetch_content(url) if url else None
        
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            page_text = soup.get_text(separator=' ', strip=True)
            
            loop = asyncio.get_event_loop()
            analysis = await loop.run_in_executor(None, self.llm_client.verify_and_summarize_content, page_text, target_business)
            candidate.update(analysis)
        else:
            candidate['ai_summary'] = "Failed to fetch content."
            candidate['ai_company_match'] = False
        
        return candidate

    # ANNOTATION: This is the new Phase 4 function for social media corroboration.
    async def _find_social_media_presence(self, candidate: Dict, trademark: str) -> Dict:
        """Performs targeted social media searches for a verified candidate."""
        company_name = candidate.get('ai_identified_company', trademark)
        # --- START OF NEW/ALTERED CODE ---
        # ALTERED: This list will now hold complete CommonLawResult objects for an audit trail.
        social_findings = []

        for site in self.social_sites:
            query = f'site:{site} "{company_name}" "{trademark}"'
            results = self._perform_google_search(query, num_results=1)
            
            if results:
                # Create a "found" result object
                social_findings.append(CommonLawResult(
                    source_type="social_media",
                    source_name=site.split('.')[0].title(),
                    finding=f"Potential presence found for '{company_name}'.",
                    url=results[0].get('link'),
                    status="found",
                    risk_level="low" # Social media presence is informational, not a direct conflict.
                ))
            else:
                # Create a "not_found" result object for the audit trail
                social_findings.append(CommonLawResult(
                    source_type="social_media",
                    source_name=site.split('.')[0].title(),
                    finding=f"No clear presence found for '{company_name}'.",
                    status="not_found",
                    risk_level="low"
                ))
        
        # Storing the detailed findings objects instead of just a list of names
        candidate['social_media_findings'] = social_findings
        # --- END OF NEW/ALTERED CODE ---
        return candidate

    async def _find_business_directory_presence(self, candidate: Dict, trademark: str) -> Dict:
        """Performs targeted business directory searches."""
        company_name = candidate.get('ai_identified_company', trademark)
        directory_findings = []

        for site in self.config.business_directories:
            query = f'site:{site} "{company_name}"'
            results = self._perform_google_search(query, num_results=1)
            
            if results:
                directory_findings.append(CommonLawResult(
                    source_type="business_directory",
                    source_name=site.split('.')[0].title(),
                    finding=f"Potential listing found for '{company_name}'.",
                    url=results[0].get('link'),
                    status="found",
                    risk_level="medium"
                ))
            else:
                directory_findings.append(CommonLawResult(
                    source_type="business_directory",
                    source_name=site.split('.')[0].title(),
                    finding=f"No listing found for '{company_name}'.",
                    status="not_found",
                    risk_level="low"
                ))
        
        candidate['business_directory_findings'] = directory_findings
        return candidate

    def _rerank_based_on_analysis(self, analyzed_candidates: List[Dict]) -> List[Dict]:
        """Re-ranks candidates based on AI analysis and social media presence."""
        for candidate in analyzed_candidates:
            final_score = candidate.get('initial_score', 0)
            if candidate.get('ai_company_match'):
                final_score += 100
            else:
                final_score /= 4
            
            # ANNOTATION: Add a score bonus for each social media platform found.
            social_bonus = len(candidate.get('social_media_found', [])) * 10
            final_score += social_bonus
            candidate['final_score'] = final_score

        return sorted(analyzed_candidates, key=lambda x: x['final_score'], reverse=True)

    # --- HELPER METHODS ---
    def _perform_google_search(self, query: str, num_results: int = 5) -> list:
        if not self.api_key or not self.search_engine_id: return []
        try:
            service = build("customsearch", "v1", developerKey=self.api_key)
            response = service.cse().list(q=query, cx=self.search_engine_id, num=num_results).execute()
            return [{"title": i.get('title'), "link": i.get('link'), "snippet": i.get('snippet')} for i in response.get('items', [])]
        except Exception:
            return []

    async def _fetch_content(self, url: str) -> Optional[str]:
        if not url: return None
        try:
            async with self.session.get(url, timeout=15, allow_redirects=True, headers={'User-Agent': random.choice(self.user_agents)}) as response:
                if response.status == 200:
                    return await response.text()
        except Exception:
            pass
        return None