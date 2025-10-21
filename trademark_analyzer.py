#!/usr/bin/env python3
"""
Main Trademark Analysis Orchestrator
Manages the two-stage workflow:
1. Initial USPTO database search (with full basic/enhanced logic).
2. User-selected, owner-centric common law investigation.
"""

import time
import logging
import json
import re
from typing import Dict, List, Any, Set
from collections import defaultdict

# These imports are assumed to exist from the original file context
from database_manager import DatabaseManager
from word_expansion_engine import WordExpansionEngine, PhoneticEngine
from common_law_analyzer import CommonLawAnalyzer, CommonLawResult
from conceptual_slang_engine import SlangEngine

# NEW: Imports for advanced similarity calculation
import jellyfish
from difflib import SequenceMatcher

import pandas as pd

logger = logging.getLogger(__name__)

# REPLACED: The old simple SimilarityCalculator is replaced with the advanced one.
class SimilarityCalculator:
    """Calculates multi-factor trademark similarity."""

    def calculate_phonetic_similarity(self, mark1: str, mark2: str) -> float:
        """Calculate phonetic similarity using a blend of algorithms."""
        mark1_clean = self._clean_for_phonetic(mark1)
        mark2_clean = self._clean_for_phonetic(mark2)
        if not mark1_clean or not mark2_clean:
            return 0.0
        try:
            soundex_score = 1.0 if jellyfish.soundex(mark1_clean) == jellyfish.soundex(mark2_clean) else 0.0
            metaphone_score = 1.0 if jellyfish.metaphone(mark1_clean) == jellyfish.metaphone(mark2_clean) else 0.0
            nysiis_score = 1.0 if jellyfish.nysiis(mark1_clean) == jellyfish.nysiis(mark2_clean) else 0.0
            jaro_score = jellyfish.jaro_winkler_similarity(mark1_clean, mark2_clean)
            
            return (metaphone_score * 0.4) + (soundex_score * 0.2) + (nysiis_score * 0.2) + (jaro_score * 0.2)
        except Exception:
            return SequenceMatcher(None, mark1_clean, mark2_clean).ratio()

    def calculate_visual_similarity(self, mark1: str, mark2: str) -> float:
        """Calculate visual similarity based on string distance and structure."""
        mark1_norm = mark1.lower().strip()
        mark2_norm = mark2.lower().strip()
        try:
            jaro_score = jellyfish.jaro_winkler_similarity(mark1_norm, mark2_norm)
            lev_dist = jellyfish.levenshtein_distance(mark1_norm, mark2_norm)
            max_len = max(len(mark1_norm), len(mark2_norm))
            lev_score = (1.0 - (lev_dist / max_len)) if max_len > 0 else 0.0

            return (jaro_score * 0.6) + (lev_score * 0.4)
        except Exception:
            return SequenceMatcher(None, mark1_norm, mark2_norm).ratio()

    def calculate_conceptual_similarity(self, mark1: str, mark2: str) -> float:
        """Placeholder for conceptual similarity."""
        words1 = set(self._clean_for_semantic(mark1).split())
        words2 = set(self._clean_for_semantic(mark2).split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _clean_for_phonetic(self, text: str) -> str:
        return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

    def _clean_for_semantic(self, text: str) -> str:
        return re.sub(r'[^a-zA-Z\s]', ' ', text).lower().strip()

class RiskAssessmentEngine:
    """Encapsulates the multi-layered, legally-aware risk logic."""
    def __init__(self):
        self.status_map = {
            'live': ['REGISTERED', 'PUBLISHED', 'ALLOWED'],
            'pending': ['PENDING', 'NEW APPLICATION', 'EXAMINER'],
            'dead': ['ABANDONED', 'CANCELLED', 'EXPIRED']
        }

    def _get_status_category(self, status_description: str) -> str:
        """Determines status category based on reliable keywords in the full description."""
        status_upper = str(status_description).upper()
        if "ABANDONED" in status_upper or "CANCELLED" in status_upper or "EXPIRED" in status_upper:
            return 'dead'
        if "REGISTERED" in status_upper or "RENEWED" in status_upper or "PUBLISHED" in status_upper or "ALLOWED" in status_upper:
            return 'live'
        return 'pending'

    # REPLACED: This function now correctly uses the UI thresholds.
    def calculate_final_risk(self, match: Dict, thresholds: Dict) -> Dict:
        risk_breakdown = {}
        status_description = match.get('full_status_description', '')
        status_category = self._get_status_category(status_description)
        risk_breakdown['status_impact'] = f"Mark is '{status_category.upper()}'."
        
        scores = match.get('similarity_scores', {})
        overall_score = scores.get('overall', 0)

        phonetic_breach = scores.get('phonetic', 0) >= thresholds.get('phonetic', 0.7)
        visual_breach = scores.get('visual', 0) >= thresholds.get('visual', 0.7)
        conceptual_breach = scores.get('conceptual', 0) >= thresholds.get('conceptual', 0.6)

        breach_reasons = []
        if phonetic_breach: breach_reasons.append(f"phonetic similarity ({scores.get('phonetic', 0):.0%})")
        if visual_breach: breach_reasons.append(f"visual similarity ({scores.get('visual', 0):.0%})")
        if conceptual_breach: breach_reasons.append(f"conceptual similarity ({scores.get('conceptual', 0):.0%})")

        if breach_reasons:
            risk_breakdown['threshold_breach'] = f"Risk suggested by threshold breach in: {', '.join(breach_reasons)}."

        final_risk_level = 'low'
        if overall_score > 0.65:
            final_risk_level = 'medium'
        if overall_score > 0.8 or (phonetic_breach and status_category != 'dead'):
            final_risk_level = 'high'
        
        if status_category == 'dead' and final_risk_level == 'high':
            final_risk_level = 'medium'
            risk_breakdown['reasoning'] = "Similarity is high, but risk is capped at MEDIUM because the mark is dead."
        else:
            risk_breakdown['reasoning'] = f"Calculated '{final_risk_level}' risk based on overall similarity and threshold analysis."

        risk_breakdown['final_risk'] = final_risk_level
        return risk_breakdown

class TrademarkAnalyzer:
    """Orchestrates the entire trademark analysis workflow."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.similarity_calc = SimilarityCalculator()
        self.risk_engine = RiskAssessmentEngine()
        self.phonetic_engine = PhoneticEngine()
        self.slang_engine = SlangEngine()
        self._last_variations = {}

    async def perform_database_search(
        self,
        trademark: str,
        classes: List[str],
        thresholds: Dict[str, float],
        search_mode: str = "basic",
        search_context: str = "clearance", # NEW: Added search_context parameter
        max_results: int = 100,
        use_variations: bool = True,
        enable_slang_search: bool = False
    ) -> Dict[str, Any]:
        
        start_time = time.time()
        word_expansion_engine = WordExpansionEngine()

        text_variations = {trademark.upper()}
        self._last_variations = {}

        is_multi_term_search = '|' in trademark

        if is_multi_term_search:
            logger.info(f"Multi-term exact search detected for: {trademark}")
            search_terms_list = [term.strip().upper() for term in trademark.split('|') if term.strip()]
            text_variations.update(search_terms_list)
            use_variations = False
            enable_slang_search = False
            self._last_variations['exact_multi_term'] = search_terms_list
        
        if use_variations:
            logger.info(f"Generating variations for '{trademark}' in {search_context} context.")
            # ALTERED: Logic now diverges based on the search context
            if search_context == 'knockout':
                variation_types = ['phonetic'] # Only high-confidence phonetics for knockout
                enable_slang_search = False # Disable slang for knockout
            else: # Default to clearance
                variation_types = ['phonetic', 'visual', 'morphological', 'conceptual']
            
            variations_dict = await word_expansion_engine.generate_variations(trademark, variation_types=variation_types)
            for var_list in variations_dict.values():
                text_variations.update(v.upper() for v in var_list)
            self._last_variations = variations_dict
        elif not is_multi_term_search:
            logger.info(f"Skipping standard variation generation for '{trademark}' as requested.")

        if enable_slang_search:
            logger.info(f"Generating slang variations for '{trademark}'.")
            slang_terms = await self.slang_engine.generate_slang_variations(trademark, limit=25)
            if slang_terms:
                logger.info(f"Adding {len(slang_terms)} slang variations to the search.")
                text_variations.update(s.upper() for s in slang_terms)
                self._last_variations['slang'] = slang_terms

        search_terms = list(text_variations)
        if len(search_terms) > 250:
            logger.warning(f"Generated {len(search_terms)} variations, limiting to 250 for database query.")
            limited_search_terms = search_terms[:250]
        else:
            limited_search_terms = search_terms

        if search_mode == 'basic' or is_multi_term_search:
            candidates = await self.db_manager.search_trademarks_basic(limited_search_terms, max_results * 2)
        else:
            candidates = await self.db_manager.search_trademarks_advanced(limited_search_terms, max_results)
        
        # ANNOTATION: fixed 10162025 --> switched the following two lines
        # 1. Call the scoring function FIRST to define 'final_matches'.
        final_matches = self._score_and_filter_candidates(trademark, candidates, thresholds)
        
        # 2. NOW it is safe to call the metrics function.
        metrics_data = self._calculate_results_metrics(final_matches)

        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000

        return {
            'query_trademark': trademark,
            'total_matches': len(final_matches),
            'matches': final_matches,
            'metrics': metrics_data, # NEW: Add metrics to the return payload.
            'execution_time_ms': execution_time_ms,
            'search_parameters': {
                'classes': classes,
                'search_mode': search_mode,
                'thresholds': thresholds
            }
        }

    # REPLACED: This function now performs the full multi-factor scoring.
    def _score_and_filter_candidates(self, trademark: str, candidates: List[Dict], thresholds: Dict) -> List[Dict]:
        processed_matches = []
        trademark_upper = trademark.strip().upper()

        for cand in candidates:
            mark_text = cand.get('mark_identification', '')
            if not mark_text:
                continue
            
            mark_upper = mark_text.strip().upper()

            if mark_upper == trademark_upper:
                p_score, v_score = 1.0, 1.0
                c_score = self.similarity_calc.calculate_conceptual_similarity(trademark, mark_text)
            else:
                p_score = self.similarity_calc.calculate_phonetic_similarity(trademark, mark_text)
                v_score = self.similarity_calc.calculate_visual_similarity(trademark, mark_text)
                c_score = self.similarity_calc.calculate_conceptual_similarity(trademark, mark_text)
            
            overall_score = (p_score * 0.4) + (v_score * 0.4) + (c_score * 0.2)

            cand['similarity_scores'] = {
                'phonetic': p_score,
                'visual': v_score,
                'conceptual': c_score,
                'overall': min(1.0, overall_score)
            }

            risk_assessment = self.risk_engine.calculate_final_risk(cand, thresholds)
            cand['risk_level'] = risk_assessment.get('final_risk', 'unknown')
            cand['risk_analysis'] = risk_assessment

            processed_matches.append(cand)
            
        risk_order = {'high': 0, 'medium': 1, 'low': 2, 'dead': 3}
        
        processed_matches.sort(key=lambda m: (
            risk_order.get(m['risk_level'], 4),
            -m['similarity_scores']['overall']
        ))
        
        return processed_matches

    def _calculate_results_metrics(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates aggregate metrics for a given list of search results."""
        if not matches:
            return {}
        try:
            df = pd.DataFrame(matches)
            df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')

            metrics = {
                "total_matches": len(df),
                "high_risk_count": len(df[df['risk_level'] == 'high']),
                "medium_risk_count": len(df[df['risk_level'] == 'medium']),
                "low_risk_count": len(df[df['risk_level'] == 'low']),
                "average_score": df['similarity_scores'].apply(lambda x: x.get('overall', 0)).mean(),
                "filings_by_year": df[df['filing_date'].notna()]['filing_date'].dt.year.value_counts().sort_index().to_dict(),
                "status_distribution": df['risk_analysis'].apply(lambda x: x.get('status_impact', 'Unknown').replace("Mark is '", "").replace("'.", "")).value_counts().to_dict()
            }
            # Convert numpy types to standard Python types for JSON serialization
            for key, value in metrics.items():
                if isinstance(value, dict):
                    metrics[key] = {str(k): int(v) for k, v in value.items()}
                elif pd.api.types.is_numeric_dtype(type(value)):
                    metrics[key] = float(value)
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {"error": "Failed to calculate metrics."}

    async def investigate_selected_marks(
        self, selected_marks: List[Dict[str, Any]], questionnaire_responses: Dict[str, Any]
    ) -> Dict[str, Any]:
        investigation_targets = self._extract_unique_mark_owner_pairs(selected_marks)
        if not investigation_targets:
            logger.info("No valid targets for common law investigation.")
            return {}
        
        logger.info(f"Initiating common law investigation for {len(investigation_targets)} pairs with context.")

        async with CommonLawAnalyzer() as common_law_analyzer:
            common_law_results_by_key = await common_law_analyzer.comprehensive_search(
                investigation_targets, questionnaire_responses
            )

        synthesized_results = self._synthesize_results(selected_marks, common_law_results_by_key)
        
        return {
            "success": True,
            "investigation_results": synthesized_results,
            "overall_risk_summary": self._analyze_common_law_risk(synthesized_results)
        }
    
    def _extract_unique_mark_owner_pairs(self, selected_marks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        unique_pairs = set()
        for mark in selected_marks:
            owner = (mark.get('owner') or '').strip().upper()
            mark_text = (mark.get('mark_identification') or '').strip()
            if owner and mark_text:
                unique_pairs.add((mark_text, owner))
        return [{"mark": mark_text, "owner": owner} for mark_text, owner in unique_pairs]

    def _synthesize_results(self, selected_marks: List[Dict[str, Any]], common_law_results_by_key: Dict[str, List[CommonLawResult]]) -> Dict[str, Dict[str, Any]]:
        final_report = defaultdict(lambda: defaultdict(lambda: {"uspto_marks": [], "common_law_findings": []}))
        
        for mark in selected_marks:
            mark_text = (mark.get('mark_identification') or 'UNKNOWN MARK').strip()
            owner = (mark.get('owner') or 'UNKNOWN OWNER').strip().upper()
            final_report[mark_text][owner]["uspto_marks"].append(mark)

        for key, findings in common_law_results_by_key.items():
            try:
                mark_text, owner = key.split("|", 1)
                final_report[mark_text][owner]["common_law_findings"].extend([f.__dict__ for f in findings])
            except (ValueError, KeyError):
                logger.warning(f"Could not parse or place common law result key: {key}")
                continue

        return json.loads(json.dumps(final_report))

    def _analyze_common_law_risk(self, synthesized_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        risk_counts = defaultdict(int)
        total_conflicts = 0
        
        for mark_data in synthesized_results.values():
            for owner_data in mark_data.values():
                for finding in owner_data.get("common_law_findings", []):
                    risk_counts[finding.get("risk_level", "unknown")] += 1
                    if finding.get("status") == "found":
                        total_conflicts += 1

        overall_risk = "low"
        if risk_counts["high"] > 0:
            overall_risk = "high"
        elif risk_counts["medium"] > 0:
            overall_risk = "medium"

        return {
            "overall_risk": overall_risk,
            "total_conflicts": total_conflicts,
            "high_risk_conflicts": risk_counts["high"],
            "medium_risk_conflicts": risk_counts["medium"],
            "low_risk_conflicts": risk_counts["low"]
        }
        
    def get_last_generated_variations(self) -> Dict[str, List[str]]:
        return self._last_variations