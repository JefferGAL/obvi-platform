#!/usr/bin/env python3
"""
Enhanced Application Configuration Module
Includes comprehensive NICE class system with forced 9↔42 coordination
"""

import os
import logging
import secrets
from typing import Dict, List, Any

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    raise ImportError("Please install pydantic-settings: pip install pydantic-settings")

from functools import lru_cache

logger = logging.getLogger(__name__)

try:
    from nice_classes import NICE_CLASS_COORDINATION
except ImportError:
    logger.warning("Could not import NICE_CLASS_COORDINATION from nice_classes.py")
#    return selected_classes

# Leet Speak and Visual Variations for Enhanced Search
LEET_SPEAK_MAPPING = {
    'a': ['4', '@', 'A', '/-\\'],
    'b': ['6', 'B', '8', '|3'],
    'c': ['(', '<', '[', '{'],
    'd': ['|)', '|]'],
    'e': ['3', 'E'],
    'f': ['ph', '|='],
    'g': ['9', 'G'],
    'h': ['#', '|-|', '][', '}{'],
    'i': ['1', '!', 'I', 'l', '|'],
    'j': [']'],
    'k': ['|<', '|{'],
    'l': ['1', 'I', 'L', '|'],
    'm': ['M', 'n', '/\\/\\', '|\\/|'],
    'n': ['N', 'm', '/\\/', '|\\|'],
    'o': ['0', 'O', '()'],
    'p': ['|o'],
    'q': ['9', '()_'],
    'r': ['|2'],
    's': ['5', '$', 'S'],
    't': ['7', 'T', '+'],
    'u': ['U', 'v', '|_|', '(_)'],
    'v': ['V', 'u', '\\/'],
    'w': ['W', 'vv', '\\/\\/'],
    'x': ['><', '}{'],
    'y': ['`/', 'j'],
    'z': ['2', 'Z']
}

VISUAL_SIMILARITY_MAPPING = {
    'o': ['0', 'q', 'p', 'd'], '0': ['o', 'O', 'Q'],
    'i': ['l', '1', '!', 'I'], 'l': ['i', 'I', '1'],
    'n': ['m', 'h'], 'm': ['n', 'h'], 'h': ['n', 'm'],
    'q': ['p', 'g', 'o'], 'p': ['q', 'b', 'd'],
    'b': ['d', 'p', '6'], 'd': ['b', 'p', 'q'],
    'u': ['v', 'n'], 'v': ['u', 'y'], 'y': ['v', 'i'],
    'w': ['m', 'vv'], 'rn': ['m'], 'cl': ['d']
}



#///// adding business intelligence to the classes


# Business-to-NICE Class Mapping for AI Analysis
BUSINESS_TYPE_TO_NICE_CLASSES = {
    "technology": {
        "primary": [9, 42],
        "secondary": [35, 38],
        "keywords": ["software", "app", "platform", "digital", "online", "web", "mobile", "tech", "ai", "data", "cloud", "saas", "api", "system"]
    },
    "ecommerce": {
        "primary": [35, 42],
        "secondary": [9, 39],
        "keywords": ["ecommerce", "marketplace", "retail", "store", "shop", "selling", "online", "platform", "marketplace"]
    },
    "healthcare": {
        "primary": [44, 5],
        "secondary": [10, 42],
        "keywords": ["health", "medical", "wellness", "fitness", "healthcare", "therapeutic", "clinical", "pharma", "device"]
    },
    "food_beverage": {
        "primary": [43, 29, 30],
        "secondary": [35, 39],
        "keywords": ["food", "restaurant", "catering", "beverage", "drink", "nutrition", "recipe", "cooking", "dining"]
    },
    "education": {
        "primary": [41, 42],
        "secondary": [9, 35],
        "keywords": ["education", "training", "course", "learning", "school", "teaching", "academy", "university"]
    },
    "finance": {
        "primary": [36, 42],
        "secondary": [35, 9],
        "keywords": ["finance", "investment", "banking", "payment", "money", "financial", "fintech", "crypto", "trading"]
    },
    "entertainment": {
        "primary": [41, 9],
        "secondary": [42, 35],
        "keywords": ["game", "entertainment", "media", "content", "streaming", "video", "music", "gaming", "social"]
    },
    "manufacturing": {
        "primary": [7, 40],
        "secondary": [6, 17],
        "keywords": ["manufacture", "production", "factory", "industrial", "machinery", "equipment", "materials"]
    },
    "consulting": {
        "primary": [35, 42],
        "secondary": [41, 45],
        "keywords": ["consulting", "advisory", "strategy", "management", "professional", "services", "agency"]
    },
    "fashion": {
        "primary": [25, 18],
        "secondary": [35, 26],
        "keywords": ["fashion", "clothing", "apparel", "accessories", "jewelry", "beauty", "cosmetics", "style"]
    }
}

# Status Code Definitions for Enhanced Risk Assessment
USPTO_STATUS_CODES = {
    # Active/Live Status Codes (HIGH RISK)
    "REGISTERED": {"risk_level": "high", "description": "Active registered trademark", "category": "live"},
    "PUBLISHED FOR OPPOSITION": {"risk_level": "high", "description": "Published for opposition, likely to register", "category": "live"},
    "NOTICE OF ALLOWANCE": {"risk_level": "high", "description": "Approved, awaiting registration", "category": "live"},
    "USE APPLICATION FILED": {"risk_level": "medium", "description": "Use-based application filed", "category": "live"},
    "INTENT TO USE APPLICATION FILED": {"risk_level": "medium", "description": "Intent-to-use application filed", "category": "live"},
    
    # Pending Status Codes (MEDIUM RISK)
    "NEW APPLICATION FILED": {"risk_level": "medium", "description": "Recently filed application", "category": "pending"},
    "ASSIGNED TO EXAMINER": {"risk_level": "medium", "description": "Under examination", "category": "pending"},
    "NON-FINAL ACTION MAILED": {"risk_level": "medium", "description": "Office action issued", "category": "pending"},
    "RESPONSE AFTER NON-FINAL ACTION": {"risk_level": "medium", "description": "Response filed to office action", "category": "pending"},
    "FINAL ACTION MAILED": {"risk_level": "low", "description": "Final rejection issued", "category": "pending"},
    
    # Dead/Inactive Status Codes (LOW RISK)
    "ABANDONED": {"risk_level": "very_low", "description": "Application abandoned", "category": "dead"},
    "CANCELLED": {"risk_level": "very_low", "description": "Registration cancelled", "category": "dead"},
    "EXPIRED": {"risk_level": "very_low", "description": "Registration expired", "category": "dead"},
    "DEEMED ABANDONED": {"risk_level": "very_low", "description": "Deemed abandoned by USPTO", "category": "dead"},
    "WITHDRAWN": {"risk_level": "very_low", "description": "Application withdrawn", "category": "dead"},
    
    # Special Cases
    "SUSPENDED": {"risk_level": "low", "description": "Suspended pending resolution", "category": "suspended"},
    "INTERFERENCE": {"risk_level": "medium", "description": "In interference proceeding", "category": "disputed"},
    "OPPOSITION": {"risk_level": "medium", "description": "In opposition proceeding", "category": "disputed"}
}

def analyze_business_for_nice_classes(business_description: str, max_classes: int = 5) -> List[int]:
    """Analyze business description and suggest NICE classes"""
    
    business_lower = business_description.lower()
    suggested_classes = set()
    confidence_scores = {}
    
    # Check each business type
    for business_type, info in BUSINESS_TYPE_TO_NICE_CLASSES.items():
        # Count keyword matches
        keyword_matches = sum(1 for keyword in info["keywords"] if keyword in business_lower)
        
        if keyword_matches > 0:
            # Calculate confidence based on keyword density
            confidence = keyword_matches / len(info["keywords"])
            
            # Add primary classes with high confidence
            if confidence > 0.1:  # At least 10% keyword match
                for class_id in info["primary"]:
                    suggested_classes.add(class_id)
                    confidence_scores[class_id] = confidence_scores.get(class_id, 0) + confidence
            
            # Add secondary classes with lower threshold
            if confidence > 0.05:  # At least 5% keyword match
                for class_id in info["secondary"]:
                    suggested_classes.add(class_id)
                    confidence_scores[class_id] = confidence_scores.get(class_id, 0) + confidence * 0.5
    
    # Sort by confidence and return top classes
    sorted_classes = sorted(suggested_classes, key=lambda x: confidence_scores.get(x, 0), reverse=True)
    
    return sorted_classes[:max_classes]

def get_status_risk_assessment(status_code: str) -> Dict[str, Any]:
    """Get risk assessment for a status code"""
    
    status_info = USPTO_STATUS_CODES.get(status_code.upper(), {
        "risk_level": "unknown",
        "description": f"Unknown status: {status_code}",
        "category": "unknown"
    })
    
    return {
        "status_code": status_code,
        "risk_level": status_info["risk_level"],
        "description": status_info["description"],
        "category": status_info["category"],
        "recommendation": _get_status_recommendation(status_info["category"], status_info["risk_level"])
    }

def _get_status_recommendation(category: str, risk_level: str) -> str:
    """Get recommendation based on status category and risk level"""
    
    if category == "live" and risk_level == "high":
        return "Strong conflict risk. Consider different mark or seek legal advice."
    elif category == "live" and risk_level == "medium":
        return "Potential conflict. Monitor application progress and consider alternatives."
    elif category == "pending":
        return "Application in process. Monitor for outcome and assess conflict risk."
    elif category == "dead":
        return "Low conflict risk. Dead marks generally don't prevent registration."
    elif category == "disputed":
        return "Uncertain outcome. Monitor proceeding and seek legal guidance."
    else:
        return "Unknown status. Manual review recommended."

# Add to AppConfig class (insert this property method)
def get_business_classes(self, business_description: str) -> List[int]:
    """Get suggested NICE classes for business description"""
    return analyze_business_for_nice_classes(business_description)

def assess_status_risk(self, status_code: str) -> Dict[str, Any]:
    """Assess risk based on trademark status"""
    return get_status_risk_assessment(status_code)

# Add this to your config_app_config.py file after the existing NICE class definitions

# Enhanced USPTO Status Codes for Risk Assessment
USPTO_STATUS_RISK_MAPPING = {
    # HIGH RISK - Active/Live Marks
    "REGISTERED": {"risk": "very_high", "category": "live", "weight": 1.0},
    "PUBLISHED FOR OPPOSITION": {"risk": "high", "category": "live", "weight": 0.9},
    "NOTICE OF ALLOWANCE": {"risk": "high", "category": "live", "weight": 0.85},
    "REGISTERED AND RENEWED": {"risk": "very_high", "category": "live", "weight": 1.0},
    
    # MEDIUM RISK - Pending Applications
    "NEW APPLICATION FILED": {"risk": "medium", "category": "pending", "weight": 0.6},
    "ASSIGNED TO EXAMINER": {"risk": "medium", "category": "pending", "weight": 0.55},
    "NON-FINAL ACTION MAILED": {"risk": "medium", "category": "pending", "weight": 0.5},
    "RESPONSE AFTER NON-FINAL ACTION": {"risk": "medium", "category": "pending", "weight": 0.6},
    "FINAL ACTION MAILED": {"risk": "low", "category": "pending", "weight": 0.3},
    
    # LOW RISK - Dead/Inactive
    "ABANDONED": {"risk": "very_low", "category": "dead", "weight": 0.1},
    "CANCELLED": {"risk": "very_low", "category": "dead", "weight": 0.1},
    "EXPIRED": {"risk": "very_low", "category": "dead", "weight": 0.1},
    "DEEMED ABANDONED": {"risk": "very_low", "category": "dead", "weight": 0.1},
    "WITHDRAWN": {"risk": "very_low", "category": "dead", "weight": 0.1},
    
    # SPECIAL CASES
    "SUSPENDED": {"risk": "low", "category": "suspended", "weight": 0.2},
    "OPPOSITION": {"risk": "medium", "category": "disputed", "weight": 0.4},
    "INTERFERENCE": {"risk": "medium", "category": "disputed", "weight": 0.4}
}

# Likelihood of Confusion Factors for LLM Analysis
LIKELIHOOD_OF_CONFUSION_FACTORS = {
    "mark_similarity": {
        "weight": 0.4,
        "description": "Similarity in appearance, sound, and meaning",
        "thresholds": {"high": 0.8, "medium": 0.6, "low": 0.4}
    },
    "goods_services_relatedness": {
        "weight": 0.3,
        "description": "Relatedness of goods/services in marketplace",
        "nice_class_relationships": {
            "identical": 1.0,
            "coordinated": 0.8,
            "related": 0.6,
            "unrelated": 0.2
        }
    },
    "trade_channels": {
        "weight": 0.15,
        "description": "Similarity of marketing channels and consumers",
        "factors": ["retail", "online", "b2b", "consumer", "professional"]
    },
    "mark_strength": {
        "weight": 0.1,
        "description": "Distinctiveness of the prior mark",
        "levels": {
            "fanciful": 1.0,
            "arbitrary": 0.9,
            "suggestive": 0.7,
            "descriptive": 0.4,
            "generic": 0.1
        }
    },
    "actual_confusion": {
        "weight": 0.05,
        "description": "Evidence of actual confusion in marketplace",
        "default": 0.0
    }
}

def calculate_conflict_risk_score(mark_similarity: float, nice_class_overlap: str, 
                                status_code: str, mark_strength: str = "arbitrary") -> Dict[str, Any]:
    """Calculate comprehensive conflict risk score"""
    
    # Get status risk
    status_info = USPTO_STATUS_RISK_MAPPING.get(status_code.upper(), {
        "risk": "unknown", "category": "unknown", "weight": 0.5
    })
    
    # Calculate base similarity score
    similarity_score = mark_similarity * LIKELIHOOD_OF_CONFUSION_FACTORS["mark_similarity"]["weight"]
    
    # Add goods/services relatedness
    relatedness_weights = LIKELIHOOD_OF_CONFUSION_FACTORS["goods_services_relatedness"]["nice_class_relationships"]
    relatedness_score = relatedness_weights.get(nice_class_overlap, 0.5) * \
                       LIKELIHOOD_OF_CONFUSION_FACTORS["goods_services_relatedness"]["weight"]
    
    # Add mark strength factor
    strength_weights = LIKELIHOOD_OF_CONFUSION_FACTORS["mark_strength"]["levels"]
    strength_score = strength_weights.get(mark_strength, 0.7) * \
                    LIKELIHOOD_OF_CONFUSION_FACTORS["mark_strength"]["weight"]
    
    # Calculate composite score
    base_score = similarity_score + relatedness_score + strength_score
    
    # Apply status modifier
    final_score = base_score * status_info["weight"]
    
    # Determine risk level
    if final_score >= 0.8:
        risk_level = "very_high"
    elif final_score >= 0.6:
        risk_level = "high"
    elif final_score >= 0.4:
        risk_level = "medium"
    elif final_score >= 0.2:
        risk_level = "low"
    else:
        risk_level = "very_low"
    
    return {
        "risk_score": final_score,
        "risk_level": risk_level,
        "status_category": status_info["category"],
        "status_weight": status_info["weight"],
        "components": {
            "similarity": similarity_score,
            "relatedness": relatedness_score,
            "mark_strength": strength_score,
            "status_modifier": status_info["weight"]
        },
        "recommendation": _get_risk_recommendation(risk_level, status_info["category"])
    }

def _get_risk_recommendation(risk_level: str, status_category: str) -> str:
    """Get specific recommendation based on risk level and status"""
    
    if risk_level == "very_high" and status_category == "live":
        return "HIGH CONFLICT RISK: Strong likelihood of confusion with active mark. Consider alternative mark or seek legal counsel."
    elif risk_level == "high" and status_category == "live":
        return "SIGNIFICANT RISK: Notable similarity to active mark. Professional clearance search recommended."
    elif risk_level == "medium":
        return "MODERATE RISK: Some similarity detected. Monitor status and consider modifications to reduce conflict."
    elif risk_level == "low" and status_category == "dead":
        return "LOW RISK: Similar mark found but inactive. Generally acceptable for registration."
    else:
        return "VARIABLE RISK: Manual review recommended for comprehensive assessment."

def get_coordinated_classes(selected_classes: List[str], force_9_42: bool = True) -> List[str]:
    """
    Get coordinated classes with FORCED 9↔42 coordination
    """

    
    all_classes = set(selected_classes)
    
    # FORCED 9↔42 coordination - ALWAYS applied
    if force_9_42:
        if any(c in ["9", "009"] for c in all_classes):
            all_classes.update(["9", "42", "009", "042"])
        if any(c in ["42", "042"] for c in all_classes):
            all_classes.update(["9", "42", "009", "042"])
    
    # Apply standard coordination from nice_classes.py data
    for class_id in list(all_classes):
        normalized_id = str(int(class_id)) if class_id.isdigit() else class_id
        
        class_info = NICE_CLASS_COORDINATION.get(normalized_id, {})
        coordinated = class_info.get("coordinated", [])
        forced_coord = class_info.get("forcedCoordination", [])
        
        # Add coordinated classes
        for coord_class in coordinated + forced_coord:
            all_classes.add(str(coord_class))
            all_classes.add(str(coord_class).zfill(3))  # Add zero-padded version
    
    # Normalize all class IDs to 3-digit format
    normalized_classes = []
    for class_id in all_classes:
        if str(class_id).isdigit():
            normalized_classes.append(str(class_id).zfill(3))
        else:
            normalized_classes.append(str(class_id))
    
    return sorted(list(set(normalized_classes)))

# Enhanced business-to-NICE class mapping with confidence scores
ENHANCED_BUSINESS_MAPPING = {
    # Technology (highest confidence coordination)
    "software_development": {"classes": [9, 42], "confidence": 0.95, "coordination": "mandatory"},
    "mobile_applications": {"classes": [9, 42], "confidence": 0.95, "coordination": "mandatory"},
    "saas_platform": {"classes": [9, 42, 35], "confidence": 0.9, "coordination": "mandatory"},
    "artificial_intelligence": {"classes": [9, 42], "confidence": 0.95, "coordination": "mandatory"},
    "cybersecurity": {"classes": [9, 42, 45], "confidence": 0.9, "coordination": "mandatory"},
    
    # E-commerce and Business
    "ecommerce_platform": {"classes": [35, 42, 9], "confidence": 0.85, "coordination": "recommended"},
    "digital_marketing": {"classes": [35, 42], "confidence": 0.8, "coordination": "recommended"},
    "business_consulting": {"classes": [35, 42], "confidence": 0.75, "coordination": "optional"},
    
    # Healthcare and Medical
    "medical_devices": {"classes": [10, 44, 5], "confidence": 0.9, "coordination": "recommended"},
    "telemedicine": {"classes": [44, 9, 42], "confidence": 0.85, "coordination": "recommended"},
    "pharmaceutical": {"classes": [5, 44, 42], "confidence": 0.9, "coordination": "recommended"},
    
    # Food and Hospitality
    "food_delivery": {"classes": [43, 39, 42], "confidence": 0.8, "coordination": "recommended"},
    "restaurant_chain": {"classes": [43, 35, 29, 30], "confidence": 0.85, "coordination": "recommended"},
    
    # Financial Technology
    "fintech": {"classes": [36, 9, 42], "confidence": 0.9, "coordination": "mandatory"},
    "cryptocurrency": {"classes": [36, 9, 42], "confidence": 0.85, "coordination": "mandatory"},
    "payment_processing": {"classes": [36, 9, 42], "confidence": 0.9, "coordination": "mandatory"}
}

# Add these methods to your AppConfig class
def get_enhanced_risk_assessment(self, mark_similarity: float, class_overlap: str, 
                               status: str) -> Dict[str, Any]:
    """Get enhanced risk assessment with detailed factors"""
    return calculate_conflict_risk_score(mark_similarity, class_overlap, status)

def get_business_class_mapping(self, business_type: str) -> Dict[str, Any]:
    """Get enhanced business-to-class mapping with confidence"""
    return ENHANCED_BUSINESS_MAPPING.get(business_type.lower(), {
        "classes": [35, 42],
        "confidence": 0.5,
        "coordination": "optional"
    })
#/////08132025/////

def generate_leet_variations(text: str, max_variations: int = 20) -> List[str]:
    """Generate leet speak variations"""
    variations = set()
    text_lower = text.lower()
    
    # Single character substitutions
    for char, replacements in LEET_SPEAK_MAPPING.items():
        if char in text_lower:
            for replacement in replacements:
                variations.add(text_lower.replace(char, replacement))
    
    # Multiple character substitutions (limited)
    if len(text) <= 8:
        variations.update(_recursive_leet_substitution(text_lower, 0, 2))
    
    return list(variations)[:max_variations]

def _recursive_leet_substitution(word: str, pos: int, max_subs: int) -> set:
    """Recursively generate leet substitutions"""
    if max_subs <= 0 or pos >= len(word):
        return {word}
    
    variations = set()
    char = word[pos]
    
    # Try substitutions for current character
    if char in LEET_SPEAK_MAPPING:
        for replacement in LEET_SPEAK_MAPPING[char]:
            new_word = word[:pos] + replacement + word[pos+1:]
            variations.update(_recursive_leet_substitution(new_word, pos + 1, max_subs - 1))
    
    # Continue without substitution
    variations.update(_recursive_leet_substitution(word, pos + 1, max_subs))
    
    return variations

class AppConfig(BaseSettings):
    """Enhanced configuration with comprehensive NICE class support"""
    
    # Application settings
    app_name: str = Field(default="Obvi Trademark Search API")
    app_version: str = Field(default="2.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    
    # API settings
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=1)
    
    # Database settings
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5432)
    db_name: str = Field(default="uspto_trademarks")
    db_username: str = Field(default="postgres")
    db_password: str = Field(default="")
    auth_db_name: str = Field(default="trademark_auth")
    create_auth_db: bool = Field(default=True)
    db_read_only_mode: bool = Field(default=True)
    db_allow_schema_creation: bool = Field(default=False)
    db_min_connections: int = Field(default=5)
    db_max_connections: int = Field(default=20)
    db_connection_timeout: int = Field(default=30)
    db_query_timeout: int = Field(default=60)
    db_max_retries: int = Field(default=3)
    
    # Security settings
    jwt_secret_key: str = Field(default="")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_hours: int = Field(default=24)
    session_timeout_minutes: int = Field(default=30)
    max_concurrent_sessions: int = Field(default=3)
    rate_limit_per_minute: int = Field(default=100)
    rate_limit_burst: int = Field(default=150)
    failed_login_threshold: int = Field(default=5)
    lockout_duration_minutes: int = Field(default=15)
    allowed_origins: str = Field(default="http://localhost:3000,http://localhost:8080")
    
    # Enhanced Search settings
    default_phonetic_threshold: float = Field(default=0.7)
    default_visual_threshold: float = Field(default=0.7)
    default_conceptual_threshold: float = Field(default=0.6)
    max_search_results: int = Field(default=1000)
    default_batch_size: int = Field(default=1000)
    enable_phonetic_expansion: bool = Field(default=True)
    enable_visual_expansion: bool = Field(default=True)
    enable_morphological_expansion: bool = Field(default=True)
    enable_conceptual_expansion: bool = Field(default=True)
    enable_leet_speak_detection: bool = Field(default=True)
    enable_ai_analysis: bool = Field(default=False)
    ai_model_name: str = Field(default="llama2")
    ai_api_timeout: int = Field(default=30)
    enable_search_cache: bool = Field(default=True)
    cache_ttl_minutes: int = Field(default=60)
    max_variations_per_type: int = Field(default=50)
    
    # NICE Class settings
    force_9_42_coordination: bool = Field(default=True)
    enable_class_coordination: bool = Field(default=True)
    max_coordinated_classes: int = Field(default=20)

    # Coordination settings 08112025
    enable_optional_coordination: bool = Field(default=True, description="Enable optional class coordination (9↔42 is always forced)")

    # Search Mode settings
    basic_search_exact_only: bool = Field(default=True)
    advanced_search_fuzzy_matching: bool = Field(default=True)
    advanced_search_partial_matching: bool = Field(default=True)
    
    # Logging settings
    log_level: str = Field(default="INFO")
    db_log_level: str = Field(default="INFO")
    log_directory: str = Field(default="./logs")
    log_file_max_size: str = Field(default="10MB")
    log_file_backup_count: int = Field(default=5)
    enable_audit_logging: bool = Field(default=True)
    audit_log_retention_days: int = Field(default=90)
    log_slow_queries: bool = Field(default=True)
    slow_query_threshold_ms: int = Field(default=1000)
    
    # File paths
    data_directory: str = Field(default="./data")
    temp_directory: str = Field(default="./temp")
    
    # Performance settings
    max_request_size: int = Field(default=10 * 1024 * 1024)
    request_timeout: int = Field(default=60)
    
    # Feature flags
    enable_detailed_logging: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)
    enable_health_checks: bool = Field(default=True)
    enable_download_exports: bool = Field(default=True)
    
    def __init__(self, **kwargs):
        # Auto-generate JWT secret if not provided or too short
        if not kwargs.get('jwt_secret_key') or len(kwargs.get('jwt_secret_key', '')) < 32:
            kwargs['jwt_secret_key'] = secrets.token_urlsafe(32)
        
        super().__init__(**kwargs)
    
    def setup_directories(self):
        directories = [self.data_directory, self.temp_directory, self.log_directory, "./auth", "./cache"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def is_production(self) -> bool:
        return self.environment == 'production'
    
    def is_development(self) -> bool:
        return self.environment == 'development'
    
    def get_cors_origins(self) -> List[str]:
        origins = self.allowed_origins.split(',') if isinstance(self.allowed_origins, str) else self.allowed_origins
        if self.is_development():
            origins.extend(["http://127.0.0.1:3000", "http://127.0.0.1:8080"])
        return origins
    
    # Properties for backward compatibility
    @property
    def database(self):
        return SimpleNamespace(
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            username=self.db_username,
            password=self.db_password,
            auth_database=self.auth_db_name,
            create_auth_db=self.create_auth_db,
            read_only_mode=self.db_read_only_mode,
            allow_schema_creation=self.db_allow_schema_creation,
            min_connections=self.db_min_connections,
            max_connections=self.db_max_connections,
            connection_timeout=self.db_connection_timeout,
            query_timeout=self.db_query_timeout,
            max_retries=self.db_max_retries,
            connection_string=f"postgresql://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}",
            async_connection_string=f"postgresql+asyncpg://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}",
            auth_connection_string=f"postgresql://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.auth_db_name}"
        )
    
    @property
    def security(self):
        return SimpleNamespace(
            jwt_secret_key=self.jwt_secret_key,
            jwt_algorithm=self.jwt_algorithm,
            jwt_expiration_hours=self.jwt_expiration_hours,
            session_timeout_minutes=self.session_timeout_minutes,
            max_concurrent_sessions=self.max_concurrent_sessions,
            rate_limit_per_minute=self.rate_limit_per_minute,
            rate_limit_burst=self.rate_limit_burst,
            failed_login_threshold=self.failed_login_threshold,
            lockout_duration_minutes=self.lockout_duration_minutes,
            allowed_origins=self.allowed_origins.split(',') if isinstance(self.allowed_origins, str) else self.allowed_origins
        )
    
    @property
    def search(self):
        return SimpleNamespace(
            default_phonetic_threshold=self.default_phonetic_threshold,
            default_visual_threshold=self.default_visual_threshold,
            default_conceptual_threshold=self.default_conceptual_threshold,
            max_search_results=self.max_search_results,
            default_batch_size=self.default_batch_size,
            enable_phonetic_expansion=self.enable_phonetic_expansion,
            enable_visual_expansion=self.enable_visual_expansion,
            enable_morphological_expansion=self.enable_morphological_expansion,
            enable_conceptual_expansion=self.enable_conceptual_expansion,
            enable_leet_speak_detection=self.enable_leet_speak_detection,
            enable_ai_analysis=self.enable_ai_analysis,
            ai_model_name=self.ai_model_name,
            ai_api_timeout=self.ai_api_timeout,
            enable_search_cache=self.enable_search_cache,
            cache_ttl_minutes=self.cache_ttl_minutes,
            max_variations_per_type=self.max_variations_per_type,
            force_9_42_coordination=self.force_9_42_coordination,
            enable_class_coordination=self.enable_class_coordination,
            enable_optional_coordination=self.enable_optional_coordination,
            basic_search_exact_only=self.basic_search_exact_only,
            advanced_search_fuzzy_matching=self.advanced_search_fuzzy_matching,
            advanced_search_partial_matching=self.advanced_search_partial_matching
        )
    
    @property
    def logging(self):
        return SimpleNamespace(
            log_level=self.log_level,
            db_log_level=self.db_log_level,
            log_directory=self.log_directory,
            log_file_max_size=self.log_file_max_size,
            log_file_backup_count=self.log_file_backup_count,
            enable_audit_logging=self.enable_audit_logging,
            audit_log_retention_days=self.audit_log_retention_days,
            log_slow_queries=self.log_slow_queries,
            slow_query_threshold_ms=self.slow_query_threshold_ms
        )
    
    # NICE class coordination - enhanced property
    @property
    def nice_class_coordination(self):
        return NICE_CLASS_COORDINATION
    
    # Helper methods for NICE classes
    def get_nice_class_info(self, class_id: str) -> Dict[str, str]:
        """Get comprehensive NICE class information"""
        normalized_id = str(int(class_id)) if class_id.isdigit() else class_id
        
        return {
            "id": normalized_id,
            "title": NICE_CLASS_TITLES.get(normalized_id, f"Class {normalized_id}"),
            "category": "Goods" if int(normalized_id) <= 34 else "Services",
            "coordinated_classes": self.nice_class_coordination.get(normalized_id, [])
        }

class SimpleNamespace:
    """Simple namespace object for backward compatibility"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

@lru_cache()
def get_config() -> AppConfig:
    config = AppConfig()
    config.setup_directories()
    return config

def validate_config() -> bool:
    try:
        config = get_config()
        try:
            import asyncpg
        except ImportError:
            return False
        if len(config.security.jwt_secret_key) < 32:
            return False
        return True
    except Exception:
        return False

# Keep these classes for compatibility with existing code
class DatabaseConfig:
    pass

class SecurityConfig:
    pass

class SearchConfig:
    pass

class LoggingConfig:
    pass