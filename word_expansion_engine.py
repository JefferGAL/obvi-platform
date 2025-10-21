#!/usr/bin/env python3
"""
Word Expansion Engine - Enhanced Version with Full Backward Compatibility
Replaces the original word_expansion_engine.py with enhanced features
Maintains exact API compatibility for existing code
"""

import re
import logging
import asyncio
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from functools import lru_cache
import unicodedata

from config_app_config import LEET_SPEAK_MAPPING, VISUAL_SIMILARITY_MAPPING, generate_leet_variations

logger = logging.getLogger(__name__)

# Enhanced NLP and phonetic libraries
try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available - some features will be limited")

try:
    import jellyfish
    JELLYFISH_AVAILABLE = True
except ImportError:
    JELLYFISH_AVAILABLE = False
    logger.warning("Jellyfish not available - phonetic analysis will be limited")

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class VariationConfig:
    """Configuration for variation generation - EXACT backward compatibility"""
    enable_phonetic: bool = True
    enable_visual: bool = True
    enable_morphological: bool = True
    enable_conceptual: bool = True
    max_variations_per_type: int = 50
    min_word_length: int = 2
    confidence_threshold: float = 0.5

@dataclass
class EnhancedVariationConfig:
    """Enhanced configuration for advanced features"""
    enable_phonetic: bool = True
    enable_visual: bool = True
    enable_morphological: bool = True
    enable_conceptual: bool = True
    enable_leet_speak: bool = True
    enable_case_variations: bool = True
    enable_unicode_confusables: bool = True
    max_variations_per_type: int = 50
    min_word_length: int = 2
    confidence_threshold: float = 0.5
    target_total_variations: int = 500

# =============================================================================
# PHONETIC ENGINE
# =============================================================================

class PhoneticEngine:
    """Generate phonetic variations - Enhanced but API compatible"""
    
    def __init__(self):
        self.soundex_cache = {}
        self.metaphone_cache = {}
        
        # Enhanced phonetic substitution patterns
        self.phonetic_patterns = [
            # Consonant clusters
            ('ck', 'k'), ('qu', 'kw'), ('x', 'ks'), ('ph', 'f'),
            # Vowel sounds
            ('ai', 'ay'), ('ay', 'ai'), ('ei', 'ay'), ('ey', 'ay'),
            ('ou', 'ow'), ('ow', 'ou'), ('oo', 'u'), ('u', 'oo'),
            # Silent letters
            ('kn', 'n'), ('wr', 'r'), ('mb', 'm'), ('bt', 't'),
            ('gh', ''), ('ght', 't'), ('tch', 'ch'), ('dge', 'ge'),
            # Common substitutions
            ('c', 'k'), ('k', 'c'), ('s', 'z'), ('z', 's'),
            ('j', 'g'), ('g', 'j'), ('i', 'y'), ('y', 'i'),
            ('v', 'w'), ('w', 'v'), ('b', 'p'), ('p', 'b'),
            ('d', 't'), ('t', 'd'), ('f', 'v'), ('v', 'f')
        ]
    
    def generate_phonetic_variations(self, word: str) -> Set[str]:
        """Generate phonetic variations - EXACT API compatibility"""
        variations = set()
        word_clean = self._clean_word(word)
        
        if not word_clean or len(word_clean) < 2:
            return variations
        
        # Enhanced jellyfish-based variations
        if JELLYFISH_AVAILABLE:
            variations.update(self._jellyfish_variations(word_clean))
        
        # Pattern-based substitutions
        variations.update(self._pattern_substitutions(word_clean))
        
        # Advanced phonetic rules
        variations.update(self._advanced_phonetic_rules(word_clean))
        
        # Double letter variations
        variations.update(self._double_letter_variations(word_clean))
        
        # Remove original and filter
        variations.discard(word_clean.lower())
        variations = {v for v in variations if len(v) >= 2 and v.isalpha()}
        
        return variations
    
    def _jellyfish_variations(self, word: str) -> Set[str]:
        """Generate variations using Jellyfish algorithms"""
        variations = set()
        
        try:
            # Soundex
            soundex = jellyfish.soundex(word)
            if soundex:
                variations.add(soundex.lower())
            
            # Metaphone
            metaphone = jellyfish.metaphone(word)
            if metaphone:
                variations.add(metaphone.lower())
                # Double metaphone
                dm1, dm2 = jellyfish.dmetaphone(word)
                if dm1: variations.add(dm1.lower())
                if dm2: variations.add(dm2.lower())
            
            # NYSIIS
            nysiis = jellyfish.nysiis(word)
            if nysiis:
                variations.add(nysiis.lower())
            
            # Match Rating Codex
            mrc = jellyfish.match_rating_codex(word)
            if mrc:
                variations.add(mrc.lower())
                
        except Exception as e:
            logger.debug(f"Jellyfish error for {word}: {e}")
        
        return variations
    
    def _pattern_substitutions(self, word: str) -> Set[str]:
        """Apply phonetic pattern substitutions"""
        variations = set()
        word_lower = word.lower()
        
        for original, replacement in self.phonetic_patterns:
            if original in word_lower:
                new_word = word_lower.replace(original, replacement)
                if new_word != word_lower and len(new_word) >= 2:
                    variations.add(new_word)
        
        return variations
    
    def _advanced_phonetic_rules(self, word: str) -> Set[str]:
        """Apply advanced phonetic transformation rules"""
        variations = set()
        word_lower = word.lower()
        
        # Beginning sound variations
        if word_lower.startswith('c'):
            variations.add('k' + word_lower[1:])
            variations.add('s' + word_lower[1:])
        elif word_lower.startswith('k'):
            variations.add('c' + word_lower[1:])
        elif word_lower.startswith('s'):
            variations.add('c' + word_lower[1:])
            variations.add('z' + word_lower[1:])
        
        # Ending sound variations
        if word_lower.endswith('s'):
            variations.add(word_lower[:-1] + 'z')
        elif word_lower.endswith('z'):
            variations.add(word_lower[:-1] + 's')
        
        return variations
    
    def _double_letter_variations(self, word: str) -> Set[str]:
        """Generate variations with doubled/undoubled letters"""
        variations = set()
        word_lower = word.lower()
        
        # Remove double letters
        for i in range(len(word_lower) - 1):
            if word_lower[i] == word_lower[i + 1]:
                undoubled = word_lower[:i] + word_lower[i + 1:]
                if len(undoubled) >= 2:
                    variations.add(undoubled)
        
        # Add double letters
        consonants = 'bcdfghjklmnpqrstvwxyz'
        for i, char in enumerate(word_lower):
            if char in consonants:
                doubled = word_lower[:i] + char + word_lower[i:]
                variations.add(doubled)
        
        return variations
    
    def _clean_word(self, word: str) -> str:
        """Clean word for phonetic analysis"""
        return re.sub(r'[^a-zA-Z]', '', word).strip()

# =============================================================================
# VISUAL ENGINE
# =============================================================================

class VisualEngine:
    """Generate visual/appearance variations - Enhanced but API compatible"""
    
    def __init__(self):
        # Extended visual similarity mappings
        self.visual_mappings = {
            **VISUAL_SIMILARITY_MAPPING,
            'a': ['@', '4', 'A', 'α'], 'e': ['3', 'E', 'ε'], 
            'c': ['(', 'C', 'ç'], 'g': ['9', 'G', 'q'],
            'r': ['R', 'γ'], 't': ['T', '7', '+'], 'f': ['F', 'ƒ'],
            'k': ['K', 'κ'], 'x': ['X', '×'], 'j': ['J', 'ĵ']
        }
        
        # Leet speak mapping
        self.leet_mapping = {
            'a': ['4', '@', 'A'], 'e': ['3', 'E'], 'i': ['1', '!', 'I', 'l'],
            'o': ['0', 'O'], 's': ['5', '$', 'S'], 't': ['7', 'T'],
            'l': ['1', 'I', 'L'], 'g': ['9', 'G'], 'b': ['6', 'B'],
            'z': ['2', 'Z'], 'u': ['U', 'v'], 'v': ['V', 'u']
        }
        
        self.shape_similar = {
            'o': ['0', 'q', 'p', 'd'], '0': ['o', 'O', 'Q'],
            'i': ['l', '1', '!', 'I'], 'l': ['i', 'I', '1'],
            'n': ['m', 'h'], 'm': ['n', 'h'], 'h': ['n', 'm'],
            'q': ['p', 'g', 'o'], 'p': ['q', 'b', 'd'],
            'b': ['d', 'p', '6'], 'd': ['b', 'p', 'q'],
            'u': ['v', 'n'], 'v': ['u', 'y'], 'y': ['v', 'i']
        }
    
    def generate_visual_variations(self, word: str) -> Set[str]:
        """Generate visual variations - EXACT API compatibility"""
        variations = set()
        word_clean = self._normalize_word(word)
        
        if not word_clean or len(word_clean) < 2:
            return variations
        
        # Leet speak variations
        variations.update(generate_leet_variations(word_clean, max_variations=30))
        
        # Shape-similar character substitutions
        variations.update(self._shape_substitutions(word_clean))
        
        # Case variations
        variations.update(self._case_variations(word_clean))
        
        # Unicode confusables
        variations.update(self._unicode_confusables(word_clean))
        
        # Mixed case patterns
        variations.update(self._mixed_case_patterns(word_clean))
        
        # Remove original and filter
        variations.discard(word_clean.lower())
        variations = {v for v in variations if len(v) >= 2}
        
        return variations
    
    def _shape_substitutions(self, word: str) -> Set[str]:
        """Generate shape-based character substitutions"""
        variations = set()
        word_lower = word.lower()
        
        for char, alternatives in self.visual_mappings.items():
            if char in word_lower:
                for alt in alternatives[:3]:  # Limit alternatives
                    new_word = word_lower.replace(char, alt)
                    if new_word != word_lower:
                        variations.add(new_word)
        
        return variations
    
    def _case_variations(self, word: str) -> Set[str]:
        """Generate case variations"""
        variations = set()
        
        # Basic case variations
        variations.add(word.upper())
        variations.add(word.lower())
        variations.add(word.title())
        variations.add(word.capitalize())
        
        # Advanced case patterns
        if len(word) <= 12:  # Avoid explosion for long words
            # Alternating case
            alt1 = ''.join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(word))
            alt2 = ''.join(c.lower() if i % 2 == 0 else c.upper() for i, c in enumerate(word))
            variations.update([alt1, alt2])
        
        return variations
    
    def _unicode_confusables(self, word: str) -> Set[str]:
        """Generate Unicode confusable variations"""
        variations = set()
        
        confusables = {
            'a': ['à', 'á', 'â', 'ã', 'ä', 'å'],
            'e': ['è', 'é', 'ê', 'ë', 'ē', 'ĕ'],
            'i': ['ì', 'í', 'î', 'ï', 'ĩ', 'ī'],
            'o': ['ò', 'ó', 'ô', 'õ', 'ö', 'ø'],
            'u': ['ù', 'ú', 'û', 'ü', 'ũ', 'ū'],
            'c': ['ç', 'ć', 'ĉ', 'ċ', 'č'],
            'n': ['ñ', 'ń', 'ņ', 'ň'],
            's': ['š', 'ś', 'ŝ', 'ş'],
            'z': ['ž', 'ź', 'ż']
        }
        
        word_lower = word.lower()
        for char, alternatives in confusables.items():
            if char in word_lower:
                for alt in alternatives[:2]:  # Limit to prevent explosion
                    variations.add(word_lower.replace(char, alt))
        
        return variations
    
    def _mixed_case_patterns(self, word: str) -> Set[str]:
        """Generate mixed case patterns"""
        variations = set()
        
        if len(word) <= 10:  # Reasonable limit
            # camelCase patterns
            for i in range(1, len(word)):
                camel = word[:i].lower() + word[i:].capitalize()
                variations.add(camel)
        
        return variations
    
    def _normalize_word(self, word: str) -> str:
        """Normalize word for visual analysis"""
        return re.sub(r'\s+', '', word.strip())

# =============================================================================
# MORPHOLOGICAL ENGINE
# =============================================================================

class MorphologicalEngine:
    """Generate morphological variations - Enhanced but API compatible"""
    
    def __init__(self):
        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.stemmer = PorterStemmer()
                nltk.download('wordnet', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
            except Exception as e:
                logger.warning(f"NLTK initialization error: {str(e)}")
                self.lemmatizer = None
                self.stemmer = None
        else:
            self.lemmatizer = None
            self.stemmer = None
        
        # Enhanced affix collections
        self.common_prefixes = [
            'un', 're', 'pre', 'dis', 'mis', 'over', 'under', 'out',
            'up', 'sub', 'inter', 'super', 'auto', 'co', 'de', 'non',
            'anti', 'pro', 'semi', 'multi', 'mega', 'micro', 'ultra'
        ]
        
        self.common_suffixes = [
            'er', 'ed', 'ing', 'ly', 'ion', 'tion', 'sion', 'ness',
            'ment', 'ful', 'less', 'able', 'ible', 'al', 'ial', 'ous',
            'ious', 'ive', 'ative', 'itive', 'y', 'ty', 'ity',
            'tech', 'soft', 'ware', 'sys'
        ]
    
    def generate_morphological_variations(self, word: str) -> Set[str]:
        """Generate morphological variations - EXACT API compatibility"""
        variations = set()
        word_clean = self._clean_word_for_morphology(word)
        
        if not word_clean or len(word_clean) < 3:
            return variations
        
        # NLTK-based variations
        if self.stemmer and self.lemmatizer:
            variations.update(self._nltk_variations(word_clean))
        
        # Affix manipulations
        variations.update(self._affix_variations(word_clean))
        
        # Number variations (singular/plural)
        variations.update(self._number_variations(word_clean))
        
        # Verb form variations
        variations.update(self._verb_variations(word_clean))
        
        # Compound word variations
        variations.update(self._compound_variations(word_clean))
        
        # Remove original and filter
        variations.discard(word_clean.lower())
        variations = {v for v in variations if len(v) >= 3 and v.isalpha()}
        
        return variations
    
    def _nltk_variations(self, word: str) -> Set[str]:
        """Generate NLTK-based variations"""
        variations = set()
        
        try:
            # Stemming
            stem = self.stemmer.stem(word)
            if stem != word and len(stem) >= 3:
                variations.add(stem)
            
            # Lemmatization with different POS tags
            pos_tags = ['n', 'v', 'a', 'r']  # noun, verb, adjective, adverb
            for pos in pos_tags:
                lemma = self.lemmatizer.lemmatize(word, pos=pos)
                if lemma != word and len(lemma) >= 3:
                    variations.add(lemma)
        
        except Exception as e:
            logger.debug(f"NLTK morphological error for {word}: {e}")
        
        return variations
    
    def _affix_variations(self, word: str) -> Set[str]:
        """Generate prefix and suffix variations"""
        variations = set()
        word_lower = word.lower()
        
        # Remove prefixes
        for prefix in self.common_prefixes:
            if word_lower.startswith(prefix) and len(word_lower) > len(prefix) + 2:
                root = word_lower[len(prefix):]
                variations.add(root)
        
        # Remove suffixes
        for suffix in self.common_suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                root = word_lower[:-len(suffix)]
                variations.add(root)
        
        # Add prefixes (only for reasonable length words)
        if 3 <= len(word_lower) <= 8:
            for prefix in ['un', 're', 'pre', 'cyber', 'ultra']:
                variations.add(prefix + word_lower)
        
        # Add suffixes
        if 3 <= len(word_lower) <= 8:
            for suffix in ['er', 'ly', 'tech', 'soft', 'sys']:
                variations.add(word_lower + suffix)
        
        return variations
    
    def _number_variations(self, word: str) -> Set[str]:
        """Generate singular/plural variations"""
        variations = set()
        word_lower = word.lower()
        
        # Plural to singular
        if word_lower.endswith('s') and len(word_lower) > 3:
            variations.add(word_lower[:-1])
        
        if word_lower.endswith('es') and len(word_lower) > 4:
            variations.add(word_lower[:-2])
        
        if word_lower.endswith('ies') and len(word_lower) > 5:
            variations.add(word_lower[:-3] + 'y')
        
        # Singular to plural
        if not word_lower.endswith('s') and len(word_lower) >= 3:
            variations.add(word_lower + 's')
            
            if word_lower.endswith('y'):
                variations.add(word_lower[:-1] + 'ies')
            
            if word_lower.endswith(('ch', 'sh', 'x', 'z', 'o')):
                variations.add(word_lower + 'es')
        
        return variations
    
    def _verb_variations(self, word: str) -> Set[str]:
        """Generate verb form variations"""
        variations = set()
        word_lower = word.lower()
        
        if len(word_lower) < 4:
            return variations
        
        # Add verb endings
        if not word_lower.endswith(('ing', 'ed')):
            # Present participle
            if word_lower.endswith('e'):
                variations.add(word_lower[:-1] + 'ing')
            else:
                variations.add(word_lower + 'ing')
            
            # Past tense
            if word_lower.endswith('e'):
                variations.add(word_lower + 'd')
            elif word_lower.endswith('y'):
                variations.add(word_lower[:-1] + 'ied')
            else:
                variations.add(word_lower + 'ed')
        
        # Remove verb endings
        if word_lower.endswith('ing') and len(word_lower) > 6:
            base = word_lower[:-3]
            variations.add(base)
            variations.add(base + 'e')
        
        if word_lower.endswith('ed') and len(word_lower) > 5:
            base = word_lower[:-2]
            variations.add(base)
            variations.add(base + 'e')
        
        return variations
    
    def _compound_variations(self, word: str) -> Set[str]:
        """Generate compound word variations"""
        variations = set()
        word_lower = word.lower()
        
        # Common tech/business compound patterns
        tech_parts = ['tech', 'soft', 'ware', 'sys', 'net', 'web', 'cyber', 'digital']
        business_parts = ['pro', 'max', 'ultra', 'mega', 'super', 'smart', 'fast']
        
        # If word contains compound parts, separate them
        for part in tech_parts + business_parts:
            if part in word_lower and part != word_lower:
                if word_lower.startswith(part):
                    remainder = word_lower[len(part):]
                    if len(remainder) >= 3:
                        variations.add(remainder)
                elif word_lower.endswith(part):
                    remainder = word_lower[:-len(part)]
                    if len(remainder) >= 3:
                        variations.add(remainder)
        
        return variations
    
    def _clean_word_for_morphology(self, word: str) -> str:
        """Clean word for morphological analysis"""
        return re.sub(r'[^a-zA-Z\-]', '', word).strip().lower()

# =============================================================================
# CONCEPTUAL ENGINE
# =============================================================================

class ConceptualEngine:
    """Generate conceptual variations using synonyms - Enhanced but API compatible"""
    
    def __init__(self):
        self.synonym_cache = {}
        self.wordnet_available = NLTK_AVAILABLE
        
        if self.wordnet_available:
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
            except Exception as e:
                logger.warning(f"WordNet download error: {str(e)}")
                self.wordnet_available = False
        
        # Enhanced manual synonym mapping for trademark-relevant terms
        self.manual_synonyms = {
            # Technology terms
            'tech': ['technology', 'digital', 'cyber', 'smart', 'intelligent', 'system'],
            'digital': ['tech', 'cyber', 'electronic', 'virtual', 'online', 'web'],
            'smart': ['intelligent', 'clever', 'wise', 'bright', 'sharp', 'auto'],
            'cyber': ['digital', 'tech', 'virtual', 'electronic', 'net', 'web'],
            'system': ['sys', 'platform', 'solution', 'framework', 'tech'],
            'software': ['soft', 'app', 'program', 'application', 'solution'],
            'network': ['net', 'web', 'system', 'connection', 'link'],
            'cloud': ['server', 'virtual', 'online', 'remote', 'network', 'computing'],
            
            # Speed and performance
            'fast': ['quick', 'rapid', 'swift', 'speedy', 'express', 'turbo'],
            'quick': ['fast', 'rapid', 'swift', 'instant', 'immediate', 'speed'],
            'rapid': ['fast', 'quick', 'swift', 'speedy', 'express'],
            'instant': ['immediate', 'quick', 'rapid', 'fast', 'now'],
            'turbo': ['fast', 'quick', 'rapid', 'boost', 'power', 'speed'],
            
            # Scale and scope
            'global': ['worldwide', 'international', 'universal', 'planetary', 'world'],
            'mega': ['huge', 'massive', 'giant', 'enormous', 'ultra', 'super'],
            'ultra': ['extreme', 'maximum', 'supreme', 'mega', 'super', 'max'],
            'micro': ['tiny', 'small', 'mini', 'compact', 'little'],
            'mini': ['small', 'tiny', 'compact', 'micro', 'little'],
            'max': ['maximum', 'ultimate', 'supreme', 'peak', 'top', 'ultra'],
            
            # Quality and premium
            'premium': ['luxury', 'elite', 'exclusive', 'superior', 'high-end', 'pro'],
            'pro': ['professional', 'expert', 'advanced', 'master', 'elite', 'premium'],
            'elite': ['premium', 'exclusive', 'superior', 'pro', 'advanced'],
            'luxury': ['premium', 'high-end', 'exclusive', 'elite', 'superior'],
            
            # Security and protection
            'secure': ['safe', 'protected', 'guarded', 'defended', 'shielded'],
            'safe': ['secure', 'protected', 'guarded', 'reliable', 'trusted'],
            'protect': ['secure', 'guard', 'defend', 'shield', 'safe'],
            
            # Environmental and sustainability
            'eco': ['green', 'environmental', 'sustainable', 'natural', 'organic'],
            'green': ['eco', 'environmental', 'sustainable', 'natural', 'clean'],
            'bio': ['organic', 'natural', 'life', 'living', 'biological', 'eco'],
            'organic': ['natural', 'bio', 'green', 'eco', 'pure']
        }
    
    def generate_conceptual_variations(self, word: str, max_synonyms: int = 5) -> Set[str]:
        """Generate conceptual variations - EXACT API compatibility"""
        variations = set()
        word_clean = self._clean_word_for_concepts(word)
        
        if not word_clean or len(word_clean) < 3:
            return variations
        
        # Use manual synonyms first
        variations.update(self._get_manual_synonyms(word_clean, max_synonyms))
        
        # Use WordNet if available
        if self.wordnet_available:
            variations.update(self._get_wordnet_synonyms(word_clean, max_synonyms))
        
        # Generate semantic variations
        variations.update(self._generate_semantic_variations(word_clean))
        
        # Remove original word
        variations.discard(word_clean.lower())
        
        return variations
    
    def _get_manual_synonyms(self, word: str, max_count: int) -> Set[str]:
        """Get synonyms from manual mapping"""
        word_lower = word.lower()
        
        if word_lower in self.manual_synonyms:
            return set(self.manual_synonyms[word_lower][:max_count])
        
        # Check if word is a synonym of any key
        for key, synonyms in self.manual_synonyms.items():
            if word_lower in synonyms:
                result = {key}
                result.update(synonyms[:max_count-1])
                result.discard(word_lower)
                return result
        
        return set()
    
    def _get_wordnet_synonyms(self, word: str, max_count: int) -> Set[str]:
        """Get synonyms from WordNet"""
        if word in self.synonym_cache:
            return self.synonym_cache[word]
        
        synonyms = set()
        
        try:
            # Get all synsets for the word
            synsets = wordnet.synsets(word)
            
            for synset in synsets[:3]:  # Limit to first 3 synsets for performance
                # Get lemma names (synonyms) from the synset
                for lemma in synset.lemmas()[:max_count]:
                    synonym = lemma.name().lower().replace('_', ' ')
                    if synonym != word and len(synonym) >= 3:
                        synonyms.add(synonym)
                
                # Get related synsets (hypernyms, hyponyms)
                for hypernym in synset.hypernyms()[:2]:
                    for lemma in hypernym.lemmas()[:2]:
                        synonym = lemma.name().lower().replace('_', ' ')
                        if synonym != word and len(synonym) >= 3:
                            synonyms.add(synonym)
                
                if len(synonyms) >= max_count:
                    break
            
            # Cache the result
            self.synonym_cache[word] = synonyms
            
        except Exception as e:
            logger.warning(f"WordNet synonym lookup error for '{word}': {str(e)}")
        
        return synonyms
    
    def _generate_semantic_variations(self, word: str) -> Set[str]:
        """Generate semantic variations based on word patterns"""
        variations = set()
        
        # Common business/brand term variations
        business_mappings = {
            'corp': ['corporation', 'company', 'inc', 'ltd'],
            'inc': ['incorporated', 'company', 'corp', 'co'],
            'ltd': ['limited', 'company', 'corp', 'co'],
            'co': ['company', 'corporation', 'inc', 'ltd'],
            'tech': ['technology', 'technologies', 'systems'],
            'sys': ['systems', 'solutions', 'services'],
            'solutions': ['systems', 'services', 'technologies'],
            'services': ['solutions', 'systems', 'support'],
            'group': ['companies', 'corporation', 'holdings'],
            'labs': ['laboratory', 'research', 'development'],
            'studio': ['studios', 'creative', 'design'],
            'media': ['communications', 'broadcasting', 'digital'],
            'net': ['network', 'networks', 'internet', 'web'],
            'web': ['internet', 'online', 'digital', 'net'],
            'app': ['application', 'software', 'program'],
            'soft': ['software', 'applications', 'programs']
        }
        
        word_lower = word.lower()
        if word_lower in business_mappings:
            variations.update(business_mappings[word_lower])
        
        return variations
    
    def _clean_word_for_concepts(self, word: str) -> str:
        """Clean word for conceptual analysis"""
        # Remove non-alphabetic characters
        cleaned = re.sub(r'[^a-zA-Z]', '', word)
        return cleaned.strip().lower()

# =============================================================================
# MAIN WORD EXPANSION ENGINE - WITH BACKWARD COMPATIBILITY
# =============================================================================

class WordExpansionEngine:
    """Main engine coordinating all variation generation - Enhanced with full backward compatibility"""
    
    def __init__(self, config: Optional[VariationConfig] = None):
        self.config = config or VariationConfig()
        
        # Initialize sub-engines with EXACT same names as original
        self.phonetic_engine = PhoneticEngine()
        self.visual_engine = VisualEngine()
        self.morphological_engine = MorphologicalEngine()
        self.conceptual_engine = ConceptualEngine()
        
        # Variation cache
        self.variation_cache = {}
        
        logger.info("WordExpansionEngine initialized with enhanced features and backward compatibility")
    
    async def generate_variations(
        self,
        trademark: str,
        variation_types: List[str] = None
    ) -> Dict[str, List[str]]:  # EXACT return type match for backward compatibility
        """
        Generate comprehensive variations for a trademark
        BACKWARD COMPATIBLE: Returns Dict[str, List[str]] as expected by existing code
        """
        
        if variation_types is None:
            variation_types = ['phonetic', 'visual', 'morphological', 'conceptual']
        
        # Check cache first
        cache_key = f"{trademark}:{':'.join(sorted(variation_types))}"
        if cache_key in self.variation_cache:
            return self.variation_cache[cache_key]
        
        # Split trademark into words
        words = self._tokenize_trademark(trademark)
        
        # Generate variations for each word using EXACT same method names
        all_variations = {vtype: set() for vtype in variation_types}
        
        for word in words:
            if len(word) < self.config.min_word_length:
                continue
            
            # Call the sub-engines using EXACT same method names as original
            if 'phonetic' in variation_types and self.config.enable_phonetic:
                phonetic_vars = self.phonetic_engine.generate_phonetic_variations(word)
                all_variations['phonetic'].update(phonetic_vars)
            
            if 'visual' in variation_types and self.config.enable_visual:
                visual_vars = self.visual_engine.generate_visual_variations(word)
                all_variations['visual'].update(visual_vars)
            
            if 'morphological' in variation_types and self.config.enable_morphological:
                morph_vars = self.morphological_engine.generate_morphological_variations(word)
                all_variations['morphological'].update(morph_vars)
            
            if 'conceptual' in variation_types and self.config.enable_conceptual:
                concept_vars = self.conceptual_engine.generate_conceptual_variations(word)
                all_variations['conceptual'].update(concept_vars)
        
        # Generate combined variations for multi-word trademarks
        if len(words) > 1:
            combined_variations = await self._generate_combined_variations(
                trademark, words, all_variations
            )
            for vtype, variations in combined_variations.items():
                all_variations[vtype].update(variations)
        
        # Convert sets to lists and limit - EXACT format match
        result = {}
        for vtype, variations_set in all_variations.items():
            limited_variations = list(variations_set)[:self.config.max_variations_per_type]
            result[vtype] = limited_variations  # Returns List[str] as expected
        
        # Cache the result
        self.variation_cache[cache_key] = result
        
        logger.info(f"Generated {sum(len(v) for v in result.values())} variations for '{trademark}' across {len(result)} types")
        return result  # Returns Dict[str, List[str]] - EXACT backward compatibility
    
    def _tokenize_trademark(self, trademark: str) -> List[str]:
        """Tokenize trademark into words - Enhanced but compatible"""
        # Handle various separators and camelCase
        trademark = re.sub(r'([a-z])([A-Z])', r'\1 \2', trademark)  # Split camelCase
        words = re.findall(r'\b\w+\b', trademark.lower())
        
        # Filter out very short words and common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in words if len(word) >= self.config.min_word_length and word not in stop_words]
        
        return words
    
    async def _generate_combined_variations(
        self,
        original_trademark: str,
        words: List[str],
        word_variations: Dict[str, Set[str]]
    ) -> Dict[str, Set[str]]:
        """Generate variations by combining word variations - Enhanced"""
        
        combined = {vtype: set() for vtype in word_variations.keys()}
        
        # For each variation type, try combining variations of different words
        for vtype, all_word_vars in word_variations.items():
            if not all_word_vars:
                continue
            
            # Create combinations by replacing one word at a time
            for i, original_word in enumerate(words):
                # Get variations for this specific word position
                word_vars = {var for var in all_word_vars if self._is_variation_of_word(var, original_word)}
                
                for variation in list(word_vars)[:10]:  # Limit combinations to prevent explosion
                    new_words = words.copy()
                    new_words[i] = variation
                    combined_trademark = ' '.join(new_words)
                    combined[vtype].add(combined_trademark)
                    
                    # Also add compound version (no spaces)
                    combined[vtype].add(''.join(new_words))
            
            # Add individual word variations as single terms
            combined[vtype].update(all_word_vars)
        
        return combined
    
    def _is_variation_of_word(self, variation: str, original_word: str) -> bool:
        """Check if a variation likely belongs to a specific word - Enhanced heuristic"""
        variation_lower = variation.lower()
        original_lower = original_word.lower()
        
        # If lengths are very different, probably not related
        if abs(len(variation_lower) - len(original_lower)) > 3:
            return False
        
        # Check character overlap
        common_chars = set(variation_lower) & set(original_lower)
        min_length = min(len(variation_lower), len(original_lower))
        
        return len(common_chars) >= min_length * 0.6
    
    def get_variation_confidence(self, original: str, variation: str, variation_type: str) -> float:
        """Calculate confidence score for a variation - EXACT backward compatibility"""
        
        # Base confidence scores by type
        base_confidence = {
            'phonetic': 0.8,
            'visual': 0.7,
            'morphological': 0.6,
            'conceptual': 0.5
        }
        
        confidence = base_confidence.get(variation_type, 0.5)
        
        # Adjust based on similarity metrics
        original_lower = original.lower()
        variation_lower = variation.lower()
        
        # Length similarity
        length_ratio = min(len(original_lower), len(variation_lower)) / max(len(original_lower), len(variation_lower))
        confidence *= (0.7 + 0.3 * length_ratio)
        
        # Character overlap
        common_chars = set(original_lower) & set(variation_lower)
        char_ratio = len(common_chars) / max(len(set(original_lower)), len(set(variation_lower)))
        confidence *= (0.5 + 0.5 * char_ratio)
        
        return min(confidence, 1.0)

# =============================================================================
# ENHANCED WORD EXPANSION ENGINE ALIAS
# =============================================================================

class EnhancedWordExpansionEngine(WordExpansionEngine):
    """
    Enhanced Word Expansion Engine with additional features
    Extends the base WordExpansionEngine with enhanced capabilities
    """
    
    def __init__(self, config: Optional[EnhancedVariationConfig] = None):
        # Convert enhanced config to base config for compatibility
        if isinstance(config, EnhancedVariationConfig):
            base_config = VariationConfig(
                enable_phonetic=config.enable_phonetic,
                enable_visual=config.enable_visual,
                enable_morphological=config.enable_morphological,
                enable_conceptual=config.enable_conceptual,
                max_variations_per_type=config.max_variations_per_type,
                min_word_length=config.min_word_length,
                confidence_threshold=config.confidence_threshold
            )
        else:
            base_config = config or VariationConfig()
        
        super().__init__(base_config)
        
        # Store enhanced config for additional features
        self.enhanced_config = config or EnhancedVariationConfig()
        
        # Progress tracking
        self.progress_callbacks = []
        
        logger.info("EnhancedWordExpansionEngine initialized with advanced features")
    
    def add_progress_callback(self, callback):
        """Add progress callback for real-time tracking"""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, stage: str, count: int, details: str = ""):
        """Notify all progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(stage, count, details)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")
    
    async def generate_comprehensive_variations(
        self,
        trademark: str,
        variation_types: List[str] = None,
        target_count: int = 200
    ) -> Dict[str, List[str]]:
        """
        Enhanced variation generation with higher target count and progress tracking
        """
        if variation_types is None:
            variation_types = ['phonetic', 'visual', 'morphological', 'conceptual']
        
        self._notify_progress("initialization", 0, f"Starting comprehensive variation generation for '{trademark}'")
        
        # Temporarily increase max variations per type for enhanced mode
        original_max = self.config.max_variations_per_type
        enhanced_max = min(target_count // len(variation_types), 100)  # Cap at 100 per type
        self.config.max_variations_per_type = enhanced_max
        
        try:
            # Use the base method but with enhanced limits
            result = await self.generate_variations(trademark, variation_types)
            
            total_variations = sum(len(v) for v in result.values())
            self._notify_progress("completion", total_variations, f"Generated {total_variations} comprehensive variations")
            
            return result
        finally:
            # Restore original setting
            self.config.max_variations_per_type = original_max
    
    def format_variations_for_display(self, variations: Dict[str, List[str]]) -> Dict[str, Any]:
        """Format variations for user-friendly display"""
        
        display_format = {
            'summary': {
                'total_variations': sum(len(vars_list) for vars_list in variations.values()),
                'categories': len(variations),
                'original_trademark': getattr(self, '_last_trademark', 'Unknown')
            },
            'categories': {},
            'all_variations_list': []
        }
        
        # Category descriptions
        category_descriptions = {
            'phonetic': 'Sound-alike variations (how the trademark sounds when spoken)',
            'visual': 'Visual variations (how the trademark looks, including character substitutions)',
            'morphological': 'Word form variations (prefixes, suffixes, verb forms, plurals)',
            'conceptual': 'Meaning-based variations (synonyms and related concepts)'
        }
        
        for category, vars_list in variations.items():
            display_format['categories'][category] = {
                'description': category_descriptions.get(category, f'{category.title()} variations'),
                'count': len(vars_list),
                'variations': vars_list[:50],  # Limit display
                'sample': vars_list[:5] if vars_list else []
            }
            
            display_format['all_variations_list'].extend(vars_list)
        
        # Remove duplicates from all variations list
        display_format['all_variations_list'] = list(set(display_format['all_variations_list']))
        
        return display_format

# =============================================================================
# BACKWARD COMPATIBILITY EXPORTS
# =============================================================================

# Export all classes for different use cases
__all__ = [
    'WordExpansionEngine',           # Main backward-compatible engine
    'EnhancedWordExpansionEngine',   # Enhanced version with additional features
    'VariationConfig',               # Basic configuration
    'EnhancedVariationConfig',       # Enhanced configuration
    'PhoneticEngine',                # Individual engines
    'VisualEngine',
    'MorphologicalEngine', 
    'ConceptualEngine'
]

# For absolute backward compatibility, create the exact class aliases
# that might be expected by legacy code
def create_legacy_aliases():
    """Create legacy aliases for maximum backward compatibility"""
    # These ensure that any old import patterns continue to work
    globals()['PhoneticEngine'] = PhoneticEngine
    globals()['VisualEngine'] = VisualEngine  
    globals()['MorphologicalEngine'] = MorphologicalEngine
    globals()['ConceptualEngine'] = ConceptualEngine

# Initialize legacy aliases
create_legacy_aliases()

logger.info("Enhanced Word Expansion Engine loaded with full backward compatibility")