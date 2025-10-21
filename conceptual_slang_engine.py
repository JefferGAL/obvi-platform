#!/usr/bin/env python3
"""
Conceptual Slang Engine - Refactored as a Reusable Module
This module provides a SlangEngine class to orchestrate an LLM for generating
slang words for a given term.
"""
import asyncio
import os
import sys
import logging
import json
import ollama
import inflect
import requests
import re
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup
import aiohttp

# =============================================================================
# Streamlined LLM Integration for Slang Generation
# =============================================================================
class OllamaClient:
    def __init__(self, api_url="http://localhost:11434"):
        self.api_url = api_url
        self.model = self._select_preferred_model()
        
    def _select_preferred_model(self):
        """Tries to find a model that is likely to be available."""
        try:
            models_url = f"{self.api_url}/api/tags"
            response = requests.get(models_url, timeout=5)
            response.raise_for_status()
            models = response.json().get('models', [])
            
            for model_info in models:
                name = model_info.get('name', '').split(':')[0]
                if name in ['llama3', 'phi3', 'mistral']:
                    return name
            
            if models:
                return models[0]['name'].split(':')[0]
        
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Warning: Could not connect to Ollama server at {self.api_url}. "
                  "Please ensure Ollama is running and a model is downloaded.")
            print(f"Error details: {e}")
            return None
        
        return None

    async def get_slang_terms_from_llm(self, term: str, num_words: int = 10) -> List[str]:
        """
        Uses the LLM to generate a list of slang words related to a given term.
        """
        if not self.model:
            raise ConnectionError("LLM model not available. Please check your Ollama installation.")

        prompt = f"""
        You are a cultural expert and a master of slang. The term is '{term}'. 
        Generate a list of {num_words} modern English slang words or phrases
        that are conceptually related to this term. 
        For example, if the term is 'money', you might include 'cash' or 'dough'. 
        If the term is 'dope', you might include 'cool' or 'awesome'.
        Provide only the words, separated by commas, with no additional text or explanations.
        """
        try:
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            raw_output = response['message']['content']
            slang_words = [word.strip() for word in raw_output.split(',') if word.strip()]
            return slang_words
        except Exception as e:
            print(f"Failed to get slang terms from LLM: {e}")
            return []

# =============================================================================
# Main Slang Engine Class
# =============================================================================
class SlangEngine:
    """
    A reusable engine to generate slang variations for a given term.
    """
    def __init__(self):
        self.ollama_client = OllamaClient()

    async def generate_slang_variations(self, term: str, limit: int = 25) -> List[str]:
        """
        The main method to get slang variations for a term, up to a specified limit.

        Args:
            term (str): The term to generate slang for.
            limit (int): The maximum number of slang terms to return.

        Returns:
            List[str]: A list of slang terms.
        """
        if not term:
            return []
        
        try:
            # We ask the LLM for a slightly larger number to have options
            llm_request_count = min(limit + 5, 40) 
            slang_words_from_llm = await self.ollama_client.get_slang_terms_from_llm(term, num_words=llm_request_count)
            
            if not slang_words_from_llm:
                return []
            
            # Ensure the original term and unique variations are included and the limit is respected
            final_variations = {term.lower()}
            for word in slang_words_from_llm:
                final_variations.add(word.lower())

            return list(final_variations)[:limit]

        except ConnectionError as e:
            print(f"Error connecting to LLM for slang generation: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred in SlangEngine: {e}")
            return []

# Example usage for testing purposes
async def _test_engine():
    print("Testing SlangEngine...")
    engine = SlangEngine()
    test_term = "friend"
    variations = await engine.generate_slang_variations(test_term, limit=10)
    print(f"Slang for '{test_term}': {variations}")

if __name__ == '__main__':
    asyncio.run(_test_engine())