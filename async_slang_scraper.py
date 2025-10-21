import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict

async def search_slang_async(session: aiohttp.ClientSession, term: str, num_results: int = 5) -> Dict[str, List[str]]:
    """
    Asynchronously searches for a term on slang.net and returns its top N definitions.

    Args:
        session: The aiohttp client session
        term (str): The word or phrase to search for.
        num_results (int): The maximum number of definitions to return.

    Returns:
        Dict[str, List[str]]: Dictionary with term as key and list of definitions as value.
    """
    search_term = term.replace(" ", "-").strip()
    url = f"https://www.slang.net/meaning/{search_term}"
    
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 404:
                return {term: [f"Term '{term}' not found in slang.net dictionary"]}
            
            response.raise_for_status()
            text = await response.text()
            
            soup = BeautifulSoup(text, 'html.parser')
            
            # Find all divs with the class 'meaning'
            definition_divs = soup.find_all('div', class_='meaning')
            
            definitions = [div.get_text(strip=True) for div in definition_divs]
            
            if definitions:
                # Return only the top N results
                top_definitions = definitions[:num_results]
                return {term: [f"Definition {i+1}: {d}" for i, d in enumerate(top_definitions)]}
            else:
                return {term: [f"No definitions available for '{term}'"]}
                
    except asyncio.TimeoutError:
        return {term: [f"Search timed out for '{term}'"]}
    except aiohttp.ClientError as e:
        return {term: [f"Connection error for '{term}': {str(e)}"]}
    except Exception as e:
        return {term: [f"Unexpected error for '{term}': {str(e)}"]}

async def search_multiple_terms(terms: List[str], num_results: int = 5) -> Dict[str, List[str]]:
    """
    Asynchronously searches multiple terms and returns all results.
    
    Args:
        terms: List of terms to search
        num_results: Maximum number of definitions per term
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping terms to their definitions
    """
    async with aiohttp.ClientSession() as session:
        tasks = [search_slang_async(session, term.strip(), num_results) for term in terms]
        results = await asyncio.gather(*tasks)
        
        # Combine all results into a single dictionary
        combined_results = {}
        for result in results:
            combined_results.update(result)
        
        return combined_results

def parse_terms(user_input: str) -> List[str]:
    """
    Parse user input to extract terms separated by |
    
    Args:
        user_input: Raw input string with terms separated by |
        
    Returns:
        List[str]: List of cleaned terms
    """
    terms = [term.strip() for term in user_input.split("|") if term.strip()]
    return terms

def display_results(results: Dict[str, List[str]]):
    """
    Display search results in a clean format
    
    Args:
        results: Dictionary mapping terms to their definitions
    """
    for term, definitions in results.items():
        print(f"\n=== Results for '{term}' ===")
        for definition in definitions:
            print(f"  {definition}")
        print("-" * 40)

async def main():
    """
    Main function to handle user input and coordinate the search
    """
    print("Slang Dictionary Search Tool")
    print("Enter slang terms to search (separate multiple terms with |)")
    print("Example: tea | bae | flex | bussin")
    print()
    
    while True:
        user_input = input("Enter terms to search (or 'quit' to exit): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not user_input:
            print("Please enter at least one term to search.")
            continue
        
        terms = parse_terms(user_input)
        
        if not terms:
            print("No valid terms found. Please try again.")
            continue
            
        print(f"\nSearching for {len(terms)} term(s)...")
        
        try:
            results = await search_multiple_terms(terms)
            display_results(results)
            
        except KeyboardInterrupt:
            print("\nSearch interrupted by user.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            continue
        
        print()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")