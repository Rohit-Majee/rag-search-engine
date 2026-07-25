from .hybrid_search import HybridSearch
from .search_utils import load_movies
from .llm import llm_answer, llm_summarizer, llm_citations,llm_question

def _fetch_and_format_docs(query: str, limit: int, include_citations: bool = False) -> str:
    """Helper function to run the search and format the context string."""
    movies = load_movies()
    hs = HybridSearch(movies)

    results = hs.rrf_search(query=query, limit=limit)
    print("Search Results:")
    
    docs_formatted = ""
    for i, res in enumerate(results, start=1):
        doc_dict = res['document']
        title = doc_dict.get('title', 'Unknown')
        # Truncate to 500 chars to save VRAM
        doc_text = doc_dict.get('document', doc_dict.get('description', ''))[:500] 
        
        if include_citations:
            print(f"  - {title}")
            docs_formatted += f"\n[{i}] Title: {title}\nSynopsis: {doc_text}\n"
        else:
            print(f"- {title}")
            docs_formatted += f"\nTitle: {title}\nSynopsis: {doc_text}\n"
            
    return docs_formatted


def rag(query: str, limit: int = 5):
    docs_formatted = _fetch_and_format_docs(query, limit)
    print("\nRAG Response:")
    print(llm_answer(query, docs_formatted))


def summarize(query: str, limit: int = 5):
    docs_formatted = _fetch_and_format_docs(query, limit)
    print("\nLLM Summary:")
    print(llm_summarizer(query, docs_formatted))


def citations(query: str, limit: int = 5):
    docs_formatted = _fetch_and_format_docs(query, limit, include_citations=True)
    print("\nLLM Answer:")
    print(llm_citations(query, docs_formatted))

def ask_question(query: str, limit: int = 5):
    docs_formatted = _fetch_and_format_docs(query, limit)
    print("\nAnswer:")
    print(llm_question(query, docs_formatted))