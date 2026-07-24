from .search_utils import load_movies, load_test_cases
from .hybrid_search import HybridSearch

def evaluate(limit=5):
    movies = load_movies()
    test_cases = load_test_cases()

    hs = HybridSearch(movies)
    print(f"k={limit}\n")

    for test_case in test_cases:
        query = test_case["query"]
        relevant_titles = test_case['relevant_docs']
        relevant_set = set(relevant_titles)
        
        rrf_results = hs.rrf_search(query=query, k=60, limit=limit)
        
        relevant_retrieved = sum(1 for res in rrf_results if res['document']['title'] in relevant_set)
        # precision = relevant_retrieved / total_retrieved
        precision = relevant_retrieved / limit
        
        total_relevant = len(relevant_titles)
        # recall = relevant_retrieved / total_relevant
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0

        # f1 = 2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        retrieved_str = ", ".join([res["document"]["title"] for res in rrf_results])
        relevant_str = ", ".join(relevant_titles)

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score@{limit}: {f1:.4f}")
        print(f"  - Retrieved: {retrieved_str}")
        print(f"  - Relevant: {relevant_str}\n")

