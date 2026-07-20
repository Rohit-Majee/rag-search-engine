import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

from lib.search_utils import load_movies

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        fetch_limit = limit * 500
        
        # 1. Fetch from both engines
        bm25_results = self._bm25_search(query,fetch_limit)
        semantic_results = self.semantic_search.search_chunks(query,fetch_limit)
        
        # 2. Extract raw scores into lists
        bm25_raw_scores = [scores for (movies,scores)in bm25_results]
        semantic_raw_scores = [res['score'] for res in semantic_results]
        
        # 3. Normalize the score lists
        bm25_norm_scores = normalize_scores(bm25_raw_scores)
        semantic_norm_scores = normalize_scores(semantic_raw_scores)
        
        combined_results = {}
        
        for idx, result in enumerate(bm25_results):
            doc_id = result[0]['id']
            combined_results[doc_id] = {
                "document": result[0],  
                "bm25_score": bm25_norm_scores[idx],
                "semantic_score": 0.0  # Default to 0 in case it wasn't found by semantic engine
            }
            
        
        for idx, result in enumerate(semantic_results):
            doc_id = result['id']
            if doc_id in combined_results:
                combined_results[doc_id]['semantic_score'] = semantic_norm_scores[idx]
            else:
                combined_results[doc_id] = {
                "document": result,  
                "bm25_score": 0.0,
                "semantic_score": semantic_norm_scores[idx] 
            }

        # 5. Calculate Hybrid Scores
        final_list = []
        for doc_id, data in combined_results.items():
            data["hybrid_score"] = hybrid_score(data['bm25_score'],data['semantic_score'],alpha)
            final_list.append(data)
            
        # 6. Sort and slice
        final_result = sorted(final_list,key = lambda x: x['hybrid_score'],reverse=True)
        return final_result[:limit]

    def rrf_search(self, query: str, k: int = 60, limit: int = 10) -> list[dict]:
        fetch_limit = limit * 500
        
        # Fetch from both engines
        bm25_results = self._bm25_search(query,fetch_limit)
        semantic_results = self.semantic_search.search_chunks(query,fetch_limit)
        
        combined_results = {}
        
        #  Process BM25 Ranks
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result[0]['id']
            combined_results[doc_id] = {
                "document": result[0],  
                "bm25_rank": rank,
                "semantic_rank": None, # Use None to indicate it wasn't found yet
                "rrf_score": rrf_score(rank, k)
            }
            
            
        # Process Semantic Ranks
        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result['id']
            if doc_id in combined_results:
                combined_results[doc_id]['semantic_rank'] = rank
                combined_results[doc_id]['rrf_score'] += rrf_score(rank, k)
            else:
                combined_results[doc_id] = {
                "document": result,  
                "bm25_rank": None,
                "semantic_rank": rank,
                "rrf_score": rrf_score(rank, k)
            }
            
        final_list = [data for doc_id, data in sorted(
                        combined_results.items(), 
                        key=lambda item: item[1]["rrf_score"], 
                        reverse=True
                    )]
        return final_list[:limit]
    
def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    range_score = max_score - min_score

    if max_score == min_score : return [1.]*len(scores)
    
    return [(score-min_score)/ range_score for score in scores]

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search_command(query: str, alpha: float, limit: int = 5):
    movies = load_movies()
    hs = HybridSearch(movies)
    result = hs.weighted_search(query,alpha,limit)

    for i, res in enumerate(result,start=1):
        print(f"{i}. {res['document']['title']}")
        print(f"  Hybrid Score: {res['hybrid_score']:.3f}")
        print(f"  BM25: {res['bm25_score']:.3f}, Semantic: {res['semantic_score']:.3f}")
        print(f"  {res['document']['description'][:100]}...\n") 

def rrf_search_command(query: str, k: int = 60, limit: int = 10):
    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.rrf_search(query,k,limit)

    for i, res in enumerate(results, start=1):
        bm_rank = res['bm25_rank'] if res['bm25_rank'] is not None else "N/A"
        sem_rank = res['semantic_rank'] if res['semantic_rank'] is not None else "N/A"

        doc_dict = res['document']
        snippet = doc_dict.get('document', doc_dict.get('description', ''))[:100]

        print(f"\n{i}. {doc_dict['title']}")
        print(f"  RRF Score: {res['rrf_score']:.3f}")
        print(f"  BM25 Rank: {bm_rank}, Semantic Rank: {sem_rank}")
        print(f"  {snippet}...")
