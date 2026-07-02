from lib.search_utils import CACHE_PATH,load_movies

from sentence_transformers import SentenceTransformer
import numpy as np
import os

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

        self.embeddings_path = CACHE_PATH/'movie_embeddings.npy'

    def build_embeddings(self, documents):
        """Generates embeddings from scratch and saves them to disk."""
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings,show_progress_bar= True)

        np.save(self.embeddings_path , self.embeddings)
        return self.embeddings        

    def load_or_create_embeddings(self, documents):
        """Smart loader that checks the cache first."""
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)
    
    def generate_embedding(self, text):
        if not text or text.isspace():
            raise ValueError("Input text cannot be empty or just whitespace.")
        embedding = self.model.encode([text])
        return embedding[0]
    
    def search(self,  query:str, limit: int = 5):
        """Searches the document embeddings for the closest semantic matches to the query."""

        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)
        
        similarities = []
        
        for doc_emb,doc in zip(self.embeddings,self.documents):
            similarity = cosine_similarity(query_embedding, doc_emb)
            similarities.append((similarity,doc))
            
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        final_results = []
        for sim , doc in similarities[:limit]:
            final_results.append({
                "similarity": sim,
                "title": doc['title'],
                "description": doc['description']
            })
        
        return final_results
    
def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text: str):
    """Top-level function to test embedding generation."""
    ss = SemanticSearch()
    
    embedding = ss.generate_embedding(text)
    
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {ss.embeddings.shape[0]} vectors in {ss.embeddings.shape[1]} dimensions")

def embed_query_text(query: str) -> None:
    """Converts a user's natural language search query into a dense vector coordinate."""
    ss = SemanticSearch()
    
    embedding = ss.generate_embedding(query)
    
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_command(query:str, limit: int = 5):
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embeddings(documents)
    search_result = ss.search(query,limit)

    for i,doc in enumerate(search_result,start=1):
        print(f"{i}. {doc['title']} (score: {doc['similarity']:.4f}) ")
        print(f"{doc['description'][:100]}...")