from lib.search_utils import CACHE_PATH,load_movies

import numpy as np
import os
import re
import json
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

from sentence_transformers import SentenceTransformer

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
    

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

        self.chunk_emb_path = CACHE_PATH / 'chunk_embeddings.npy'
        self.chunk_meta_path = CACHE_PATH / 'chunk_metadata.json'

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}

        for doc in self.documents:
            self.document_map[doc['id']] = doc
        
        all_chunks = []
        chunk_metadata = []

        for movie_idx, doc in enumerate(self.documents):
            if not doc['description'].strip():
                continue

            chunks = semantic_chunking(doc['description'], max_chunk_size=4, overlap=1)
            all_chunks.extend(chunks)
            chunks_len = len(chunks)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_metadata.append({
                    'movie_idx' : movie_idx,
                    'chunk_idx' : chunk_idx,
                    'total_chunks' : chunks_len
                })
                
        self.chunk_embeddings = self.model.encode(all_chunks,show_progress_bar= True)
        self.chunk_metadata = chunk_metadata
        # --- Caching ---
        np.save(self.chunk_emb_path,self.chunk_embeddings)
        with open(self.chunk_meta_path, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings
            

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc['id']:doc for doc in documents}
        if os.path.exists(self.chunk_emb_path) and os.path.exists(self.chunk_meta_path):
            self.chunk_embeddings = np.load(self.chunk_emb_path)
            with open(self.chunk_meta_path, "r") as f:
                loaded_data = json.load(f)
                self.chunk_metadata = loaded_data["chunks"]
            return self.chunk_embeddings
            
        else:
            return self.build_chunk_embeddings(documents)
        

    def search_chunks(self, query: str, limit: int = 5):
        query_emb = self.generate_embedding(query)
        chunk_scores = []
        movie_scores = defaultdict(lambda:0)
        for i in range(len(self.chunk_embeddings)):
            chunk_emb = self.chunk_embeddings[i]
            score = cosine_similarity(query_emb,chunk_emb)
            meta = self.chunk_metadata[i]
            midx= meta["movie_idx"]
            chunk_scores.append({'chunk_idx' : meta['chunk_idx'], 'movie_idx' : midx, 'score' : score})
            movie_scores[midx] = max(movie_scores[midx], score)

        sorted_movies = sorted(movie_scores.items(), key = lambda x : x[1], reverse=True)
        
        final_results = []
        for movie_idx, max_score in sorted_movies[:limit]:
            doc = self.documents[movie_idx]
            
            final_results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100], 
                "score": round(float(max_score), 4),
                "metadata": {
                    "movie_idx": movie_idx,
                    "score": float(max_score)
                }
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

def fixed_sized_chunking(text , chunk_size=200, overlap=0): 
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")
    
    words = text.split()
    chunks = []
    step_size = chunk_size - overlap

    if not words:
        return chunks
    
    for i in range(0,len(words),step_size):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))

    return chunks

def chunk_text(text,chunk_size=200, overlap=0):
    chunks = fixed_sized_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks,start=1):
        print(f"{i}. {chunk}")

def semantic_chunking(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    if overlap >= max_chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")
    
    text = text.strip()

    if not text:
        return []
    
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 1 and not sentences[0].endswith(('.', '!', '?')):
        sentences = [text]
    
    chunks = []
    step_size = max_chunk_size - overlap
    
    if not sentences:
        return chunks
    
    for i in range(0,len(sentences),step_size):
        chunk_sentences = sentences[i : i + max_chunk_size]
        if i > 0 and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))

    return chunks

def semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap: int = 0):
    chunks = semantic_chunking(text, max_chunk_size, overlap)
    
    if not chunks:
        return
        
    print(f"Semantically chunking {len(text.strip())} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")
    
def search_command(query:str, limit: int = 5):
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embeddings(documents)
    search_result = ss.search(query,limit)

    for i,doc in enumerate(search_result,start=1):
        print(f"{i}. {doc['title']} (score: {doc['similarity']:.4f}) ")
        print(f"{doc['description'][:100]}...")


def embed_chunks():
    css = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = css.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked(query , limit=5):
    css = ChunkedSemanticSearch()
    movies = load_movies()
    css.load_or_create_chunk_embeddings(movies)

    results = css.search_chunks(query,limit)
            
    for i, result in enumerate(results, start=1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")