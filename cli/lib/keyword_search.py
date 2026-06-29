import os
import pickle
import math
import string
from collections import defaultdict, Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from lib.search_utils import load_movies, CACHE_PATH, BM25_K1, BM25_B

# --- GLOBALS & INITIALIZATION ---
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- TEXT PROCESSING PIPELINE ---
def normalize(text: str) -> str:
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))

def tokenize(text: str) -> list[str]:
    return [tok for tok in text.split() if tok]

def remove_stop_words(tokens: list[str]) -> list[str]:
    return [word for word in tokens if word not in stop_words]

def stem(tokens: list[str]) -> list[str]:
    return [stemmer.stem(tok) for tok in tokens]

def process(raw_text: str) -> list[str]:
    """The Master Pipeline"""
    normalized_text = normalize(raw_text)
    tokens = tokenize(normalized_text)
    meaningful_tokens = remove_stop_words(tokens)
    return stem(meaningful_tokens)

# --- CORE SEARCH ENGINE ---
class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict] = {} # maps document ID to document
        self.term_frequencies = defaultdict(Counter) 
        self.doc_lengths = {}

        self.index_path = CACHE_PATH / "index.pkl"
        self.docmap_path = CACHE_PATH / "docmap.pkl"
        self.term_frequencies_path = CACHE_PATH / "term_frequencies.pkl"
        self.doc_lengths_path = CACHE_PATH / "doc_lengths.pkl"

    def __add_document(self, doc_id: int, text: str):
        """Adds a single document's text to the index."""
        tokens = process(text)
        for token in set(tokens):
            self.index[token].add(doc_id) 
        
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        """Gets sorted document IDs for a single preprocessed token."""
        return sorted(list(self.index[term]))
    
    def get_tf(self, doc_id: int, term: str):
        """Gets the term frequency"""
        tokens = process(term)
        if len(tokens) == 0:
            return 0
        if len(tokens) > 1:
            raise ValueError("get_tf only accepts a single term.")
        
        return self.term_frequencies[doc_id][tokens[0]]
    
    def get_idf(self, term:str) -> float:
        """Calculates the Inverse Document Frequency of a term."""
        tokens = process(term)
        if len(tokens) == 0:
            return 0.0
        if len(tokens) > 1:
            raise ValueError("get_idf only accepts a single term.")
            
        token = tokens[0]

        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])
        idf_score = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf_score
    
    def get_tf_idf(self, doc_id: int, term: str) -> float:
        """Calculates the TF-IDF score for a term in a document."""
        return self.get_tf(doc_id,term) * self.get_idf(term)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id,term)
        if tf == 0:
            return 0.0

        doc_length = self.doc_lengths.get(doc_id,0)
        avg_doc_length =self.__get_avg_doc_length()

        if avg_doc_length == 0.0 :
            length_norm = 1.0 
        else :
            length_norm = 1.0 - b + b * (doc_length / avg_doc_length)

        bm25_tf =  (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf
    
    def get_bm25_idf(self, term: str) -> float:
        """Calculates the BM25 IDF score for a term."""
        tokens = process(term)
        if len(tokens) == 0:
            return 0.0
        if len(tokens) > 1:
            raise ValueError("get_bm25_idf only accepts a single term.")
            
        token = tokens[0]

        N = len(self.docmap) # Total documents
        df = len(self.index[token]) # Document frequency

        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def bm25(self, doc_id: int, term: str) -> float:
        """Calculates the full BM25 score for a term in a document."""
        bm25_tf = self.get_bm25_tf(doc_id,term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query: str, limit: int = 5) -> list[tuple[dict, float]]:
        """Searches documents using full BM25 scoring."""
        query_tokens = process(query)
        scores: dict[int, float] = defaultdict(float)

        for qt in query_tokens:
            matching_doc_ids = self.get_documents(qt)
            for doc_id in matching_doc_ids:
                scores[doc_id] += self.bm25(doc_id,qt) 
                

        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        result = []
        for doc_id, score in sorted_docs[:limit]:
            movie_object = self.docmap[doc_id]
            result.append((movie_object, score))
            
        return result
    
    
    def build(self):
        """Loads movies, populates the index and docmap, and saves to disk."""
        print("Loading movies and building index...")
        movies = load_movies()
        
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        """Saves the index and docmap to disk using pickle."""
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        """Loads the index and docmap from disk."""
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

# --- CLI COMMANDS ---

def bm25_search_command(query: str, limit: int = 5) -> list[tuple[dict, float]]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query,limit)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def tf_command(doc_id:int, term:str):
    idx = InvertedIndex()
    idx.load()
    print(idx.get_tf(doc_id,term))

def idf_command(term:str):
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)

def tf_idf_command(doc_id:int, term:str):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id,term)

def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    print("Index successfully built and cached!")


def search_command(query: str, n_result: int = 5) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    
    query_tokens = process(query)
    scores: dict[int, float] = defaultdict(float)

    for qt in query_tokens:
        matching_doc_ids = idx.get_documents(qt)
        for doc_id in matching_doc_ids:
            scores[doc_id] += idx.get_tf_idf(doc_id, qt)

    sorted_doc_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
    
    result = []
    for doc_id in sorted_doc_ids[:n_result]:
        result.append(idx.docmap[doc_id])
        
    return result