import os
import pickle
import math
import string
from collections import defaultdict, Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from lib.search_utils import load_movies, CACHE_PATH

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

        self.index_path = CACHE_PATH / "index.pkl"
        self.docmap_path = CACHE_PATH / "docmap.pkl"
        self.term_frequencies_path = CACHE_PATH / "term_frequencies.pkl"

    def __add_document(self, doc_id: int, text: str):
        """Adds a single document's text to the index."""
        tokens = process(text)
        for token in set(tokens):
            self.index[token].add(doc_id) 
        
        self.term_frequencies[doc_id].update(tokens)

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

    def load(self):
        """Loads the index and docmap from disk."""
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

# --- CLI COMMANDS ---

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
    
    seen = set()
    result = []
    query_tokens = process(query)

    for qt in query_tokens:
        matching_doc_ids = idx.get_documents(qt)
        for matching_doc_id in matching_doc_ids:
            if matching_doc_id in seen:
                continue

            seen.add(matching_doc_id)
            matching_doc = idx.docmap[matching_doc_id]
            result.append(matching_doc)

            if len(result) >= n_result:
                return result
            
    return result