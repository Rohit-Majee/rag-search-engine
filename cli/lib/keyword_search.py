from lib.search_utils import load_movies
import string
import nltk
# nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize(text: str) -> list[str]:
    return [tok for tok in text.split() if tok]

def remove_stop_words(tokens: list[str]) -> list[str]:
    return [word for word in tokens if word not in stop_words]

def stem(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(tok) for tok in tokens]

def process(raw_text: str) -> list[str]:
    """The Master Pipeline"""
    normalized_text = normalize(raw_text)
    tokens = tokenize(normalized_text)
    meaningful_tokens = remove_stop_words(tokens)
    final_tokens = stem(meaningful_tokens)
    return final_tokens

def has_match(query_tokens,movie_tokens):
    for querry_tok in query_tokens:
        for movie_tok in movie_tokens:
            if querry_tok in movie_tok:
                return True
            
    return False

def search_command(query, n_result=5):
    movies = load_movies()
    result = []
    query_tokens = process(query)
    for movie in movies:
        if has_match(query_tokens, process(movie["title"])):
            result.append(movie)
        if len(result) == n_result:
            break
        
    return result