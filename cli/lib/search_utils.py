import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
CACHE_PATH = PROJECT_ROOT / "cache"

BM25_K1 = 1.5
BM25_B = 0.75

def load_movies() -> list[dict]:
    """Load movies from movies.json."""
    try:
        with open(DATA_PATH, "r") as f:
            data = json.load(f)
            return data['movies']
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {DATA_PATH}.")
        return []