import json
from PIL import Image
from .search_utils import load_movies
from sentence_transformers import SentenceTransformer, util


class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents

        # Form text string for each movie
        self.texts = [
            f"{doc['title']}: {doc['description']}" for doc in documents
        ]

        # Pre-compute text embeddings
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        embeddings = self.model.encode([image])
        return embeddings[0]

    def search_with_image(self, image_path: str, limit: int = 5) -> list[dict]:
        image_embedding = self.embed_image(image_path)

        # Cosine similarity against all movie text embeddings
        similarities = util.cos_sim(image_embedding, self.text_embeddings)[0]

        results = []
        for doc, score in zip(self.documents, similarities):
            results.append({
                "id": doc.get("id"),
                "title": doc.get("title", "Unknown"),
                "description": doc.get("description", ""),
                "score": float(score),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


def verify_image_embedding(image_path: str):
    search = MultimodalSearch(documents=[])
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str) -> list[dict]:
    movies = load_movies()
    if not movies:
        return []

    search = MultimodalSearch(documents=movies)
    return search.search_with_image(image_path)