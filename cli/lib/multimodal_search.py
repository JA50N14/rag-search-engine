import os
import numpy as np

from .search_utils import load_movies
from PIL import Image
from sentence_transformers import SentenceTransformer

model_name = "clip-ViT-B-32"

class MultimodalSearch:
    def __init__(self, documents, model_name=model_name):
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.model = SentenceTransformer(model_name)
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        img = Image.open(image_path)
        image_embedding = self.model.encode([img])
        return image_embedding[0]

    def search_with_image(self, image_path: str):
        image_embedding = self.embed_image(image_path)

        similarities = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(text_embedding, image_embedding)
            similarities.append(
                {
                    "doc_id": self.documents[i]["id"],
                    "title": self.documents[i]["title"],
                    "description": self.documents[i]["description"],
                    "similarity_score": similarity,
                }
            )
        similarities.sort(key=lambda item:item["similarity_score"], reverse=True)
        return similarities[:5]

       
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def verify_image_embedding(image_path):
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path:str):
    movies = load_movies()
    searcher = MultimodalSearch(movies)
    return searcher.search_with_image(image_path)

