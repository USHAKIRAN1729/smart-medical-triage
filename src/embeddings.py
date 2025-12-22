from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(texts):
    emb = model.encode(texts, show_progress_bar=True)
    return normalize(emb)
