import os
import json
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ App setup ------------------
app = FastAPI(title="Smart Medical Triage API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input(BaseModel):
    symptom_text: str

# ------------------ Lazy model loading ------------------
model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return model

# ------------------ Load centroids & mappings ------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

with open(os.path.join(DATA_DIR, "snapshots.json")) as f:
    centroids = json.load(f)

with open(os.path.join(DATA_DIR, "cluster_mappings.json")) as f:
    cluster_map = json.load(f)

centroid_vectors = np.array(list(centroids.values()))

# ------------------ Validation ------------------

GENERIC_SYMPTOMS = [
    "fever", "pain", "fatigue", "weakness", "tiredness", "body pain"
]

def is_generic_only(text: str) -> bool:
    words = text.lower().split()
    return len(words) <= 3 and any(w in GENERIC_SYMPTOMS for w in words)

def is_medical(text: str) -> bool:
    medical_words = [
        "fever","pain","cough","rash","itch","vomit","headache",
        "breath","chest","diarrhea","nausea","stomach","burning",
        "vision","dizziness","skin","redness","patch","peeling",
        "loose","weakness","tiredness"
    ]
    return any(w in text.lower() for w in medical_words)

# ------------------ Specialist Mapping ------------------

def map_specialist(disease: str) -> str:
    d = disease.lower()
    if "pneumonia" in d:
        return "Pulmonologist"
    if "fungal" in d:
        return "Dermatologist"
    if "malaria" in d or "dengue" in d:
        return "General Physician"
    return "General Physician"

# ------------------ Prediction Endpoint ------------------

@app.post("/predict_specialist")
async def predict(data: Input):

    text = data.symptom_text.strip().lower()

    if not text:
        return {
            "cluster_id": None,
            "disease_label": "No input provided",
            "recommended_specialist": "N/A",
            "confidence": 0.0
        }

    # Medical validation
    if not is_medical(text):
        return {
            "cluster_id": None,
            "disease_label": "Not a medical symptom",
            "recommended_specialist": "N/A",
            "confidence": 0.0
        }

    # Generic-only check
    if is_generic_only(text):
        return {
            "cluster_id": None,
            "disease_label": "General symptoms",
            "recommended_specialist": "General Physician",
            "confidence": 0.4
        }

    # Embedding
    model = get_model()
    emb = model.encode([text])

    # Cosine similarity (Better than Euclidean)
    sim = cosine_similarity(emb, centroid_vectors)
    idx = int(np.argmax(sim))
    confidence = float(round(sim[0][idx], 3))

    # Low confidence threshold
    if confidence < 0.25:
        return {
            "cluster_id": None,
            "disease_label": "Symptoms unclear. Please provide more details.",
            "recommended_specialist": "General Physician",
            "confidence": confidence
        }

    disease = cluster_map.get(str(idx), "Unknown")
    specialist = map_specialist(disease)

    return {
        "cluster_id": idx,
        "disease_label": disease,
        "recommended_specialist": specialist,
        "confidence": confidence
    }

@app.get("/health")
def health():
    return {"status": "ok"}
