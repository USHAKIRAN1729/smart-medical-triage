import os
import json
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

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

# ------------------ Load models & data ------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

with open(os.path.join(DATA_DIR, "snapshots.json")) as f:
    centroids = json.load(f)

with open(os.path.join(DATA_DIR, "cluster_mappings.json")) as f:
    cluster_map = json.load(f)

centroid_vectors = np.array(list(centroids.values()))

# ------------------ Validation helpers ------------------

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

# ------------------ Symptom category override ------------------

def symptom_category_override(text: str):
    t = text.lower()

    # Skin symptoms
    if any(w in t for w in ["rash", "itch", "skin", "redness", "patch", "peeling"]):
        return {
            "cluster_id": None,
            "disease_label": "Skin-related condition",
            "recommended_specialist": "Dermatologist",
            "confidence": 1.0
        }

    # Respiratory symptoms
    if any(w in t for w in ["shortness of breath", "breathing", "chest pain", "persistent cough"]):
        return {
            "cluster_id": None,
            "disease_label": "Respiratory condition",
            "recommended_specialist": "Pulmonologist",
            "confidence": 1.0
        }

    # Gastrointestinal symptoms
    if any(w in t for w in ["stomach", "vomiting", "burning", "loose motions", "diarrhea", "nausea"]):
        return {
            "cluster_id": None,
            "disease_label": "Gastrointestinal condition",
            "recommended_specialist": "Gastroenterologist",
            "confidence": 1.0
        }

    # Neurological symptoms
    if any(w in t for w in ["headache", "dizziness", "blurred vision", "sensitivity to light"]):
        return {
            "cluster_id": None,
            "disease_label": "Neurological condition",
            "recommended_specialist": "Neurologist",
            "confidence": 1.0
        }

    return None

# ------------------ Disease → Specialist mapping ------------------

def map_specialist(disease: str) -> str:
    d = disease.lower()
    if "pneumonia" in d:
        return "Pulmonologist"
    if "fungal" in d:
        return "Dermatologist"
    if "malaria" in d or "dengue" in d:
        return "General Physician"
    return "General Physician"

# ------------------ Prediction endpoint ------------------

@app.post("/predict_specialist")
def predict(data: Input):

    text = data.symptom_text.lower()

    # 1️⃣ Medical validation
    if not is_medical(text):
        return {
            "cluster_id": None,
            "disease_label": "Not a medical symptom",
            "recommended_specialist": "N/A",
            "confidence": 0.0
        }

    # 2️⃣ Generic-only symptoms → GP
    if is_generic_only(text):
        return {
            "cluster_id": None,
            "disease_label": "General symptoms",
            "recommended_specialist": "General Physician",
            "confidence": 0.0
        }

    # 3️⃣ Category override
    override = symptom_category_override(text)
    if override:
        return override

    # 4️⃣ Dynamic clustering
    emb = normalize(model.encode([text]))
    dist = cdist(emb, centroid_vectors)
    idx = int(np.argmin(dist))

    disease = cluster_map.get(str(idx), "Unknown")
    specialist = map_specialist(disease)
    confidence = round(1 / (1 + dist[0][idx]), 3)

    return {
        "cluster_id": idx,
        "disease_label": disease,
        "recommended_specialist": specialist,
        "confidence": confidence
    }

@app.get("/health")
def health():
    return {"status": "ok"}
