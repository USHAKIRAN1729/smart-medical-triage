import os
import json
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ App Setup ------------------
app = FastAPI(title="Smart Medical Triage API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input(BaseModel):
    symptom_text: str

# ------------------ Load Model ------------------
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# ------------------ Load Training Data ------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load cleaned dataset
data = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
with open(os.path.join(DATA_DIR, "Symptom2Disease_cleaned.csv"), "r", encoding="utf-8") as f:
    import pandas as pd
    df = pd.read_csv(f)

train_embeddings = data
train_diseases = df["disease"].tolist()

# ------------------ Specialist Mapping ------------------

def map_specialist(disease: str) -> str:
    d = disease.lower()

    if any(x in d for x in ["pneumonia", "asthma", "bronchitis"]):
        return "Pulmonologist"

    if any(x in d for x in ["fungal", "eczema", "psoriasis", "allergy"]):
        return "Dermatologist"

    if any(x in d for x in ["malaria", "dengue", "typhoid", "viral"]):
        return "General Physician"

    return "General Physician"

# ------------------ Validation ------------------

def is_medical(text: str) -> bool:
    medical_words = [
        "fever","pain","cough","cold","rash","itch",
        "vomit","headache","breath","chest","diarrhea",
        "nausea","stomach","burning","vision","dizziness",
        "skin","weakness","tiredness"
    ]
    return any(word in text.lower() for word in medical_words)

# ------------------ Prediction ------------------

@app.post("/predict_specialist")
async def predict(data: Input):

    text = data.symptom_text.strip().lower()

    if not text:
        return {
            "disease_label": "No input provided",
            "recommended_specialist": "N/A",
            "confidence": 0.0
        }

    if not is_medical(text):
        return {
            "disease_label": "Not a medical symptom",
            "recommended_specialist": "N/A",
            "confidence": 0.0
        }

    # Generate embedding
    emb = model.encode([text])

    # Compare with ALL training embeddings
    similarities = cosine_similarity(emb, train_embeddings)[0]

    best_idx = int(np.argmax(similarities))
    confidence = float(round(similarities[best_idx], 3))

    disease = train_diseases[best_idx]
    specialist = map_specialist(disease)

    # If extremely low similarity → fallback
    if confidence < 0.2:
        return {
            "disease_label": "Symptoms unclear. Please provide more details.",
            "recommended_specialist": "General Physician",
            "confidence": confidence
        }

    return {
        "disease_label": disease,
        "recommended_specialist": specialist,
        "confidence": confidence
    }

@app.get("/health")
def health():
    return {"status": "ok"}
