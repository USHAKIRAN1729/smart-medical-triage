import os
import json
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dynmeans import DynMeans

app = FastAPI(title="Smart Dynamic Triage API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Input(BaseModel):
    symptom_text: str

model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load Initial Dynamic State
initial_centers = np.load(os.path.join(DATA_DIR, "cluster_centers.npy"))
dm = DynMeans(lambda_dist=0.8) # Threshold set to 0.8
dm.centers = list(initial_centers)
dm.counts = [1] * len(initial_centers)
dm.inactive = [0] * len(initial_centers)

with open(os.path.join(DATA_DIR, "cluster_mapping.json"), "r") as f:
    cluster_mapping = json.load(f)

def map_specialist(disease: str) -> str:
    d = disease.lower()
    mapping = {
        "Pulmonologist": ["pneumonia", "asthma", "bronchitis", "cough"],
        "Dermatologist": ["fungal", "eczema", "psoriasis", "rash", "acne"],
        "Cardiologist": ["hypertension", "heart", "chest pain"],
        "Gastroenterologist": ["gerd", "peptic ulcer", "jaundice", "typhoid"]
    }
    for spec, keywords in mapping.items():
        if any(k in d for k in keywords): return spec
    return "General Physician"

@app.post("/predict_specialist")
async def predict(data: Input):
    text = data.symptom_text.strip()
    if not text: return {"disease_label": "No input", "confidence": 0.0}

    user_emb = model.encode([text])
    
    # Run through DynMeans for real-time adaptation
    labels = dm.fit_batch(user_emb)
    cluster_idx = labels[0]

    all_centers = dm.get_centers()
    similarity = float(cosine_similarity(user_emb, [all_centers[cluster_idx]])[0][0])
    
    disease = cluster_mapping.get(str(cluster_idx), "Unidentified Symptom Pattern")
    specialist = map_specialist(disease) if disease != "Unidentified Symptom Pattern" else "General Physician"

    return {
        "disease_label": disease.title(),
        "recommended_specialist": specialist,
        "confidence": round(similarity, 3),
        "is_dynamic_discovery": str(cluster_idx) not in cluster_mapping
    }

@app.get("/health")
def health(): return {"status": "ok"}
