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

# 1. Initialize Global Dynamic Memory
initial_centers = np.load(os.path.join(DATA_DIR, "cluster_centers.npy"))
dm = DynMeans(lambda_dist=0.8) 
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
        "Gastroenterologist": ["gerd", "peptic ulcer", "jaundice", "typhoid", "cholera"]
    }
    for spec, keywords in mapping.items():
        if any(k in d for k in keywords): return spec
    return "General Physician"

@app.post("/predict_specialist")
async def predict(data: Input):
    text = data.symptom_text.strip()
    if not text: return {"disease_label": "No input", "confidence": 0.0}

    # 2. Generate User Embedding
    user_emb = model.encode([text])
    
    # 3. Dynamic Step: Run through DynMeans
    labels = dm.fit_batch(user_emb)
    cluster_idx = labels[0]

    # 4. Calculate Raw Similarity
all_centers = dm.get_centers()
raw_sim = float(cosine_similarity(user_emb, [all_centers[cluster_idx]])[0][0])

# 5. IMPROVED CONFIDENCE SCALING
# We now assume 0.4 is a "good enough" match and 0.7 is "perfect"
# Logic: (raw_score - floor) / (ceiling - floor)
floor = 0.35
ceiling = 0.75

if raw_sim < floor:
    # If it's below the floor, it's a very weak match (shows as < 10%)
    ui_confidence = (raw_sim / floor) * 0.2 
else:
    # Scale the range [0.35, 0.75] to [20%, 98%]
    ui_confidence = 0.2 + (raw_sim - floor) / (ceiling - floor) * 0.78

# Ensure the percentage stays within logical bounds
ui_confidence = max(0.05, min(0.98, ui_confidence))

    # 6. Identify Disease
    is_new = str(cluster_idx) not in cluster_mapping
    disease = cluster_mapping.get(str(cluster_idx), "Unidentified Symptom Pattern")
    specialist = map_specialist(disease) if not is_new else "General Physician"

    return {
        "disease_label": disease.title(),
        "recommended_specialist": specialist,
        "confidence": round(ui_confidence, 3), 
        "is_dynamic_discovery": is_new,
        "debug_raw_score": round(raw_sim, 3) # Useful for your presentation
    }

@app.get("/health")
def health(): return {"status": "ok"}
