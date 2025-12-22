🏥 Smart Medical Triage System
Using Dynamic Symptom Clustering (DynMeans)
📌 Project Overview

The Smart Medical Triage System is an AI-based healthcare support application that helps users identify the appropriate medical specialist based on their symptoms.

The system uses dynamic clustering (DynMeans algorithm) to group similar symptom patterns without fixed retraining, combined with medical triage rules to ensure safe and realistic recommendations.

This project aims to reduce patient confusion, improve initial diagnosis routing, and assist in faster access to appropriate healthcare services.

🎯 Key Objectives

Accept patient symptoms as text input

Dynamically cluster symptom embeddings using DynMeans

Map symptom clusters to likely disease categories

Recommend the correct medical specialist

Ensure medical safety using symptom category overrides

Provide a simple and interactive frontend interface

🧠 Technologies Used
Backend

Python 3.10

FastAPI

Sentence Transformers (all-MiniLM-L6-v2)

Scikit-learn

NumPy

SciPy

Machine Learning

Dynamic Clustering using DynMeans

Sentence Embeddings for symptom representation

Frontend

HTML

CSS

JavaScript (Fetch API)

🧪 Dataset

Symptom2Disease Dataset

Each record contains:

Symptom description (text)

Corresponding disease label

Dataset is preprocessed and cleaned before clustering

⚙️ Project Architecture
User Input (Symptoms)
↓
Medical Input Validation
↓
Generic Symptom Check
↓
Symptom Category Override (Safety Layer)
↓
Dynamic Clustering (DynMeans)
↓
Disease Prediction
↓
Specialist Recommendation

📁 Project Structure
Smart_Medical_Triage/
│
├── data/
│ ├── Symptom2Disease.csv
│ ├── Symptom2Disease_cleaned.csv
│ ├── embeddings.npy
│ ├── cluster_mappings.json
│ └── snapshots.json
│
├── src/
│ ├── preprocess.py
│ ├── embeddings.py
│ ├── dynmeans.py
│ ├── run_pipeline.py
│ └── api.py
│
├── frontend/
│ └── index.html
│
└── README.md

▶️ How to Run the Project

Basic Requirements before running the project

fastapi
uvicorn
sentence-transformers
scikit-learn
numpy
scipy
pandas
nltk

we can install these using the command
pip install -r requirements.txt

1️⃣ Activate Virtual Environment
triage\Scripts\activate

2️⃣ Run the ML Pipeline
cd src
python run_pipeline.py

This will:

Preprocess the dataset

Generate embeddings

Perform dynamic clustering

Save cluster mappings and centroids

3️⃣ Start Backend API
uvicorn api:app --reload

API Documentation:

http://127.0.0.1:8000/docs

4️⃣ Start Frontend
cd ..
python -m http.server 5500

Open in browser:

http://127.0.0.1:5500/frontend/index.html

🧪 Sample Inputs
fever

itchy red rash on skin

stomach pain and vomiting

cough and shortness of breath

✅ Sample Output
Disease: Skin-related condition
Recommended Specialist: Dermatologist

🛡️ Medical Safety Features

Non-medical inputs are rejected

Generic symptoms are routed to General Physician

High-risk symptom categories override clustering

Dynamic clustering is applied only when appropriate

📌 Key Highlights

Uses Dynamic Clustering (no fixed number of clusters)

Combines AI with rule-based medical safety

Real-time symptom analysis

Clinically explainable outputs

User-friendly interface

🎓 Academic Relevance

Based on DynMeans clustering algorithm

Demonstrates real-world application of AI in healthcare

Suitable for final-year engineering project submission

Aligns with AI/ML and Healthcare Informatics domains

📜 Disclaimer

This system is intended for educational purposes only.
It does not replace professional medical consultation.
