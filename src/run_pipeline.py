import os
import numpy as np
from preprocess import preprocess
from embeddings import generate_embeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def run_pipeline():
    print("Starting pipeline...")

    df = preprocess()

    X = generate_embeddings(df["symptom_clean"].tolist())

    np.save(os.path.join(DATA_DIR, "embeddings.npy"), X)

    print("Embeddings saved successfully!")

if __name__ == "__main__":
    run_pipeline()
