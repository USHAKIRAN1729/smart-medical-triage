import os
import json
import numpy as np
import pandas as pd
from preprocess import preprocess
from embeddings import generate_embeddings
from dynmeans import DynMeans

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def run_pipeline():
    df = preprocess()
    X = generate_embeddings(df["symptom_clean"].tolist())

    dm = DynMeans(lambda_dist=0.75)
    labels = dm.fit_batch(X)
    centers = dm.get_centers()

    cluster_to_disease = {}
    for i in range(len(centers)):
        indices = [j for j, label in enumerate(labels) if label == i]
        if indices:
            diseases_in_cluster = df.iloc[indices]["disease"].tolist()
            cluster_to_disease[i] = pd.Series(diseases_in_cluster).mode()[0]

    np.save(os.path.join(DATA_DIR, "cluster_centers.npy"), centers)
    with open(os.path.join(DATA_DIR, "cluster_mapping.json"), "w") as f:
        json.dump(cluster_to_disease, f)
    print("Initial training pipeline complete.")

if __name__ == "__main__":
    run_pipeline()
