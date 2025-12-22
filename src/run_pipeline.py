import json
import numpy as np
from preprocess import preprocess
from embeddings import generate_embeddings
from dynmeans import DynMeans

def run_pipeline():
    print("Starting pipeline...")

    df = preprocess()
    X = generate_embeddings(df["symptom_clean"].tolist())
    np.save("../data/embeddings.npy", X)

    dyn = DynMeans(lambda_dist=0.8, max_inactive=3)

    labels = []
    batch_size = 100
    for i in range(0, len(X), batch_size):
        labels.extend(dyn.fit_batch(X[i:i+batch_size]))

    df["cluster"] = labels
    centers = dyn.get_centers()

    cluster_map = {}
    for c in sorted(df.cluster.unique()):
        diseases = df[df.cluster == c]["disease"]
        counts = diseases.value_counts()
        confidence = counts.max() / counts.sum()
        cluster_map[str(int(c))] = counts.idxmax() if confidence >= 0.25 else "Unknown"

    with open("../data/cluster_mappings.json", "w") as f:
        json.dump(cluster_map, f, indent=2)

    with open("../data/snapshots.json", "w") as f:
        json.dump({str(i): centers[i].tolist() for i in range(len(centers))}, f)

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
