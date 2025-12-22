import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z ]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)

def preprocess():
    df = pd.read_csv("../data/Symptom2Disease.csv")
    df = df.rename(columns={"text": "symptom", "label": "disease"})
    df["symptom_clean"] = df["symptom"].apply(clean_text)
    df.to_csv("../data/Symptom2Disease_cleaned.csv", index=False)
    return df