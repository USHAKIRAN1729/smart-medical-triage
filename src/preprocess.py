import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Symptom2Disease.csv")

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)

def preprocess():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"text": "symptom", "label": "disease"})
    df = df.dropna(subset=["symptom", "disease"]).drop_duplicates()
    df["symptom_clean"] = df["symptom"].apply(clean_text)
    df = df[df["symptom_clean"].str.strip() != ""]
    return df
