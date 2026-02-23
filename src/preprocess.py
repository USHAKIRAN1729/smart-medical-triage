import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download only required resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Get absolute path safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Symptom2Disease.csv")


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    words = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in STOPWORDS and len(w) > 2
    ]

    return " ".join(words)


def preprocess():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    # Rename columns safely (if required)
    df = df.rename(columns={"text": "symptom", "label": "disease"})

    # Drop null values
    df = df.dropna(subset=["symptom", "disease"])

    # Remove duplicates
    df = df.drop_duplicates()

    print("Cleaning text...")
    df["symptom_clean"] = df["symptom"].apply(clean_text)

    # Remove empty cleaned rows
    df = df[df["symptom_clean"].str.strip() != ""]

    # Save cleaned file
    cleaned_path = os.path.join(BASE_DIR, "data", "Symptom2Disease_cleaned.csv")
    df.to_csv(cleaned_path, index=False)

    print("Preprocessing completed successfully!")

    return df
