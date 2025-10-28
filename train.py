# train.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

DATA_PATH = "gs://sentiment-demo-bucket/data/tweets.csv"
OUTPUT_DIR = os.environ.get("AIP_MODEL_DIR", "/tmp/model")

df = pd.read_csv(DATA_PATH)
X = df["text"]
y = df["label"]

vec = CountVectorizer()
X_vec = vec.fit_transform(X)
clf = MultinomialNB()
clf.fit(X_vec, y)

os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump((vec, clf), f"{OUTPUT_DIR}/model.joblib")
print(f"Saved model to {OUTPUT_DIR}/model.joblib")
