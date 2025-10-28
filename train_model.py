# train_model.py
import argparse
import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from google.cloud import storage

def upload_to_gcs(local_path: str, gcs_uri: str):
    assert gcs_uri.startswith("gs://"), "gcs_uri must start with gs://"
    _, rest = gcs_uri.split("gs://", 1)
    bucket_name, blob_name = rest.split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_uri", required=True, help="CSV with columns: text,label")
    parser.add_argument("--bucket", required=True, help="GCS bucket name (no gs://)")
    parser.add_argument("--model_name", required=True, help="Folder name under gs://bucket/models/")
    args = parser.parse_args()

    print(f"[INFO] Reading dataset from: {args.data_uri}")
    df = pd.read_csv(args.data_uri)
    assert "text" in df.columns and "label" in df.columns, "CSV must have columns: text,label"

    print("[INFO] Training scikit-learn pipeline (CountVectorizer + MultinomialNB)")
    pipe = Pipeline([
        ("vec", CountVectorizer()),
        ("clf", MultinomialNB()),
    ])
    pipe.fit(df["text"], df["label"])

    # Save locally and then upload to your requested path
    os.makedirs("/tmp/model", exist_ok=True)
    local_path = "/tmp/model/model.joblib"
    joblib.dump(pipe, local_path)
    artifact_uri = f"gs://{args.bucket}/models/{args.model_name}/model.joblib"

    print(f"[INFO] Uploading artifact to: {artifact_uri}")
    upload_to_gcs(local_path, artifact_uri)

    # Also print the parent directory as the artifact directory (Vertex expects a dir)
    artifact_dir = f"gs://{args.bucket}/models/{args.model_name}/"
    print(f"[RESULT] ARTIFACT_DIR={artifact_dir}")

if __name__ == "__main__":
    main()
