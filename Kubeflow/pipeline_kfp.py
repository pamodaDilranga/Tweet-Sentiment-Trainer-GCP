# pipeline_kfp.py
# KFP v2 pipeline that trains (CustomTrainingJob), uploads, deploys, and test-predicts.

from kfp import dsl
from kfp.dsl import component

BASE_IMAGE = "python:3.10"
TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-5:latest"
PRED_IMAGE  = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "google-cloud-aiplatform>=1.63.0",
        "google-cloud-storage>=2.16.0",
        "pandas==2.2.3",
        "scikit-learn==1.6.1",
        "joblib==1.4.2",
    ],
)
def launch_training_job(
    project_id: str,
    region: str,
    staging_bucket: str,            # e.g., gs://sentiment-demo-bucket
    data_uri: str,                  # e.g., gs://sentiment-demo-bucket/data/tweets.csv
    bucket: str,                    # e.g., sentiment-demo-bucket
    model_name: str,                # e.g., sentiment-v1
    machine_type: str = "n1-standard-4",
) -> str:
    """Submits a Vertex AI CustomTrainingJob that trains and uploads to gs://BUCKET/models/MODEL_NAME/.
    Returns the artifact_dir (gs://.../models/MODEL_NAME/)."""

    # Write the training script on the fly so we can autopackage
    TRAIN_SCRIPT = r"""
import argparse, os
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
    bucket.blob(blob_name).upload_from_filename(local_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_uri", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--model_name", required=True)
    args = ap.parse_args()

    print(f"[INFO] Reading dataset from: {args.data_uri}")
    df = pd.read_csv(args.data_uri)
    assert "text" in df.columns and "label" in df.columns, "CSV must have columns: text,label"

    pipe = Pipeline([("vec", CountVectorizer()), ("clf", MultinomialNB())])
    pipe.fit(df["text"], df["label"])

    os.makedirs("/tmp/model", exist_ok=True)
    local_path = "/tmp/model/model.joblib"
    joblib.dump(pipe, local_path)

    artifact_uri = f"gs://{args.bucket}/models/{args.model_name}/model.joblib"
    print(f"[INFO] Uploading model artifact to: {artifact_uri}")
    upload_to_gcs(local_path, artifact_uri)
    print(f"[RESULT] ARTIFACT_DIR=gs://{args.bucket}/models/{args.model_name}/")

if __name__ == "__main__":
    main()
"""
    import textwrap
    with open("train_model_inline.py", "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(TRAIN_SCRIPT))

    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=region, staging_bucket=staging_bucket)

    job = aiplatform.CustomTrainingJob(
        display_name=f"train-{model_name}",
        script_path="train_model_inline.py",
        container_uri=TRAIN_IMAGE,
        requirements=[
            "google-cloud-storage>=2.16.0",
            "pandas==2.2.3",
            "scikit-learn==1.6.1",
            "joblib==1.4.2",
        ],
    )

    job.run(
        args=["--data_uri", data_uri, "--bucket", bucket, "--model_name", model_name],
        replica_count=1,
        machine_type=machine_type,
    )

    # Vertex AI training wrote to gs://{bucket}/models/{model_name}/ via the script.
    artifact_dir = f"gs://{bucket}/models/{model_name}/"
    print("[PIPELINE] artifact_dir:", artifact_dir)
    return artifact_dir


@component(
    base_image=BASE_IMAGE,
    packages_to_install=["google-cloud-aiplatform>=1.63.0"],
)
def upload_and_deploy_model(
    project_id: str,
    region: str,
    artifact_dir: str,                 # output of training step, ends with '/models/<name>/'
    display_name: str = "sentiment-model",
    endpoint_name: str = "sentiment-endpoint",
    machine_type: str = "n1-standard-2",
) -> str:
    """Uploads the sklearn model from artifact_dir and deploys it to an endpoint. Returns endpoint_id."""
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    print(f"[INFO] Uploading model from: {artifact_dir}")
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_dir,
        serving_container_image_uri=PRED_IMAGE,
        serving_container_health_route="/ping",
        serving_container_predict_route="/predict",
    )
    print(f"[INFO] Model uploaded: {model.resource_name}")

    # Create endpoint (new endpoint each run; adjust if you want to reuse)
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    print(f"[INFO] Endpoint created: {endpoint.resource_name}")

    endpoint = model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        traffic_split={"0": 100},
    )
    endpoint_id = endpoint.name.split("/")[-1]
    print(f"[RESULT] ENDPOINT_ID={endpoint_id}")
    return endpoint_id


@component(
    base_image=BASE_IMAGE,
    packages_to_install=["google-cloud-aiplatform>=1.63.0"],
)
def test_predict(
    project_id: str,
    region: str,
    endpoint_id: str,
    text1: str = "I love my school",
    text2: str = "This exam is terrible",
) -> str:
    """Runs a quick prediction to verify the deployment."""
    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=region)

    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{project_id}/locations/{region}/endpoints/{endpoint_id}"
    )
    pred = endpoint.predict(instances=[text1, text2])
    print("[RESULT] predictions:", pred.predictions)
    return str(pred.predictions)


@dsl.pipeline(
    name="sentiment-train-deploy-pipeline",
    description="Train scikit-learn on Vertex AI, upload, deploy, and test-predict."
)
def sentiment_pipeline(
    project_id: str = "sentiment-demo-476518",
    region: str = "us-central1",
    staging_bucket: str = "gs://sentiment-demo-bucket",
    data_uri: str = "gs://sentiment-demo-bucket/data/tweets.csv",
    bucket: str = "sentiment-demo-bucket",
    model_name: str = "sentiment-v1",
    display_name: str = "sentiment-model",
    endpoint_name: str = "sentiment-endpoint",
):
    # Step 1: train
    train_task = launch_training_job(
        project_id=project_id,
        region=region,
        staging_bucket=staging_bucket,
        data_uri=data_uri,
        bucket=bucket,
        model_name=model_name,
    )

    # Step 2: upload & deploy
    deploy_task = upload_and_deploy_model(
        project_id=project_id,
        region=region,
        artifact_dir=train_task.output,
        display_name=display_name,
        endpoint_name=endpoint_name,
    )

    # Step 3: test prediction
    _ = test_predict(
        project_id=project_id,
        region=region,
        endpoint_id=deploy_task.output,
    )
