# custom_vertex_ai_job.py
import argparse
from google.cloud import aiplatform

# Prebuilt training container for sklearn
TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--staging_bucket", required=True, help="gs://bucket")
    parser.add_argument("--data_uri", required=True, help="gs://.../tweets.csv")
    parser.add_argument("--bucket", required=True, help="Bucket name (no gs://)")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--machine_type", default="n1-standard-4")
    args = parser.parse_args()

    aiplatform.init(
        project=args.project_id,
        location=args.region,
        staging_bucket=args.staging_bucket
    )

    job = aiplatform.CustomTrainingJob(
        display_name=f"train-{args.model_name}",
        script_path="train_model.py",
        container_uri=TRAIN_IMAGE,
        # These are extra deps installed into the prebuilt at runtime
        requirements=[
            "pandas==2.2.3",
            "scikit-learn==1.6.1",
            "joblib==1.4.2",
            "google-cloud-storage>=2.16.0",
        ],
    )

    job.run(
        args=[
            "--data_uri", args.data_uri,
            "--bucket", args.bucket,
            "--model_name", args.model_name,
        ],
        replica_count=1,
        machine_type=args.machine_type,
    )

    print("[INFO] Training job finished. Check the logs above for ARTIFACT_DIR.")
    print(f"[HINT] Your model artifacts should be at: gs://{args.bucket}/models/{args.model_name}/")

if __name__ == "__main__":
    main()
