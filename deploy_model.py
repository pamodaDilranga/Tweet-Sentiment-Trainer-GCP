# deploy_model.py
import argparse
from google.cloud import aiplatform

# Prebuilt prediction container for sklearn
PRED_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--artifact_dir", required=True, help="gs://bucket/models/model_name/")
    parser.add_argument("--display_name", default="tweet-sentiment-model")
    parser.add_argument("--machine_type", default="n1-standard-2")
    parser.add_argument("--endpoint_name", default="tweet-sentiment-endpoint")
    args = parser.parse_args()

    aiplatform.init(project=args.project_id, location=args.region)

    print(f"[INFO] Uploading model from: {args.artifact_dir}")
    model = aiplatform.Model.upload(
        display_name=args.display_name,
        artifact_uri=args.artifact_dir,  # folder that contains model.joblib
        serving_container_image_uri=PRED_IMAGE,
    )
    print(f"[INFO] Model uploaded: {model.resource_name}")

    print(f"[INFO] Creating/Deploying endpoint: {args.endpoint_name}")
    endpoint = aiplatform.Endpoint.create(display_name=args.endpoint_name)
    endpoint = model.deploy(
        endpoint=endpoint,
        machine_type=args.machine_type,
        traffic_split={"0": 100},
    )
    print(f"[RESULT] ENDPOINT_ID={endpoint.name.split('/')[-1]}")
    print(f"[RESULT] ENDPOINT_RESOURCE={endpoint.resource_name}")

if __name__ == "__main__":
    main()
