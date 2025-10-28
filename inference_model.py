# inference_model.py
import argparse
from google.cloud import aiplatform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--endpoint_id", required=True)
    parser.add_argument("--texts", nargs="+", required=True, help='Texts to classify, e.g. --texts "I love school" "Homework is awful"')
    args = parser.parse_args()

    aiplatform.init(project=args.project_id, location=args.region)

    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{args.project_id}/locations/{args.region}/endpoints/{args.endpoint_id}"
    )

    print(f"[INFO] Predicting on {len(args.texts)} texts...")
    pred = endpoint.predict(instances=args.texts)
    print("[RESULT] Raw predictions:", pred.predictions)

if __name__ == "__main__":
    main()
