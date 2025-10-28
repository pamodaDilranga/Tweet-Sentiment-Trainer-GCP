# shutdown_mode.py
import argparse
from google.cloud import aiplatform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--endpoint_id", required=True)
    parser.add_argument("--delete_endpoint", action="store_true", help="Also delete the endpoint after undeploy")
    parser.add_argument("--delete_model", action="store_true", help="Delete model(s) that are currently deployed on the endpoint")
    args = parser.parse_args()

    aiplatform.init(project=args.project_id, location=args.region)

    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{args.project_id}/locations/{args.region}/endpoints/{args.endpoint_id}"
    )

    # capture deployed model IDs before undeploy
    deployed_models = [dm.model for dm in endpoint.deployed_models]  # resource names

    print(f"[INFO] Undeploying all models from endpoint {args.endpoint_id} ...")
    for dm in list(endpoint.deployed_models):
        endpoint.undeploy(deployed_model_id=dm.id, traffic_percentage=0)
    print("[INFO] Undeploy complete.")

    if args.delete_endpoint:
        print("[INFO] Deleting endpoint...")
        endpoint.delete()
        print("[INFO] Endpoint deleted.")

    if args.delete_model:
        for model_res in deployed_models:
            try:
                print(f"[INFO] Deleting model: {model_res}")
                aiplatform.Model(model_name=model_res).delete()
            except Exception as e:
                print(f"[WARN] Failed to delete {model_res}: {e}")

if __name__ == "__main__":
    main()

