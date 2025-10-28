from google.cloud import aiplatform

PROJECT_ID = "sentiment-demo-476518"
REGION = "us-central1"
BUCKET = "sentiment-demo-bucket"

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

job = aiplatform.CustomTrainingJob(
    display_name="tweet-sentiment-train",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest",
    requirements=[
        "pandas==2.2.3",
        "scikit-learn==1.6.1",
        "joblib==1.4.2",
        "python-json-logger==2.0.7",
    ],
)
job.run(
    replica_count=1,
    machine_type="n1-standard-4",
)