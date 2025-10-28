from google.cloud import aiplatform

model = aiplatform.Model.upload(
    display_name="tweet-sentiment-model",
    artifact_uri=job.output_dir,
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
)
endpoint = model.deploy(machine_type="n1-standard-2")