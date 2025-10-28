# Train Local
python train_model.py --data_uri gs://sentiment-demo-bucket/data/tweets.csv --bucket sentiment-demo-bucket --model_name sentiment-v1

# Train on the custom job
python custom_vertex_ai_job.py --project_id sentiment-demo-476518 --region us-central1 --staging_bucket gs://sentiment-demo-bucket --data_uri gs://sentiment-demo-bucket/data/tweets.csv --bucket sentiment-demo-bucket --model_name sentiment-v1

# Deploy
python deploy_model.py --project_id sentiment-demo-476518 --region us-central1 --artifact_dir gs://sentiment-demo-bucket/models/sentiment-v1/ --display_name sentiment-model --endpoint_name sentiment-endpoint

# Inference
python inference_model.py --project_id sentiment-demo-476518 --region us-central1 --endpoint_id 6638806128534749184 --texts "I love my school" "This exam is terrible" "Homework ruined my weekend"

# Shutdown
python shutdown_mode.py --project_id sentiment-demo-476518 --region us-central1 --endpoint_id 6638806128534749184 --delete_endpoint --delete_model
