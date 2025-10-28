gcloud auth application-default login
gcloud projects create sentiment-demo
gcloud config set project sentiment-demo-476518

gcloud services enable aiplatform.googleapis.com storage.googleapis.com

gsutil mb gs://sentiment-demo-bucket/
gsutil cp tweets.csv gs://sentiment-demo-bucket/data/tweets.csv

set PROJECT_ID=sentiment-demo-476518
set BUCKET=sentiment-demo-bucket
for /f "tokens=* usebackq" %A in (`gcloud projects describe %PROJECT_ID% --format="value(projectNumber)"`) do set PROJECT_NUMBER=%A

# Grant bucket-level access to the Vertex AI service agent
gsutil iam ch serviceAccount:service-%PROJECT_NUMBER%@gcp-sa-aiplatform-cc.iam.gserviceaccount.com:roles/storage.objectAdmin gs://%BUCKET%

# Grant bucket-level access to the training job runtime SA (compute default)
gsutil iam ch serviceAccount:%PROJECT_NUMBER%-compute@developer.gserviceaccount.com:roles/storage.objectAdmin gs://%BUCKET%
