Dataset (CSV in Cloud Storage)
   ↓
Vertex AI Custom Training Job (prebuilt scikit-learn container)
   ↓
Trained model saved to GCS
   ↓
Vertex AI Endpoint (for online prediction)
   ↓
(Optionally) Pipeline automates these steps