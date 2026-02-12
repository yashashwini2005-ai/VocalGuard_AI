📌 Project Overview

VocalGuard AI is a machine learning–based REST API that detects whether an audio sample is:
AI-generated (synthetic speech)
Real human speech
The system is language-agnostic and works across multiple languages by analyzing acoustic forensic features instead of text content.

This solution addresses risks related to:
Voice-based fraud
Deepfake impersonation
Scam detection
Synthetic media verification
Audio forensics

🧠 How It Works

The system processes Base64-encoded audio and performs:
Audio Decoding & Normalization
Acoustic Feature Extraction
Pitch mean & variance
MFCC (Mel-Frequency Cepstral Coefficients)
Spectral centroid
RMS energy variation
Zero-crossing rate
Machine Learning Classification
Logistic Regression
StandardScaler normalization
Probabilistic output
Structured JSON Response

The model was trained on:

Real speech samples (LibriSpeech)
AI-generated speech samples (ElevenLabs)

🚀 API Specification
Endpoint
POST /api/voice-detection

Headers
x-api-key: sk_live_vocalguard_2026
Content-Type: application/json

Request Body
{
"language": "English",
"audioFormat": "wav",
"audioBase64": "<FULL_BASE64_AUDIO_STRING>"
}

Response Example
{
"status": "success",
"language": "English",
"classification": "AI_GENERATED",
"confidenceScore": 0.87,
"explanation": "ML model detected synthetic acoustic patterns."
}

🛠️ Run Locally
Prerequisites

Python 3.10+

pip

1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Start the Server
uvicorn main:app --reload

Server will run at:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs

3️⃣ Test the API

Use the included testing script:
python test_api.py
Or use Swagger UI to manually send Base64 audio.

📊 Model Details

Model: Logistic Regression
Feature Scaling: StandardScaler
Training Strategy: Balanced dataset (Real vs AI)
Output: Probabilistic classification

📁 Project Structure
VocalGuard_AI/
│
├── main.py
├── train_model.py
├── test_api.py
├── voice_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── .gitignore

⚠️ Limitations

Very short audio (<2 seconds) may reduce confidence.

Extremely high-quality synthetic voices may resemble human speech.

Noisy environments can affect acoustic feature extraction.

🏆 Applications

Fraud prevention

Scam call detection

Deepfake voice verification

AI compliance monitoring

Media authenticity verification
