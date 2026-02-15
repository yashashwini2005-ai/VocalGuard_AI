# 🎙️ VocalGuard AI  
### AI Voice Detection System using Classical Machine Learning

VocalGuard AI is a production-ready voice authenticity detection system that classifies whether an audio clip is **Human** or **AI-Generated** using acoustic feature engineering and a RandomForest ensemble model.

The system is built with Python, Librosa, Scikit-learn, and FastAPI, and achieves **~91% validation F1-score** and **~97.7% real-world API accuracy**.

---

## 🚀 Key Features

- 🎧 AI vs Human Voice Classification
- 📊 218-Dimensional Acoustic Feature Extraction
- 🌲 RandomForest Ensemble Model
- ⚡ FastAPI Production API
- 🔐 API Key Authentication
- 📦 Clean GitHub-ready architecture
- 🧪 Batch Testing Support

---

## 🧠 Model Overview

### 📌 Dataset
- ~2000 balanced English audio samples
  - 1001 Real voices
  - 1000 AI-generated voices

### 📌 Feature Engineering (218 Features)
Extracted using Librosa:

- 40 MFCCs (mean + std)
- Delta coefficients
- Spectral Centroid
- Spectral Rolloff
- Spectral Bandwidth
- Zero Crossing Rate
- Chroma Features
- Spectral Contrast
- Tonnetz Features

All features are normalized and fixed to 4-second duration audio clips at 22050 Hz.

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~90.8% |
| F1-Score | ~0.91 |
| API Batch Accuracy | **97.7%** |
| Feature Count | 218 |
| Model Type | RandomForestClassifier |

---

## 🏗️ Project Architecture

VocalGuard_AI/
│
├── main.py # FastAPI inference server
├── train_model.py # Model training script
├── test_api.py # Batch API tester
├── convert_to_base64.py # Audio encoding utility
├── requirements.txt # Dependencies
├── runtime.txt # Python runtime version
├── README.md # Documentation
└── .gitignore # Ignored files


---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yashashwini2005-ai/VocalGuard_AI.git
cd VocalGuard_AI


2️⃣ Create Virtual Environment (Python 3.10 Recommended)
py -3.10 -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt


📊 Train the Model
Place dataset in:
dataset/English/Real
dataset/English/Fake


Then run:
python train_model.py
This generates:
english_voice_model.pkl
english_scaler.pkl


🌐 Run the API
http://127.0.0.1:8000/docs


Access:
http://127.0.0.1:8000/docs

Swagger UI will open automatically.


📡 API Endpoint
POST /api/voice-detection

Request JSON:
{
  "language": "English",
  "audioFormat": "wav",
  "audioBase64": "BASE64_STRING"
}
Response:
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.947,
  "explanation": "Detected synthetic acoustic patterns and reduced natural vocal variability."
}


🧪 Batch Testing
To test entire folder:
python test_api.py

Example Result:
Total Tested : 1000
Correct      : 977
Accuracy     : 97.7%



🔐 Security
API Key authentication required
CORS enabled for development
Production-ready structure



🛠️ Technologies Used
Python 3.10
Librosa
NumPy
Scikit-learn
FastAPI
Uvicorn
Joblib



🎯 Future Improvements
Multi-language expansion
Deep learning model integration
Explainable AI feature importance dashboard
Cloud deployment (Render / AWS / GCP)
Real-time streaming detection




👩‍💻 Author

Yashashwini
AI & Machine Learning Developer
GitHub: https://github.com/yashashwini2005-ai



⭐ Project Goal

VocalGuard AI aims to combat misinformation and deepfake voice misuse by providing a lightweight, scalable, and production-ready AI voice detection system.
