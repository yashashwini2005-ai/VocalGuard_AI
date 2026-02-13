import base64
import io
import os
import librosa
import numpy as np
import soundfile as sf
import joblib

from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# ---------------------------
# Load RandomForest Model
# ---------------------------

model = joblib.load("voice_model.pkl")
scaler = joblib.load("scaler.pkl")

SAMPLE_RATE = 16000

# ---------------------------
# FastAPI App
# ---------------------------

app = FastAPI(
    title="VocalGuard AI Detection API",
    description="RandomForest-Based AI Voice Detection System",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# API Key Setup
# ---------------------------

API_KEY = "sk_live_vocalguard_2026"

api_key_header = APIKeyHeader(name="x-api-key")

def verify_api_key(api_key: str):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Invalid API key"}
        )

# ---------------------------
# Request / Response Models
# ---------------------------

class DetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


class DetectionResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str

# ---------------------------
# Health Endpoint
# ---------------------------

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "engine": "RandomForest Acoustic Model",
        "mode": "Production Version"
    }

# ---------------------------
# Feature Extraction (Same as Training)
# ---------------------------

def extract_features(y, sr):

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(spectral_contrast.T, axis=0)

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # RMS Energy
    rms = np.mean(librosa.feature.rms(y=y))

    return np.hstack([mfcc_mean, chroma_mean, contrast_mean, zcr, rms])

# ---------------------------
# Voice Detection Endpoint
# ---------------------------

@app.post("/api/voice-detection", response_model=DetectionResponse)
async def detect_voice(
    request: DetectionRequest,
    api_key: str = Security(api_key_header)
):
    verify_api_key(api_key)

    try:
        # Decode Base64
        audio_bytes = base64.b64decode(request.audioBase64)
        audio_stream = io.BytesIO(audio_bytes)

        # Load Audio
        try:
            y, sr = librosa.load(audio_stream, sr=SAMPLE_RATE)
        except Exception:
            audio_stream.seek(0)
            data, samplerate = sf.read(audio_stream)

            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            y = librosa.resample(data, orig_sr=samplerate, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        if len(y) == 0:
            raise Exception("Audio file is empty")

        y = librosa.util.normalize(y)

        # Extract features
        features = extract_features(y, sr)
        features_scaled = scaler.transform([features])

        # Predict probabilities
        probabilities = model.predict_proba(features_scaled)[0]
        ai_probability = probabilities[1]

        # Threshold tuning (adjust if needed)
        threshold = 0.6

        is_ai = ai_probability > threshold

        confidence = round(
            ai_probability if is_ai else (1 - ai_probability),
            3
        )

        explanation = (
            "Model detected synthetic acoustic patterns."
            if is_ai else
            "Model detected natural human vocal characteristics."
        )

        return {
            "status": "success",
            "language": request.language,
            "classification": "AI_GENERATED" if is_ai else "HUMAN",
            "confidenceScore": confidence,
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": str(e)}
        )

# ---------------------------
# Run Server
# ---------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
