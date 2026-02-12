import base64
import io
import os
import librosa
import numpy as np
import soundfile as sf
import joblib

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ---------------------------
# Load Trained ML Model
# ---------------------------

model = joblib.load("voice_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------
# FastAPI App
# ---------------------------

app = FastAPI(
    title="VocalGuard AI Detection API",
    description="Multilingual AI-Generated Voice Detection System",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# API Key Configuration
# ---------------------------

DEFAULT_KEY = "sk_live_vocalguard_2026"
VALID_API_KEYS = {os.getenv("PRODUCTION_API_KEY", DEFAULT_KEY): "Production_User"}

# ---------------------------
# Request & Response Models
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
# API Key Verification
# ---------------------------

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if not x_api_key or x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Invalid API key"}
        )
    return x_api_key


# ---------------------------
# Health Endpoint
# ---------------------------

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "engine": "Logistic Regression + Acoustic Forensics",
        "mode": "Buildathon Final Version"
    }


# ---------------------------
# Feature Extraction Function
# ---------------------------

def extract_features(y, sr):
    # Pitch features
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    pitch_var = np.var(pitch_values) if len(pitch_values) > 0 else 0

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))

    # Spectral centroid
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # RMS energy variation
    rms = np.std(librosa.feature.rms(y=y))

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    return [pitch_mean, pitch_var, mfcc, centroid, rms, zcr]


# ---------------------------
# Voice Detection Endpoint
# ---------------------------

@app.post("/api/voice-detection", response_model=DetectionResponse)
async def detect_voice(request: DetectionRequest, apiKey: str = Depends(verify_api_key)):
    try:
        # 1️⃣ Decode Base64
        try:
            audio_bytes = base64.b64decode(request.audioBase64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid Base64 string")

        audio_stream = io.BytesIO(audio_bytes)

        # 2️⃣ Load Audio
        try:
            y, sr = librosa.load(audio_stream, sr=16000)
        except Exception:
            audio_stream.seek(0)
            data, samplerate = sf.read(audio_stream)

            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            y = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
            sr = 16000

        if len(y) == 0:
            raise Exception("Audio file is empty")

        y = librosa.util.normalize(y)

        # 3️⃣ Extract Features
        features = extract_features(y, sr)
        features_scaled = scaler.transform([features])

        # 4️⃣ ML Prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][prediction]

        is_ai = prediction == 1
        confidence = round(float(probability), 3)

        explanation = (
            "ML model detected synthetic acoustic patterns."
            if is_ai else
            "ML model detected natural human vocal characteristics."
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
