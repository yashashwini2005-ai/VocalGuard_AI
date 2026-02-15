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

# =====================================
# CONFIG (MUST MATCH TRAINING EXACTLY)
# =====================================

SAMPLE_RATE = 22050
FIXED_DURATION = 4   # MUST MATCH train_model.py

MODEL_PATH = "english_voice_model.pkl"
SCALER_PATH = "english_scaler.pkl"

# =====================================
# LOAD MODEL + SCALER
# =====================================

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise Exception("Model files not found. Run train_model.py first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =====================================
# FASTAPI APP
# =====================================

app = FastAPI(
    title="VocalGuard AI Detection API",
    description="RandomForest-Based AI Voice Detection System",
    version="10.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# API KEY SECURITY
# =====================================

API_KEY = "sk_live_vocalguard_2026"
api_key_header = APIKeyHeader(name="x-api-key")

def verify_api_key(api_key: str):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Invalid API key"}
        )

# =====================================
# REQUEST / RESPONSE MODELS
# =====================================

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

# =====================================
# HEALTH CHECK
# =====================================

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "engine": "RandomForest + MFCC + Delta + Spectral + Contrast + Tonnetz",
        "mode": "English Production Model (218 Features)"
    }

# =====================================
# FEATURE EXTRACTION (MUST MATCH TRAINING)
# =====================================

def extract_features(y, sr):

    y, _ = librosa.effects.trim(y)

    required_length = SAMPLE_RATE * FIXED_DURATION
    if len(y) < required_length:
        y = np.pad(y, (0, required_length - len(y)))
    else:
        y = y[:required_length]

    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    zero_crossing = librosa.feature.zero_crossing_rate(y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    features = np.hstack([
        np.mean(mfcc.T, axis=0), np.std(mfcc.T, axis=0),
        np.mean(delta.T, axis=0), np.std(delta.T, axis=0),
        np.mean(spectral_centroid.T, axis=0), np.std(spectral_centroid.T, axis=0),
        np.mean(spectral_rolloff.T, axis=0), np.std(spectral_rolloff.T, axis=0),
        np.mean(spectral_bandwidth.T, axis=0), np.std(spectral_bandwidth.T, axis=0),
        np.mean(zero_crossing.T, axis=0), np.std(zero_crossing.T, axis=0),
        np.mean(chroma.T, axis=0), np.std(chroma.T, axis=0),
        np.mean(spectral_contrast.T, axis=0), np.std(spectral_contrast.T, axis=0),
        np.mean(tonnetz.T, axis=0), np.std(tonnetz.T, axis=0),
    ])

    return features

# =====================================
# VOICE DETECTION ENDPOINT
# =====================================

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

        # Load audio safely
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

        # Extract features
        features = extract_features(y, sr)

        # Scale
        features_scaled = scaler.transform([features])

        # Predict
        probabilities = model.predict_proba(features_scaled)[0]
        ai_probability = probabilities[1]

        threshold = 0.55
        is_ai = ai_probability > threshold

        confidence = round(
            ai_probability if is_ai else (1 - ai_probability),
            3
        )

        explanation = (
            "Detected synthetic acoustic patterns and reduced natural vocal variability."
            if is_ai else
            "Detected natural human vocal dynamics and organic speech variations."
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

# =====================================
# RUN SERVER
# =====================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
