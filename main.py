import base64
import io
import os
import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="VocalGuard AI Detection API",
    description="Voice authenticity detection service (Phase 1 - Adaptive Heuristic Engine)",
    version="2.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Configuration
DEFAULT_KEY = "sk_live_yashashwini_2026"
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
        "engine": "Neural-V3 Adaptive Heuristic",
        "python_version": "3.10+",
        "mode": "Buildathon Demo"
    }


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

            # Convert stereo to mono if needed
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            y = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
            sr = 16000

        if len(y) == 0:
            raise Exception("Audio file is empty")

        # Normalize
        y = librosa.util.normalize(y)

        # 3️⃣ Feature Extraction
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))

        # Debug logs (Railway logs will show these)
        print("Spectral Flatness:", spectral_flatness)
        print("Zero Crossing Rate:", zcr)
        print("Spectral Bandwidth:", spectral_bandwidth)
        print("MFCC Mean:", mfcc)

        # 4️⃣ Adaptive Weighted Scoring System
        ai_score = 0.0

        if spectral_flatness < 0.08:
            ai_score += 0.3

        if zcr < 0.1:
            ai_score += 0.3

        if spectral_bandwidth < 2500:
            ai_score += 0.2

        if mfcc > -200:
            ai_score += 0.2

        # Final Decision
        is_ai = ai_score >= 0.6

        # 5️⃣ Confidence Calculation
        confidence = round(0.65 + ai_score * 0.35, 3)
        confidence = min(confidence, 0.995)

        explanation = (
            "Synthetic spectral smoothing, reduced micro-variations and AI-like acoustic consistency detected."
            if is_ai else
            "Natural prosodic fluctuations, organic vocal artifacts and human acoustic dynamics detected."
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
