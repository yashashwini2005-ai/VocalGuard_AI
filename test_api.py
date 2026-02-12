import base64
import requests
import os

API_URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "sk_live_vocalguard_2026"

AUDIO_FOLDER = "dataset/ai"   # change to dataset/real to test human voices


def encode_audio(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


if not os.path.exists(AUDIO_FOLDER):
    print("Folder not found:", AUDIO_FOLDER)
    exit()

files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith((".wav", ".mp3", ".flac"))]

print("Total audio files found:", len(files))

for file in files:
    file_path = os.path.join(AUDIO_FOLDER, file)

    print("\n----------------------------------")
    print("Processing file:", file)

    try:
        audio_base64 = encode_audio(file_path)

        payload = {
            "language": "English",
            "audioFormat": "wav",
            "audioBase64": audio_base64
        }

        headers = {
            "x-api-key": API_KEY
        }

        response = requests.post(API_URL, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            print("Classification:", result.get("classification"))
            print("Confidence:", result.get("confidenceScore"))
        else:
            print("Error:", response.status_code)
            print("Response:", response.text)

    except Exception as e:
        print("Failed to process file:", file)
        print("Error:", str(e))
