import base64
import requests
import os
import sys

# =====================================
# CONFIG
# =====================================

API_URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "sk_live_vocalguard_2026"

# 🔥 CHANGE THIS TO TEST
AUDIO_FOLDER = "dataset/English/Fake"
# Use:
# "dataset/English/Fake"  → Test AI voices
# "dataset/English/Real"  → Test Human voices

LANGUAGE = "English"

# =====================================
# AUTO DETECT EXPECTED LABEL
# =====================================

if "fake" in AUDIO_FOLDER.lower():
    EXPECTED_LABEL = "AI_GENERATED"
elif "real" in AUDIO_FOLDER.lower():
    EXPECTED_LABEL = "HUMAN"
else:
    print("⚠ Folder must contain 'Real' or 'Fake'")
    sys.exit()

# =====================================
# Encode Audio
# =====================================

def encode_audio(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print("❌ Error reading:", file_path, e)
        return None

# =====================================
# Validate Folder
# =====================================

if not os.path.exists(AUDIO_FOLDER):
    print("❌ Folder not found:", AUDIO_FOLDER)
    sys.exit()

files = [
    f for f in os.listdir(AUDIO_FOLDER)
    if f.lower().endswith((".wav", ".mp3", ".flac"))
]

if len(files) == 0:
    print("⚠ No audio files found.")
    sys.exit()

print("\n======================================")
print("Testing Folder :", AUDIO_FOLDER)
print("Expected Label :", EXPECTED_LABEL)
print("Total Files    :", len(files))
print("======================================")

correct = 0
total = 0
failed = 0

# =====================================
# TEST LOOP
# =====================================

for file in files:
    file_path = os.path.join(AUDIO_FOLDER, file)

    print("\n----------------------------------")
    print("Processing:", file)

    audio_base64 = encode_audio(file_path)
    if audio_base64 is None:
        failed += 1
        continue

    extension = file.split(".")[-1].lower()

    payload = {
        "language": LANGUAGE,
        "audioFormat": extension,
        "audioBase64": audio_base64
    }

    headers = {
        "x-api-key": API_KEY
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers=headers,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()

            classification = result.get("classification")
            confidence = result.get("confidenceScore")

            print("Prediction :", classification)
            print("Confidence :", confidence)

            total += 1

            if classification == EXPECTED_LABEL:
                correct += 1
                print("Result     : ✅ Correct")
            else:
                print("Result     : ❌ Wrong")

        else:
            print("❌ API Error:", response.status_code)
            print("Response:", response.text)
            failed += 1

    except Exception as e:
        print("❌ Request failed:", e)
        failed += 1

# =====================================
# FINAL SUMMARY
# =====================================

if total > 0:
    accuracy = round((correct / total) * 100, 2)

    print("\n======================================")
    print("📊 TEST SUMMARY")
    print("Total Tested :", total)
    print("Correct      :", correct)
    print("Failed       :", failed)
    print("Accuracy     :", accuracy, "%")
    print("======================================")
else:
    print("\n⚠ No successful tests completed.")
