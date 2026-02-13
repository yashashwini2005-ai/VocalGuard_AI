import os
import random
import librosa
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ======================================
# CONFIG
# ======================================
DATASET_ROOT = "dataset"
SAMPLE_RATE = 16000
LANGUAGES = ["English", "Hindi"]

HINDI_LIMIT = 14  # 🔥 Force 14 human and 14 nonhuman

X = []
y = []

# ======================================
# FEATURE EXTRACTION
# ======================================
def extract_features(audio, sr):

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    contrast_mean = np.mean(spectral_contrast.T, axis=0)

    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms = np.mean(librosa.feature.rms(y=audio))

    return np.hstack([mfcc_mean, chroma_mean, contrast_mean, zcr, rms])


# ======================================
# LOAD DATA
# ======================================
for language in LANGUAGES:

    print(f"\n===== Loading {language} Data =====")

    for label_name in ["human", "nonhuman"]:

        folder_path = os.path.join(DATASET_ROOT, language, label_name)

        if not os.path.exists(folder_path):
            print(f"⚠ Skipping missing folder: {folder_path}")
            continue

        files = os.listdir(folder_path)

        # 🔥 BALANCE HINDI ONLY
        if language == "Hindi":
            random.shuffle(files)
            files = files[:HINDI_LIMIT]

        print(f"Processing {language} - {label_name}: {len(files)} files")

        for file in files:
            file_path = os.path.join(folder_path, file)

            try:
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                features = extract_features(audio, sr)

                X.append(features)
                y.append(0 if label_name == "human" else 1)

            except:
                print("Error:", file_path)

X = np.array(X)
y = np.array(y)

print("\nTotal samples:", len(X))

# ======================================
# NORMALIZE
# ======================================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ======================================
# SPLIT
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================
# RANDOM FOREST MODEL
# ======================================
model = RandomForestClassifier(
    n_estimators=400,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# ======================================
# EVALUATE
# ======================================
y_pred = model.predict(X_test)

print("\nFinal Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ======================================
# SAVE MODEL
# ======================================
joblib.dump(model, "voice_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nBalanced Multilingual RandomForest model saved successfully!")
