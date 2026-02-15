import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# =====================================
# CONFIGURATION
# =====================================

BASE_PATH = os.path.join("dataset", "English")
SAMPLE_RATE = 22050
FIXED_DURATION = 4   # MUST match main.py
MODEL_NAME = "english_voice_model.pkl"
SCALER_NAME = "english_scaler.pkl"

# =====================================
# FEATURE EXTRACTION (218 FEATURES)
# =====================================

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        # Remove silence
        y, _ = librosa.effects.trim(y)

        # Fix duration
        required_length = SAMPLE_RATE * FIXED_DURATION
        if len(y) < required_length:
            y = np.pad(y, (0, required_length - len(y)))
        else:
            y = y[:required_length]

        # Normalize
        y = librosa.util.normalize(y)

        # === Feature Extraction ===

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        zero_crossing = librosa.feature.zero_crossing_rate(y)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

        # Stack mean + std (ORDER MUST MATCH API)
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

    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None


# =====================================
# LOAD DATASET
# =====================================

print("🔄 Loading dataset...")

X = []
y_labels = []

for label_name, label_value in [("Real", 0), ("Fake", 1)]:
    folder = os.path.join(BASE_PATH, label_name)

    if not os.path.exists(folder):
        raise Exception(f"❌ Folder not found: {folder}")

    files = [f for f in os.listdir(folder)
             if f.lower().endswith((".wav", ".mp3", ".flac"))]

    print(f"📁 {label_name}: {len(files)} files")

    for file in files:
        file_path = os.path.join(folder, file)
        features = extract_features(file_path)

        if features is not None:
            X.append(features)
            y_labels.append(label_value)

X = np.array(X)
y_labels = np.array(y_labels)

if len(X) == 0:
    raise Exception("❌ No valid audio files found!")

print("✅ Feature shape:", X.shape)   # Should be (?, 218)

# =====================================
# TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_labels,
    test_size=0.2,
    random_state=42,
    stratify=y_labels
)

# =====================================
# SCALING
# =====================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# MODEL TRAINING
# =====================================

print("🚀 Training RandomForest...")

model = RandomForestClassifier(
    n_estimators=300,     # Balanced speed + performance
    max_depth=30,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =====================================
# EVALUATION
# =====================================

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print(f"\n🎯 Accuracy: {accuracy:.4f}\n")

print("📊 Classification Report:")
print(classification_report(y_test, pred))

print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, pred))

# =====================================
# SAVE MODEL
# =====================================

joblib.dump(model, MODEL_NAME)
joblib.dump(scaler, SCALER_NAME)

print("\n💾 Model saved successfully!")
print("✅ Training complete.")
