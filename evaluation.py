import os
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load model
model = joblib.load("voice_model.pkl")
scaler = joblib.load("scaler.pkl")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.normalize(y)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    pitch_var = np.var(pitch_values) if len(pitch_values) > 0 else 0

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rms = np.std(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    return [pitch_mean, pitch_var, mfcc, centroid, rms, zcr]


X = []
y_true = []

# Load real voices (label = 0)
for file in os.listdir("dataset/real"):
    if file.endswith((".wav", ".flac", ".mp3")):
        features = extract_features(os.path.join("dataset/real", file))
        X.append(features)
        y_true.append(0)

# Load AI voices (label = 1)
for file in os.listdir("dataset/ai"):
    if file.endswith((".wav", ".flac", ".mp3")):
        features = extract_features(os.path.join("dataset/ai", file))
        X.append(features)
        y_true.append(1)

X = np.array(X)
y_true = np.array(y_true)

X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

# Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\nAccuracy:", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall:", round(rec, 4))
print("F1 Score:", round(f1, 4))

# -----------------------
# Plot Confusion Matrix
# -----------------------

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Real", "AI"],
            yticklabels=["Real", "AI"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
