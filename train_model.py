import os
import numpy as np
import librosa
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.normalize(y)

    # Pitch
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

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    return [pitch_mean, pitch_var, mfcc, centroid, rms, zcr]

X = []
y = []

for label, folder in [(0, "dataset/real"), (1, "dataset/ai")]:
    for file in os.listdir(folder):
        if file.endswith((".wav", ".flac", ".mp3")):
            path = os.path.join(folder, file)
            features = extract_features(path)
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Validation Accuracy:", accuracy)

joblib.dump(model, "voice_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully.")
