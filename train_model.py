import os
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Path ke dataset ESC-50
DATASET_PATH = "ESC-50-master/audio"
CSV_PATH = "ESC-50-master/meta/esc50.csv"

# Load metadata
meta = pd.read_csv(CSV_PATH)

# Fungsi ekstraksi fitur MFCC
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# List untuk fitur dan label
features = []
labels = []

# Loop data
for index, row in meta.iterrows():
    file_name = row["filename"]
    label = row["category"]
    file_path = os.path.join(DATASET_PATH, file_name)
    try:
        feat = extract_features(file_path)
        features.append(feat)
        labels.append(label)
    except Exception as e:
        print(f"Gagal ekstrak fitur: {file_path}, Error: {e}")

# Konversi ke array
X = np.array(features)
y = np.array(labels)

# Split data & latih model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Simpan model
joblib.dump(model, "model.pkl")
print("✅ Model disimpan sebagai model.pkl")
