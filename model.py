import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

DATASET_PATH = "ESC-50-master"
META_FILE = os.path.join(DATASET_PATH, "meta/esc50.csv")
AUDIO_PATH = os.path.join(DATASET_PATH, "audio")

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def train_model():
    meta = pd.read_csv(META_FILE)
    features, labels = [], []

    for _, row in meta.iterrows():
        file_path = os.path.join(AUDIO_PATH, row["filename"])
        try:
            mfcc = extract_features(file_path)
            features.append(mfcc)
            labels.append(row["category"])
        except Exception as e:
            print(f"Error with {file_path}: {e}")

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(clf, "model.joblib")

if __name__ == "__main__":
    train_model()
