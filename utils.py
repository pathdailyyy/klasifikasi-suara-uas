import librosa
import numpy as np
import joblib

# Load model (bisa dilakukan sekali saja saat module dibuka)
model = joblib.load("model.pkl")  # Pastikan file model.pkl ada di folder yang sama dengan utils.py

def predict(audio_file):
    # Load audio
    y, sr = librosa.load(audio_file, sr=None)

    # Ekstrak fitur (MFCC)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Prediksi
    prediction = model.predict([mfcc_mean])[0]

    # Tambahan: ekstrak mel spectrogram jika perlu ditampilkan
    mel = librosa.feature.melspectrogram(y=y, sr=sr)

    return prediction, mel
