import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile
import joblib

# === Feature Extraction Functions ===
def mfcc_values(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

def delta_values(filename):
    mfcc = mfcc_values(filename)
    delta_mfcc = librosa.feature.delta(mfcc)
    return delta_mfcc

def log_mel_values(file_path, duration=3, offset=0.5, n_mels=128):
    y, sr = librosa.load(file_path, duration=duration, offset=offset)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec)
    log_mel_mean = np.mean(log_mel_spec.T, axis=0)
    return log_mel_mean

def zcr_values(file_name):
    sig, sr = librosa.load(file_name, duration=3, offset=0.5)
    zcr = np.mean(librosa.feature.zero_crossing_rate(sig).T, axis=0)
    return zcr

def spectral_features(file_path, duration=3, offset=0.5):
    y, sr = librosa.load(file_path, duration=duration, offset=offset)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    spectral_feats = np.hstack([centroid, bandwidth, rolloff, flatness, contrast])
    return spectral_feats

def extract_features(file_name):
    mfcc = mfcc_values(file_name)
    delta = delta_values(file_name)
    log_mel = log_mel_values(file_name)
    zcr = zcr_values(file_name)
    spectral = spectral_features(file_name)
    all_features = np.hstack([mfcc, delta, log_mel, zcr, spectral])
    return all_features

# === Load RandomForest Model ===
model = joblib.load("model/trained_model.pkl")

# === Streamlit UI ===
st.title("🎧 Speech & Song Emotion Recognition")

uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.write("Processing...")

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        features = extract_features(tmp_path)
        features = np.expand_dims(features, axis=0)

        prediction = model.predict(features)
        predicted_label = int(prediction[0])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)
            confidence = np.max(probs)
        else:
            confidence = None

        label_map = {
            0: "angry",
            1: "calm",
            2: "happy",
            3: "sad",
            4: "fearful",
            5: "disgust"
        }

        emotion = label_map.get(predicted_label, "Unknown")
        st.success(f"Predicted Emotion: **{emotion.upper()}** 🎯")

        if confidence:
            st.write(f"Model confidence: **{confidence:.2f}**")

    except Exception as e:
        st.error(f"Error processing audio: {e}")
