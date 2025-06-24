import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile
import joblib

# === Streamlit Page Config ===
st.set_page_config(page_title="Emotion Classifier", page_icon="🎧")
st.title("🎧 Speech & Song Emotion Recognition")
st.markdown("Upload a `.wav` or `.mp3` audio file to predict the **emotion** expressed in it.")

# === Feature Extraction Functions ===
def mfcc_values(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    return np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

def delta_values(filename):
    mfcc = mfcc_values(filename)
    return librosa.feature.delta(mfcc)

def log_mel_values(file_path, duration=3, offset=0.5, n_mels=128):
    y, sr = librosa.load(file_path, duration=duration, offset=offset)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec)
    return np.mean(log_mel_spec.T, axis=0)

def zcr_values(file_name):
    sig, sr = librosa.load(file_name, duration=3, offset=0.5)
    return np.mean(librosa.feature.zero_crossing_rate(sig).T, axis=0)

def spectral_features(file_path, duration=3, offset=0.5):
    y, sr = librosa.load(file_path, duration=duration, offset=offset)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    return np.hstack([centroid, bandwidth, rolloff, flatness, contrast])

def extract_features(file_name):
    mfcc = mfcc_values(file_name)
    delta = delta_values(file_name)
    log_mel = log_mel_values(file_name)
    zcr = zcr_values(file_name)
    spectral = spectral_features(file_name)
    return np.hstack([mfcc, delta, log_mel, zcr, spectral])

# === Load Model and Label Encoder ===
model = joblib.load("model/trained_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# === File Upload and Prediction ===
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.write("🔄 Extracting features...")

    try:
        # Write uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Extract features
        features = extract_features(tmp_path).reshape(1, -1)

        # Predict class
        predicted_index = model.predict(features)[0]
        predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]

        # Predict probability (confidence)
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba(features))

        # Show output
        st.success(f"🎯 Predicted Emotion: **{predicted_emotion.upper()}**")
        if confidence:
            st.info(f"🔎 Model confidence: **{confidence:.2f}**")

    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")

