import librosa
import numpy as np
import soundfile as sf
import tempfile

def extract_log_mel_spectrogram(uploaded_file, sr=22050, duration=3):
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path, sr=sr, duration=duration)
    y = librosa.util.fix_length(y, size=sr*duration)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec.T  # Shape: (time, 128)
