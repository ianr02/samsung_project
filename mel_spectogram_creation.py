import os
import numpy as np
import librosa
from scipy.ndimage import laplace

def process_dataset(f, sr=16000, n_mels=64, n_fft=1024, hop_length=128, duration=7):
    y, _ = librosa.load(f, sr=sr)
    target_length = sr * duration
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        y = y[:target_length]
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    S = librosa.power_to_db(S, ref=np.max)
    return S

audio_path = 'audio'
new_dataset_path = 'mel_spectogram'
os.makedirs(new_dataset_path, exist_ok=True)
for filename in sorted(os.listdir('audio')):
    file_path = os.path.join(audio_path, filename)
    log_S = process_dataset(file_path)
    lap = laplace(log_S)
    variance = np.var(lap)
    print(f"{filename} Laplacian variance:", variance)
    npy_path = os.path.join(
        new_dataset_path,
        f"{os.path.splitext(filename)[0]}.npy"
    )
    np.save(npy_path, log_S)