import os
import numpy as np
import librosa
from PIL import Image
from scipy.ndimage import laplace

def process_dataset(f, sr=16000, n_mels=256, n_fft=1024, hop_length=32, duration=5):
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
    log_S = librosa.power_to_db(S, ref=np.max, top_db=80)
    S_norm = (log_S + 80) / 80
    return S_norm

audio_path = 'audio'
new_dataset_path = 'mel_spectogram'
os.makedirs(new_dataset_path, exist_ok=True)
for filename in sorted(os.listdir(audio_path)):
    file_path = os.path.join(audio_path, filename)
    mel_spec = process_dataset(file_path)
    lap = laplace(mel_spec)
    variance = np.var(lap)
    print("Laplacian variance:", variance)
    im = (mel_spec * 255).astype(np.uint8)
    img = Image.fromarray(im)
    img.save(os.path.join(new_dataset_path, f"{filename[:-4]}.png"))