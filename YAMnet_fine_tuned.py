import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import librosa
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from keras.models import Model
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbCallback
from sklearn.model_selection import train_test_split
from collections import Counter


def extract_context_from_filename(filename):
    name = filename.lower()
    if "braking_state" in name:
        return 1
    elif "moving_state" in name:
        return 2
    elif "startup_state" in name:
        return 3
    elif "idle_state" in name:
        return 4
    else:
        return 0

def load_audio(path):
    y, sr = librosa.load(path, sr=16000)   # YAMNet expects 16kHz
    return y.astype(np.float32)

def get_embedding(path):
    waveform = load_audio(path)
    # YAMNet wants a tensor 1D float32
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

    scores, embeddings, spectrogram = yamnet(waveform)
    embedding = tf.reduce_mean(embeddings, axis=0)  # (1024,)
    return embedding.numpy()

wandb.login()

exper = wandb.init(project="car_fault_detection",
                   name="yamnet_v2",
                   config={
                        "epochs": 30,
                        "batch_size": 32,
                        "model": "YAMNet+MLP",
                        "context_embedding_dim": 4,
                        "dense_units": [256, 128]
    }
)

config = wandb.config

wandb_callback = [WandbMetricsLogger(), WandbModelCheckpoint(filepath='model.keras', monitor='val_loss', save_best_only=True)]

#DATA_PATH = "/content/drive/MyDrive/project_samsung_IA/"

df = pd.read_csv("metadata.csv")
df["context"] = df["file"].apply(extract_context_from_filename)
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
print("YAMNet loaded")

test_path = os.path.join("audio", df.iloc[0]["file"])
emb = get_embedding(test_path)

X_audio = []
X_context = []
y = []

for _, row in df.iterrows():
    filepath = os.path.join("audio", row["file"])

    emb = get_embedding(filepath)  # 1024-d vector

    X_audio.append(emb)
    X_context.append(int(row["context"]))
    y.append(row["label"])

X_audio = np.array(X_audio)
X_context = np.array(X_context)
y = np.array(y)

counter = Counter(y)
max_count = max(counter.values())

X_audio_new = []
X_context_new = []
y_new = []

for cls in counter:
    # get indices of this class
    idx = np.where(y == cls)[0]
    n_repeat = max_count // len(idx)
    n_extra = max_count % len(idx)

    # repeat full features
    X_audio_new.append(np.repeat(X_audio[idx], n_repeat, axis=0))
    X_context_new.append(np.repeat(X_context[idx], n_repeat, axis=0))
    y_new.append(np.repeat(y[idx], n_repeat, axis=0))

    # add remaining to reach max_count
    if n_extra > 0:
        extra_idx = np.random.choice(idx, size=n_extra, replace=False)
        X_audio_new.append(X_audio[extra_idx])
        X_context_new.append(X_context[extra_idx])
        y_new.append(y[extra_idx])


X_audio_bal = np.vstack(X_audio_new)
X_context_bal = np.concatenate(X_context_new)
y_bal = np.concatenate(y_new)

le = LabelEncoder()
y_int = le.fit_transform(y_bal)
y_cat = to_categorical(y_int)

X_audio_train, X_audio_test, X_context_train, X_context_test, y_train, y_test = train_test_split(
    X_audio_bal,
    X_context_bal,
    y_cat,        # one-hot labels
    test_size=0.2,
    stratify=y_int,   # MUY IMPORTANTE
    random_state=42
)

y_test_int = np.argmax(y_test, axis=1)

# Inputs
inp_audio = Input(shape=(1024,), name="audio_embedding")
inp_context = Input(shape=(1,), dtype="int32", name="context_id")

# Context embedding
ctx_emb = Embedding(input_dim=5, output_dim=4, name="context_embedding")(inp_context)
ctx_emb = Flatten()(ctx_emb)

# Combine audio + context
x = Concatenate()([inp_audio, ctx_emb])

# Dense layers
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(64, activation="relu")(x)

# Output layer
out = Dense(len(le.classes_), activation="softmax", name="output")(x)

# Build model
model = Model(inputs=[inp_audio, inp_context], outputs=out)

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
# model.summary()
history = model.fit(
    [X_audio_train, X_context_train],
    y_train,
    validation_split=0.2,
    batch_size=config.batch_size,
    callbacks=wandb_callback,
    epochs=config.epochs
)
exper.finish()