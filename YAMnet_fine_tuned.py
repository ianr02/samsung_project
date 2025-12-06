from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import wandb

#wandb.login()
cnn_summary = """
Hierarchical Multi-Output CNN Architecture (Mel-Spectrogram Input)

Input:
- Shape: (64, 874, 1) — Mel-spectrogram of engine audio

Shared Convolutional Base:
1. Conv2D(512, kernel=(4,4), padding='same', activation='relu') -> BatchNormalization -> MaxPool2D((2,2))
2. Conv2D(256, kernel=(4,4), padding='same', activation='relu') -> BatchNormalization -> MaxPool2D((2,2))
3. Conv2D(256, kernel=(4,4), padding='same', activation='relu') -> BatchNormalization -> MaxPool2D((2,2))
4. GlobalAveragePooling2D()
5. Dense(128, activation='relu') -> Dropout(0.4)

Hierarchical Multi-Task Heads:
- Primary state: Dense(128) -> Dropout(0.2) -> Dense(64) -> Dense(4, softmax)
- Braking: Dense(64) -> Dense(2, softmax)
- Idle: Dense(64) -> Dense(4, softmax)
- Moving: Dense(64) -> Dense(2, softmax)
- Startup: Dense(64) -> Dense(3, softmax)
"""
exper = wandb.init(project= "hierarchical_cnn", #identificador del proyecto
                   name = "first_try", #nombre cuando creo el proyecto
                   config= {#AQUI SE ESCRIBEN LOS PARAMETROS PERO NO SON LOS QUE VA A URILIZAR EL MODELO
                            #SON LPS QUE NOSOTROS CARGAMOS COMO NOTAS
                            "epochs": 50,
                            "batch_size":32,
                            "loss_function":"sparse_categorical_crossentropy",
                            "arquitecture":cnn_summary,
                            "Input":"876x64"
                            })
config = wandb.config

wandb_callback = [WandbMetricsLogger(), #WandbMetricsLogger todas las metricas las manda a la pagina web obligatoria
                  WandbModelCheckpoint(filepath = "model.keras", #Fichero que almacena se pone directorio
                  monitor="val_loss",
                  save_best_only=True) #Me guarda el mejor y solo uno cada vez que se actualice
                  ]

data = []
for filename in os.listdir('mel_spectogram'):
    spec = np.load('mel_spectogram/'+filename)
    spec = np.expand_dims(spec, axis=-1)
    data.append(spec)
labels = pd.read_csv('metadata.csv')[['label', 'context']]

(x_train, x_test, y_train, y_test) = train_test_split(np.array(data), labels, test_size=0.2, random_state=42, stratify=labels['context'])

braking_map = {'normal_brakes':0, 'worn_out_brakes':1}
idle_map = {'low_oil':0, 'normal_engine_idle':1, 'power_steering':2, 'serpentine_belt':3}
moving_map = {'car_clean':0, 'car_knocking':1}
startup_map = {'bad_ignition':0, 'dead_battery':1, 'normal_engine_startup':2}

y_train_dict = {
    'primary_state': y_train['context'].values,
    'braking_substate': y_train['label'].map(braking_map).fillna(0).astype(np.int32).values,
    'idle_substate': y_train['label'].map(idle_map).fillna(0).astype(np.int32).values,
    'moving_substate': y_train['label'].map(moving_map).fillna(0).astype(np.int32).values,
    'startup_substate': y_train['label'].map(startup_map).fillna(0).astype(np.int32).values,
}

weights = {
    'primary_state': np.ones(len(y_train)),
    'braking_substate': (y_train_dict['braking_substate'] != 0).astype(np.float64),
    'idle_substate': (y_train_dict['idle_substate'] != 0).astype(np.float64),
    'moving_substate': (y_train_dict['moving_substate'] != 0).astype(np.float64),
    'startup_substate': (y_train_dict['startup_substate'] != 0).astype(np.float64),
}

y_train_list = [
    y_train_dict['primary_state'],
    y_train_dict['braking_substate'],
    y_train_dict['idle_substate'],
    y_train_dict['moving_substate'],
    y_train_dict['startup_substate'],
]

weights_list = [
    weights['primary_state'],
    weights['braking_substate'],
    weights['idle_substate'],
    weights['moving_substate'],
    weights['startup_substate'],
]

input = Input(shape=(64,876,1))
x = Conv2D(64, kernel_size=(4,4), padding = 'same', activation='relu')(input)
x = BatchNormalization()(x)
x = MaxPool2D((2,2))(x)

x = Conv2D(128, kernel_size=(4,4), padding = 'same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D((2,2))(x)

x = Conv2D(256, kernel_size=(4,4), padding = 'same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D((2,2))(x)

x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)

# Identifiable head
primary_state = Dropout(0.2, name='primary_dropout')(x)
primary_state = Dense(64, activation='relu', name='primary_dense2')(primary_state)
primary_output = Dense(4, activation='softmax', name='primary_state')(primary_state)

# Braking head
braking_features = Dropout(0.4, name='braking_dropout')(x)
braking_features = Dense(64, activation='relu', name='braking_features')(braking_features)
braking_output = Dense(2, activation='softmax', name='braking_substate')(braking_features)

# Idle head
idle_features = Dropout(0.3, name='idle_dropout')(x)
idle_features = Dense(64, activation='relu', name='idle_features')(idle_features)
idle_output = Dense(4, activation='softmax', name='idle_substate')(idle_features)

# Moving head
moving_features = Dropout(0.2, name='moving_dropout')(x)
moving_features = Dense(64, activation='relu', name='moving_features')(moving_features)
moving_output = Dense(2, activation='softmax', name='moving_substate')(moving_features)

# Startup head
startup_features = Dropout(0.2, name='startup_dropout')(x)
startup_features = Dense(64, activation='relu', name='startup_features')(startup_features)
startup_output = Dense(3, activation='softmax', name='startup_substate')(startup_features)

model = Model(inputs=input, outputs=[primary_output, braking_output, idle_output, moving_output, startup_output])
model.compile(
    optimizer='adam',
    loss={
        'primary_state': 'sparse_categorical_crossentropy',
        'braking_substate': 'sparse_categorical_crossentropy',
        'idle_substate': 'sparse_categorical_crossentropy',
        'moving_substate': 'sparse_categorical_crossentropy',
        'startup_substate': 'sparse_categorical_crossentropy',
    },
    metrics={
        'primary_state': ['accuracy'],
        'braking_substate': ['accuracy'],
        'idle_substate': ['accuracy'],
        'moving_substate': ['accuracy'],
        'startup_substate': ['accuracy'],
    }
)

resumen = model.fit(
    x_train,
    y_train_list,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    sample_weight=weights_list
)
exper.finish()