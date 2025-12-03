from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

def make_sample_weights(context_labels):
    return {
        'primary_state': np.ones(len(context_labels)),
        'braking_substate': (context_labels == 0).astype(float),
        'idle_substate': (context_labels == 1).astype(float),
        'moving_substate': (context_labels == 2).astype(float),
        'startup_substate': (context_labels == 3).astype(float),
    }

data = []
for filename in os.listdir('mel_spectogram'):
    data.append(np.load('mel_spectogram/'+filename))
labels = pd.read_csv('metadata.csv')[['label', 'context']]

(x_train, x_test, y_train, y_test) = train_test_split(np.array(data), labels, test_size=0.2, random_state=42, stratify=labels['context'])

braking_map = {'normal_brakes':0, 'worn_out_brakes':1}
idle_map = {'combined':0, 'low_oil':1, 'normal_engine_idle':2, 'power_steering':3, 'serpentine_belt':4}
moving_map = {'car_clean':0, 'car_knocking':1}
startup_map = {'bad_ignition':0, 'dead_battery':1, 'normal_engine_startup':2}

y_train_dict = {
    'primary_state': y_train['context'].values,  # 0,1,2,3
    'braking_substate': y_train['label'].map(braking_map).fillna(0).astype(int).values,
    'idle_substate': y_train['label'].map(idle_map).fillna(0).astype(int).values,
    'moving_substate': y_train['label'].map(moving_map).fillna(0).astype(int).values,
    'startup_substate': y_train['label'].map(startup_map).fillna(0).astype(int).values
}

sample_weights = make_sample_weights(y_train['context'].values)

input = Input(shape=(876,64,1))
x = Conv2D(512, kernel_size=(4,4), padding = 'same', activation='relu')(input)
x = BatchNormalization()(x)
x = MaxPool2D((2,2))(x)

x = Conv2D(256, kernel_size=(4,4), padding = 'same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D((2,2))(x)

x = Conv2D(256, kernel_size=(4,4), padding = 'same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D((2,2))(x)

x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)

# Identifiable head
primary_state = Dense(128, activation='relu', name='primary_dense1')(x)
primary_state = Dropout(0.2, name='primary_dropout')(primary_state)
primary_state = Dense(64, activation='relu', name='primary_dense2')(primary_state)
primary_output = Dense(4, activation='softmax', name='primary_state')(primary_state)

# Braking head
braking_features = Dense(64, activation='relu', name='braking_features')(x)
braking_output = Dense(2, activation='softmax', name='braking_substate')(braking_features)

# Idle head
idle_features = Dense(64, activation='relu', name='idle_features')(x)
idle_output = Dense(4, activation='softmax', name='idle_substate')(idle_features)

# Moving head
moving_features = Dense(64, activation='relu', name='moving_features')(x)
moving_output = Dense(2, activation='softmax', name='moving_substate')(moving_features)

# Startup head
startup_features = Dense(64, activation='relu', name='startup_features')(x)
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
    metrics = ['accuracy']
)
model.summary()

resumen = model.fit(x_train, y_train_dict, epochs=10, batch_size=32, validation_split=0.1, sample_weight=sample_weights)
