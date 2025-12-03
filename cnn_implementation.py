from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPool2D, BatchNormalization, Flatten, Dropout

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

x = Flatten()(x)
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
idle_output = Dense(3, activation='softmax', name='idle_substate')(idle_features)

# Moving head
moving_features = Dense(64, activation='relu', name='moving_features')(x)
moving_output = Dense(3, activation='softmax', name='moving_substate')(moving_features)

# Startup head
startup_features = Dense(64, activation='relu', name='startup_features')(x)
startup_output = Dense(3, activation='softmax', name='startup_substate')(startup_features)

model = Model(inputs=input, outputs=[primary_output, braking_output, idle_output, moving_output, startup_output])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()