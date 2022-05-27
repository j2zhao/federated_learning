import tensorflow as tf
import keras

def gen_net():
    model = keras.Sequential([
    keras.layers.Dense(1, input_dim=104, activation='sigmoid')
])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return model