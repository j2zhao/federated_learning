import tensorflow as tf
import keras

def gen_net():
    base_model = tf.keras.applications.ResNet50V2(weights='imagenet',
                                                input_shape=(200, 200, 3),
                                                include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("covid_classifier_model.h5", save_best_only=True, verbose=0),
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return model