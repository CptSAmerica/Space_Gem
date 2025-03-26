import numpy as np
import tensorflow as tf

from colorama import Fore, Style
from typing import Tuple

from keras import Model, Sequential, layers, callbacks, optimizers

def initialize_baseline_model(input_shape=(225, 225, 3)) -> Model:
    """
    Initialize the baseline model
    """
    base_model = Sequential()

    base_model.add(layers.Input(input_shape))

    base_model.add(layers.Flatten())

    base_model.add(layers.Dense(87, activation="softmax"))

    print("✅ Baseline model initialized")

    return base_model

def initialize_model(input_shape=(225, 225, 3)) -> Model:
    """
    Initialize the baseline model
    """
    model = Sequential()

    model.add(layers.Input(input_shape))
    model.add(layers.Rescaling(1./225))

    model.add(layers.Conv2D(filters = 32, kernel_size = (3,3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )


    model.add(layers.Conv2D(filters = 32, kernel_size = (3,3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )


    model.add(layers.Conv2D(filters = 64, kernel_size = (3,3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )

    # Here we flatten our data to end up with just one dimension

    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation="relu"))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(87, activation="softmax"))

    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    adam = optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss='categorical_crossentropy',
                optimizer=adam,
                metrics=['accuracy'])

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        batch_size=64,
        patience=10
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    MODEL = "../models/model_v2_1.keras"

    modelCheckpoint = callbacks.ModelCheckpoint(MODEL,
                                                monitor="val_loss",
                                                verbose=0,
                                                save_best_only=True)

    LRreducer = callbacks.ReduceLROnPlateau(monitor="val_loss",
                                            factor=0.1,
                                            patience=patience,
                                            verbose=1,
                                            min_lr=0)

    EarlyStopper = callbacks.EarlyStopping(monitor='val_loss',
                                    patience=patience,
                                    verbose=0,
                                    restore_best_weights=True)

    history = model.fit(
        train_ds,
        epochs=50,
        validation_data=val_ds,
        batch_size=batch_size,
        callbacks=[modelCheckpoint,LRreducer, EarlyStopper]
        )

    print(f"✅ Model trained on {len(train_ds)} rows with min val accuracy: {round(np.min(history.history['val_accuracy']), 2)}")

    return model, history
