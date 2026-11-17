"""Autoencoder-based deep feature extraction."""
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))

    encoded = Dense(64, activation="relu")(input_layer)
    encoded = Dense(32, activation="relu")(encoded)

    decoded = Dense(64, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="sigmoid")(decoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def extract_deep_features(X):
    """Train the autoencoder on X and return its 32-dimensional latent features."""
    autoencoder, encoder = build_autoencoder(X.shape[1])

    autoencoder.fit(
        X, X,
        epochs=20,
        batch_size=32,
        verbose=1
    )

    deep_features = encoder.predict(X, verbose=0)
    return np.asarray(deep_features)
