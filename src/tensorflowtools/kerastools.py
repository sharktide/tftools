import tensorflow as tf
from pathlib import Path
import pkg_resources
import warnings
import os

CACHE_DIR = Path(os.path.expanduser("~/.cache/tensorflowtools"))

HUGGINGFACE_DIR = CACHE_DIR / "huggingface"
HUGGINGFACE_DIR.mkdir(parents=True, exist_ok=True)

PACKAGE_DATA_DIR = HUGGINGFACE_DIR

def load_from_hf_cache(username: str, repository: str,  model_name: str):
    """
    Loads a TensorFlow model from the package's data directory.

    :param model_name: Name of the model to load (e.g., 'custom_model.h5')
    :return: Loaded TensorFlow model
    """
    model_path = PACKAGE_DATA_DIR / username / repository / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"The model {model_name} is not found in the huggingface package data directory.")
    
    model = tf.keras.models.load_model(model_path)
    return model

def default_image_augmentation(rate: float):
    """
    Gives basic but useful image augmentation custom layers with only 1 parameter.
    :param rate: The intensity of the augmentations (float, recommended 0.2)
    :return: Generated tensorflow keras custom layers (type: sequential model)
    """

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(rate),
        tf.keras.layers.RandomTranslation(height_factor=rate ,width_factor=rate),
        tf.keras.layers.RandomZoom(rate),
    ])

    return augmentation



def basic_ffnn(input_dim, output_dim, loss, activation, compile_model=True):
    """
    Gives basic starter architecture for a feedforward neural network
    :param input_dim: The input dimensions
    :param output_dim: The output dimensions
    :param loss: Loss function
    :param activation: the activation for the final dense
    :param compile_model: Optionally compiles the model.
    :return: Generated ffnn with optional compilation.
    """

    from tf.keras.regularizers import l2

    loss = (str(loss))

    ffnn_basic = tf.keras.Sequential([
        tf.keras.layers.Dense(256, input_dim=input_dim, activation='LeakyReLU', kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='LeakyReLU', kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='LeakyReLU', kernel_regularizer=l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(output_dim, activation=activation)  # for multi-class classification
    ])
    
    if compile_model:
        ffnn_basic.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    return ffnn_basic

def basic_cnn(input_shape, num_classes, loss, compile_model=True):
    """
    Gives basic starter architecture for a convolutinal neural network
    :param input_shape: Input shape for the first conv2d layer
    :param num_classes: If binary, leave 2, if multi-class, increase to number of desired classes. Used for final dense.
    :param loss: Loss function
    :param compile_model: Optionally compiles the model.
    :return: Generated cnn with optional compilation.
    """

    cnn_basic0 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='LeakyReLU', padding='same', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='LeakyReLU', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, (3, 3), activation='LeakyReLU', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='LeakyReLU'),
        tf.keras.layers.Dropout(0.5),
        
    ])
    
    if num_classes > 2:
        cnn_basic = tf.keras.Sequential([
            cnn_basic0,
            tf.keras.layers.Dense(num_classes, activation='softmax')  # for multi-class classification
        ])
    elif num_classes == 2:
        cnn_basic = tf.keras.Sequential([
            cnn_basic0,
            tf.keras.layers.Dense(num_classes, activation='sigmoid')  # for multi-class classification
        ])
    else:
        warnings.warn("Num classes input was not received as integer, or less than 2. Defaulting to sigmoid binary classification with inputted num_classes")
        cnn_basic = tf.keras.Sequential([
            cnn_basic0,
            tf.keras.layers.Dense(num_classes, activation='sigmoid')  
        ])
    if compile_model:
        cnn_basic.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    return cnn_basic


def basic_lstm(input_shape, output_dim, loss, activation, compile_model=True):
    """
    Gives basic starter architecture for a lstm model.
    :param input_shape: Input shape for the first lstm layer
    :param output_dim: Output dimensions of final dense.
    :param loss: Loss function
    :param activation: The activation function for the last dense layer.
    :param compile_model: Optionally compiles the model.
    :return: Generated lstm model with optional compilation.
    """
    loss = (str(loss))
    lstm_basic = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True, dropout=0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='LeakyReLU'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(output_dim, activation=activation)  # for multi-class classification
    ])
    
    if compile_model:
        lstm_basic.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    return lstm_basic


def basic_autoencoder(input_shape, compile_model=True):
    """
    Gives basic starter architecture for a basic autencoder model.
    :param input_shape: Input shape for the first conv2d layer
    :param compile_model: Optionally compiles the model.
    :return: Generated lstm model with optional compilation.
    """

    input_layer = tf.keras.layers.Input(shape=input_shape)
    # Encoder
    encoded = tf.keras.layers.Conv2D(32, (3, 3), activation='LeakyReLU', padding='same')(input_layer)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    encoded = tf.keras.layers.BatchNormalization()(encoded)
    encoded = tf.keras.layers.Dropout(0.2)(encoded)
    
    encoded = tf.keras.layers.Conv2D(64, (3, 3), activation='LeakyReLU', padding='same')(encoded)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    encoded = tf.keras.layers.BatchNormalization()(encoded)
    encoded = tf.keras.layers.Dropout(0.3)(encoded)
    
    # Decoder
    decoded = tf.keras.layers.Conv2D(64, (3, 3), activation='LeakyReLU', padding='same')(encoded)
    decoded = tf.keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = tf.keras.layers.BatchNormalization()(decoded)
    decoded = tf.keras.layers.Dropout(0.4)(decoded)
    
    decoded = tf.keras.layers.Conv2D(32, (3, 3), activation='LeakyReLU', padding='same')(decoded)
    decoded = tf.keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = tf.keras.layers.BatchNormalization()(decoded)
    
    decoded = tf.keras.layers.Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(decoded)

    autoencoder_basic = tf.keras.Model(input_layer, decoded)
    
    if compile_model:
        autoencoder_basic.compile(optimizer='adam', loss='mse')
    
    return autoencoder_basic




#Copyright 2025 Rihaan Meher
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.