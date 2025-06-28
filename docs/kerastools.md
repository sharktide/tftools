## kerastools

<details>
<summary>load_from_hf_cache(username, repository, filename)</summary>

This loads a model using tf.keras.models.load_model() from tensorflowtools's cache. Use with the hftools submodule to download a model to the cache.

##### Example

```py
import tensorflowtools
tensorflowtools.hftools.download_model_from_huggingface("sharktide", "recyclebot0", "sharktide/recyclebot0")
model = tensorflowtools.kerastools.load_from_hf_cache("sharktide", "recyclebot0", "tf_model.h5")
model.summary
```

</details>

<details>
<summary>default_image_augmentation(rate)</summary>

This returns a sequential model of some basic image augmentation. The rate float is the amount of augmentation that should be applied. An average rate is 0.2.

##### Example

```py
import tensorflowtools

model = tf.keras.Sequential([
    tensorflowtools.kerastools.default_image_augmentation(0.2),
    #rest of your layers here
])
```

</details>

<details>
<summary>basic_ffnn(input_dim, output_dim, loss, compile_model=True)</summary>

This returns a very basic fully connected neural network. The input dimensions are the dimensions for the first dense layer, the output dimensions are the dimensions for the last dense layer, the loss is the loss function to be used if compile_model is set to true. If you aren't planning to compile the model, still pick as loss function.

##### Example

    ```py
# Example usage of the basic FFNN
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create and compile the model
model = basic_ffnn(input_dim=28*28, output_dim=10, 'categorical_crossentropy')

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

</details>

<details>
<summary>basic_cnn(input_shape, num_classes, loss, compile_model=True)</summary>

This returns a basic convolutional neural network for image classification. The input dimensions are the dimensions for the first convolutional layer, the number of classes is used in the last dense layer. the actication of the last layer will automatically be switched between sigmoid and softmax depending on the type of classification.

##### Example
```py
# Example usage of the basic CNN
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create and compile the model
model = basic_cnn(input_shape=(28, 28, 1), num_classes=10, 'sparse_categorical_crossentropy')

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

</details>

<details>
<summary>basic_lstm(input_shape, output_dim, loss, activation, compile_model=True)</summary>

Gives basic starter architecture for a lstm model.
:param input_shape: Input shape for the first lstm layer
:param output_dim: Output dimensions of final dense.
:param loss: Loss function
:param activation: The activation function for the last dense layer.
:param compile_model: Optionally compiles the model.
:return: Generated lstm model with optional compilation.

##### Example
```py
# Example usage of the basic LSTM model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = imdb.load_data()
x_train = pad_sequences(x_train, maxlen=500)
x_test = pad_sequences(x_test, maxlen=500)

# Create and compile the model
model = basic_lstm(input_shape=(500, ), output_dim=1, 'categorical_crossentropy', 'softmax')

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64)
```

</details>

<details>
<summary>basic_autoencoder(input_shape, compile_model=True)</summary>

Gives basic starter architecture for a basic autencoder model.
:param input_shape: Input shape for the first conv2d layer
:param compile_model: Optionally compiles the model with mse loss.
:return: Generated lstm model with optional compilation.

##### Example
```py
# Example usage of the basic Autoencoder
from tensorflow.keras.datasets import mnist

# Load and preprocess data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Create and compile the model
model = basic_autoencoder(input_shape=(28, 28, 1))

# Train the model
model.fit(x_train, x_train, epochs=10, batch_size=128)
```
</details>
