# TFtools 0.1.1


### TFtools Latest


    pip install --upgrade tensorflowtools


### TFtools Nightly


    pip install --upgrade -i https://test.pypi.org/simple/ tensorflowtools


This is a small package with various utilites to do with tensorflow.






**Disclaimer**: This package is not affiliated with or endorsed by TensorFlow, Google, or Huggingface. It is an independent open-source project built to integrate TensorFlow with various tools.


## License


Copyright 2025 Rihaan Meher


   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at


       http://www.apache.org/licenses/LICENSE-2.0


   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


## Dependencies


This project uses the following open-source libraries:


- **TensorFlow**: Licensed under the Apache 2.0 License.
- **Hugging Face Hub**: Licensed under the Apache 2.0 License.
- **setuptools**: Licensed under the MIT License (https://opensource.org/licenses/MIT).
- **pathlib**: Part of Python (Standard Library).
- **warnings** Part of Python (Standard Library)
- **os**: Part of Python (Standard Library).






# Submodules


## hftools


<details>
<summary>Functions</summary>


---


<details>
<summary>download_model_from_huggingface(username, repository, model_id)</summary>


This downloads a model named tf_model.h5 or tf_model.keras from huggingface to the tensorflowtools data directory. It can be used with the load_from_hf_cache function in the kerastools submodule


##### Example


    import tensorflowtools
    tensorflowtools.hftools.download_model_from_huggingface("sharktide", "recyclebot0", "sharktide/recyclebot0")
    model = tensorflowtools.kerastools.load_from_hf_cache("sharktide", "recyclebot0", "tf_model.h5")
    model.summary


</details>


<details>
<summary>clear_model_cache()</summary>


This clears the model cache; all downloaded models and configuration files will be deleted


##### Example


    import tensorflowtools
    tensorflowtools.hftools.download_model_from_huggingface("sharktide", "recyclebot0", "sharktide/recyclebot0")
    model = tensorflowtools.kerastools.load_from_hf_cache("sharktide", "recyclebot0", "tf_model.h5")
    model.summary
    tensorflowtools.hftools.clear_model_cache()
    try:
        model = tensorflowtools.kerastools.load_from_hf_cache("sharktide", "recyclebot0", "tf_model.h5")
    except:
        print("It worked!")


</details>


</details>


## kerastools


<details>
<summary>Functions</summary>


---


<details>
<summary>load_from_hf_cache(username, repository, filename)</summary>


This loads a model using tf.keras.models.load_model() from tensorflowtools's cache. Use with the hftools submodule to download a model to the cache.




##### Example


    import tensorflowtools
    tensorflowtools.hftools.download_model_from_huggingface("sharktide", "recyclebot0", "sharktide/recyclebot0")
    model = tensorflowtools.kerastools.load_from_hf_cache("sharktide", "recyclebot0", "tf_model.h5")
    model.summary
</details>


<details>
<summary>default_image_augmentation(rate)</summary>


This returns a sequential model of some basic image augmentation. The rate float is the amount of augmentation that should be applied. An average rate is 0.2.


##### Example


    import tensorflowtools


    model = tf.keras.Sequential([
        tensorflowtools.kerastools.default_image_augmentation(0.2),
        #rest of your layers here
    ])


</details>


<details>
<summary>basic_ffnn(input_dim, output_dim, loss, compile_model=True)</summary>


This returns a very basic fully connected neural network. The input dimensions are the dimensions for the first dense layer, the output dimensions are the dimensions for the last dense layer, the loss is the loss function to be used if compile_model is set to true. If you aren't planning to compile the model, still pick as loss function.


##### Example


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


</details>


<details>
<summary>basic_cnn(input_shape, num_classes, loss, compile_model=True)</summary>


This returns a basic convolutional neural network for image classification. The input dimensions are the dimensions for the first convolutional layer, the number of classes is used in the last dense layer. the actication of the last layer will automatically be switched between sigmoid and softmax depending on the type of classification.


##### Example


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


</details>


<details>
<summary>basic_autoencoder(input_shape, compile_model=True)</summary>


Gives basic starter architecture for a basic autencoder model.
:param input_shape: Input shape for the first conv2d layer
:param compile_model: Optionally compiles the model with mse loss.
:return: Generated lstm model with optional compilation.


##### Example
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


</details>




</details>




---




## Contributing


If you’d like to contribute to this package, feel free to fork the repository, make your changes, and submit a pull request. We welcome improvements, bug fixes, and additional models.


If you have a feature request, write a comment in the discussion for the latest version or rasie an issue.





