import unittest
import tensorflow as tf
from tensorflowtools.kerastools import default_image_augmentation, basic_ffnn, basic_cnn, basic_lstm, basic_autoencoder

class TestTensorFlowTools(unittest.TestCase):

    def test_default_image_augmentation(self):
        # Test if default image augmentation creates a Sequential model with expected layers
        augmentation = default_image_augmentation(0.2)
        self.assertIsInstance(augmentation, tf.keras.Sequential)
        self.assertEqual(len(augmentation.layers), 4)  # We expect 4 augmentation layers

        # Check that the augmentation layers are of the correct types
        self.assertIsInstance(augmentation.layers[0], tf.keras.layers.RandomFlip)
        self.assertIsInstance(augmentation.layers[1], tf.keras.layers.RandomRotation)
        self.assertIsInstance(augmentation.layers[2], tf.keras.layers.RandomTranslation)
        self.assertIsInstance(augmentation.layers[3], tf.keras.layers.RandomZoom)

    def test_basic_ffnn(self):
        # Test basic fully connected neural network (FFNN)
        model = basic_ffnn(input_dim=28*28, output_dim=10, loss='categorical_crossentropy', compile_model=False)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(len(model.layers), 7)  # Check the number of layers

        # Check the first layer is a Dense layer with the correct input dimension
        self.assertIsInstance(model.layers[0], tf.keras.layers.Dense)
        self.assertEqual(model.layers[0].input_dim, 28*28)

    def test_basic_cnn(self):
        # Test basic convolutional neural network (CNN)
        model = basic_cnn(input_shape=(28, 28, 1), num_classes=10, loss='sparse_categorical_crossentropy', compile_model=False)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(len(model.layers), 9)  # Check the number of layers

        # Check the first layer is a Conv2D layer with the correct input shape
        self.assertIsInstance(model.layers[0], tf.keras.layers.Conv2D)
        self.assertEqual(model.layers[0].input_shape[1:], (28, 28, 1))

    def test_basic_lstm(self):
        # Test basic LSTM model
        model = basic_lstm(input_shape=(500,), output_dim=1, loss='categorical_crossentropy', compile_model=False)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(len(model.layers), 6)  # Check the number of layers

        # Check the first layer is an LSTM layer with the correct input shape
        self.assertIsInstance(model.layers[0], tf.keras.layers.LSTM)
        self.assertEqual(model.layers[0].input_shape[1:], (500,))

    def test_basic_autoencoder(self):
        # Test basic Autoencoder model
        model = basic_autoencoder(input_shape=(28, 28, 1), compile_model=False)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(len(model.layers), 11)  # Check the number of layers

        # Check the first layer is an Input layer
        self.assertIsInstance(model.layers[0], tf.keras.layers.InputLayer)

    # Add more tests as needed for other functions

if __name__ == '__main__':
    unittest.main()
