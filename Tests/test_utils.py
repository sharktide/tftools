import unittest
import os
from pathlib import Path
import tensorflow as tf
from tftools.hftools import download_model_from_huggingface
from tftools.kerastools import load_model_from_cache

class TestHFUtils(unittest.TestCase):
    
    def setUp(self):
        """Set up a clean environment before each test."""
        self.model_id = "sharktide/recyclebot0"  # Use an actual model ID from Hugging Face
        self.cache_dir = Path(__file__).parent / "data/models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure the cache directory exists
        
    def tearDown(self):
        """Clean up after each test."""
        for file in self.cache_dir.glob("*"):
            file.unlink()  # Delete the downloaded model files
        self.cache_dir.rmdir()  # Optionally remove the cache directory

    def test_download_model_from_huggingface_default_filename(self):
        """Test downloading model with the default filename (from HF repo)."""
        model_path = download_model_from_huggingface("sharktide", "recyclebot0", "sharktide/recyclebot0")
        
        # Ensure the model file exists
        self.assertTrue(model_path.exists())
        self.assertIn("tf_model.h5", model_path.name)  # The default filename should be 'tf_model.h5' or 'tf_model.keras'
        
        # Ensure the model can be loaded correctly
        model = load_model_from_cache("sharktide", "recyclebot0", "tf_model.h5")
        self.assertIsInstance(model, tf.keras.Model)

    def test_download_model_from_huggingface_custom_filename(self):
        """Test downloading model with a custom filename."""
        custom_filename = "recyclebot.h5"
        model_path = download_model_from_huggingface("sharktide", "recyclebot0", self.model_id)
        
        # Ensure the model file exists with the custom filename
        self.assertTrue(model_path.exists())
        
        # Ensure the model can be loaded correctly using the custom filename
        model = load_model_from_cache("sharktide", "recyclebot0", "tf_model.h5")
        self.assertIsInstance(model, tf.keras.Model)

    def test_download_non_existing_model(self):
        """Test that downloading a non-existing model raises an error."""
        with self.assertRaises(FileNotFoundError):
            download_model_from_huggingface("rnfsiufc", "aeirhfbearyfhbcear", "non-existing-model")

    def test_load_model_from_cache(self):
        """Test loading a model from cache."""
        # First, download the model and store it in cache
        model_path = download_model_from_huggingface("sharktide", "recyclebot0", self.model_id)
        
        # Load the model from the cache
        model = load_model_from_cache("sharktide", "recyclebot0", model_path.name)
        self.assertIsInstance(model, tf.keras.Model)

    def test_cache_directory_exists(self):
        """Test that the cache directory is created when the package is used."""
        # Download a model and check if the cache directory exists
        download_model_from_huggingface("sharktide", "recyclebot0", self.model_id)
        self.assertTrue(self.cache_dir.exists())

    def test_invalid_model_path(self):
        """Test loading an invalid model path."""
        with self.assertRaises(FileNotFoundError):
            load_model_from_cache("sharktide", "recyclebot0", "invalid_model_name.h5")


if __name__ == "__main__":
    unittest.main()
