import tensorflow as tf
from pathlib import Path
import pkg_resources

# Define the path to the models directory in the package's data folder
PACKAGE_DATA_DIR = Path(pkg_resources.resource_filename('tftools', 'data')) / "models"

def load_model_from_cache(model_name: str):
    """
    Loads a TensorFlow model from the package's data directory.

    :param model_name: Name of the model to load (e.g., 'custom_model.h5')
    :return: Loaded TensorFlow model
    """
    model_path = PACKAGE_DATA_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"The model {model_name} is not found in the package data directory.")
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    return model
