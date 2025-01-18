import tensorflow as tf
from pathlib import Path
import pkg_resources

# Define the path to the models directory in the package's data folder
PACKAGE_DATA_DIR = Path(pkg_resources.resource_filename('tensorflowtools', 'data')) / "models" / "huggingface"

def load_from_hf_cache(username: str, repository: str,  model_name: str):
    """
    Loads a TensorFlow model from the package's data directory.

    :param model_name: Name of the model to load (e.g., 'custom_model.h5')
    :return: Loaded TensorFlow model
    """
    model_path = PACKAGE_DATA_DIR / username / repository / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"The model {model_name} is not found in the huggingface package data directory.")
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    return model

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