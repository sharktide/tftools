import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import pkg_resources

# Define the directory to store models in the package’s data directory
PACKAGE_DATA_DIR = Path(pkg_resources.resource_filename('tftools', 'data')) / "models"
PACKAGE_DATA_DIR.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

def download_model_from_huggingface(model_id: str, custom_filename: str = None):
    """
    Downloads a TensorFlow model from a Hugging Face repository and stores it in the package's data directory.
    Optionally, allows the user to specify a custom filename.
    
    If no custom filename is provided, the filename from the Hugging Face repo is used as the default.

    :param model_id: The Hugging Face model identifier (repo name)
    :param custom_filename: The name to save the model as (e.g., 'my_model.h5'). If None, defaults to the model file name in the repo.
    :return: Path to the downloaded model file
    """
    

    if os.path.exists((PACKAGE_DATA_DIR / "tf_model.h5")):
        print(f"File {(PACKAGE_DATA_DIR / "tf_model.h5")} already exists. Deleting the existing file.")
        os.remove((PACKAGE_DATA_DIR / "tf_model.h5"))  # Delete the existing file
    if os.path.exists((PACKAGE_DATA_DIR / "tf_model.keras")):
        print(f"File {(PACKAGE_DATA_DIR / "tf_model.keras")} already exists. Deleting the existing file.")
        os.remove((PACKAGE_DATA_DIR / "tf_model.keras"))  # Delete the existing file
    # Try downloading the model as tf_model.h5 or tf_model.keras
    try:
        tf_model_path = hf_hub_download(repo_id=model_id, filename="tf_model.h5")
        filename = "tf_model.h5"  # Use the default name if tf_model.h5 is found
    except:
        try:
            tf_model_path = hf_hub_download(repo_id=model_id, filename="tf_model.keras")
            filename = "tf_model.keras"  # Use the default name if tf_model.keras is found
        except:
            raise FileNotFoundError("No TensorFlow model found in the repo with names 'tf_model.h5' or 'tf_model.keras'.")
    
    # If the user specifies a custom filename, override the default name
    if custom_filename:
        if os.path.exists((PACKAGE_DATA_DIR / custom_filename)):
            print(f"File {(PACKAGE_DATA_DIR / custom_filename)} already exists. Deleting the existing file.")
            os.remove((PACKAGE_DATA_DIR / custom_filename))  # Delete the existing file
        filename = custom_filename

    # Move the downloaded model to the package's data/models folder with the desired filename
    model_cache_path = PACKAGE_DATA_DIR / filename
    os.rename(tf_model_path, model_cache_path)

    
    return model_cache_path


