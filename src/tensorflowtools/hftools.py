import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import pkg_resources
import warnings
import shutil

# Define the cache directory for your library
CACHE_DIR = Path(os.path.expanduser("~/.cache/tensorflowtools"))

# Create the huggingface subdirectory within the cache directory
HUGGINGFACE_DIR = CACHE_DIR / "huggingface"
HUGGINGFACE_DIR.mkdir(parents=True, exist_ok=True)

PACKAGE_DATA_DIR = HUGGINGFACE_DIR

def download_model_from_huggingface(username: str, repository: str, model_id: str):
    """
    Downloads a TensorFlow model from a Hugging Face repository and stores it in the package's data directory.
    

    :param model_id: The Hugging Face model identifier (repo name) eg. sharktide/recyclebot0
    :param username: The name of the username of the publisher. eg. sharktide
    :param repository: The name of the repository of the model. eg. recyclebot0
    :return: Path to the downloaded model file
    """
    

    if os.path.exists((PACKAGE_DATA_DIR / username / repository / "tf_model.h5")):
        print(f"File {(PACKAGE_DATA_DIR / username / repository / 'tf_model.h5')} already exists. Deleting the existing file.")
        os.remove((PACKAGE_DATA_DIR / username / repository / "tf_model.h5"))  # Delete the existing file
    if os.path.exists((PACKAGE_DATA_DIR / username / repository / "tf_model.keras")):
        print(f"File {(PACKAGE_DATA_DIR / username / repository / 'tf_model.keras')} already exists. Deleting the existing file.")
        os.remove((PACKAGE_DATA_DIR / username / repository / "tf_model.keras"))  # Delete the existing file
    if os.path.exists((PACKAGE_DATA_DIR / username / repository / "config.json")):
        print(f"File {(PACKAGE_DATA_DIR / username / repository / 'config.json')} already exists. Deleting the existing file.")
        os.remove((PACKAGE_DATA_DIR / username / repository / "config.json"))  # Delete the existing file
    if os.path.exists((PACKAGE_DATA_DIR / username / repository / "model.weights.h5")):
        print(f"File {(PACKAGE_DATA_DIR / username / repository / 'model.weights.h5')} already exists. Deleting the existing file.")
        os.remove((PACKAGE_DATA_DIR / username / repository / "model.weights.h5"))  # Delete the existing file
    try:
        tf_model_path = hf_hub_download(repo_id=model_id, filename="tf_model.h5")

        filename = "tf_model.h5"  # Use the default name if tf_model.h5 is found
    except:
        try:
            tf_model_path = hf_hub_download(repo_id=model_id, filename="tf_model.keras")
            filename = "tf_model.keras"  # Use the default name if tf_model.keras is found
        except:
            raise FileNotFoundError("No TensorFlow model found in the repo with names 'tf_model.h5' or 'tf_model.keras'.")
    

    configpath = "0"
    pconfigpath = "0"


    try:
        configpath = hf_hub_download(repo_id=model_id, filename="config.json")

        configfilename = "config.json" 
    except:
        warnings.warn("No model configuration file found at the requested directory. Warning: file 'config.json' was not found at the requested directory. Skipping the download")
        configpath = "0"
    
    try:
        pconfigpath = hf_hub_download(repo_id=model_id, filename="preprocessorconfig.json")

        pconfigfilename = "preprocessorconfig.json"  
    except:
        warnings.warn("No preprocessor configuration file found at the requested directory. Warning: file 'preprocessorconfig.json' was not found at the requested directory. Skipping the download")
        pconfigpath = "0"

    if os.path.exists((PACKAGE_DATA_DIR / username / repository)):
        pass
    else:
        (PACKAGE_DATA_DIR / username / repository).mkdir(parents=True, exist_ok=True)
    
    model_cache_path = PACKAGE_DATA_DIR / username / repository / filename
    os.rename(tf_model_path, model_cache_path)
    if configfilename != "0":
        config_cache_path = PACKAGE_DATA_DIR / username / repository / configfilename
        os.rename(configpath, config_cache_path)

    if pconfigpath != "0":
        pconfig_cache_path = PACKAGE_DATA_DIR / username / repository / pconfigfilename
        os.rename(pconfigpath, pconfig_cache_path)
    
    return model_cache_path

def clear_model_cache():
    shutil.rmtree(PACKAGE_DATA_DIR)
    PACKAGE_DATA_DIR.mkdir(parents=True, exist_ok=True)  



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

