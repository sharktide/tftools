import os
import requests
from pathlib import Path
import tensorflow as tf
import shutil
import warnings

# Custom cache directory where models will be stored
CACHE_DIR = Path(os.path.expanduser("~/.cache/tensorflowtools"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HUGGINGFACE_DIR = CACHE_DIR / "huggingface"
HUGGINGFACE_DIR.mkdir(parents=True, exist_ok=True)

PACKAGE_DATA_DIR = HUGGINGFACE_DIR  # Directory to store all models

def get_model_folder(username: str, repository: str):
    """
    Returns the folder path where the model will be stored.
    """
    return PACKAGE_DATA_DIR / username / repository

def download_model_from_huggingface(username:str, repository:str, model_filename:str):
    """
    Downloads a model from huggingface
    :param username: the username of the model owner
    :param repository: the repository of the model
    :param model_filename: the model filename to download along with config.json and preprocessorconfig.json. Usually tf_model.h5 or tf_model.keras
    :return: downloaded model path. Not required if used with tensorflowtools.kerastools' load_from_hf_cache
    """
    model_folder = get_model_folder(username, repository)

    model_url = f"https://huggingface.co/{username}/{repository}/resolve/main/{model_filename}?download=true"
    model_cache_path = model_folder / model_filename

        # If the model already exists, no need to download it again
    if model_folder.exists():
        warnings.warn(f"Model already exists at {model_folder}. Deleting and re-downloading it.")
        shutil.rmtree(model_folder)
    
    model_folder.mkdir(parents=True, exist_ok=True)


    print(f"Downloading model from {model_url}...")
    response = requests.get(model_url, stream=True)
    if response.status_code == 200:
        with open(model_cache_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Model downloaded and saved to {model_cache_path}")
    else:
        raise Exception(f"Failed to download model from {model_url}. HTTP status code: {response.status_code}")
    
    
    config_url = f"https://huggingface.co/{username}/{repository}/resolve/main/config.json?download=true"
    print(f"Downloading config.json from {model_url}...")

    response = requests.get(config_url, stream=True)
    config_cache_path = model_folder / "config.json"
    if response.status_code == 200:
        with open(config_cache_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Configuration file downloaded and saved to {model_cache_path}")
    else:
        warnings.warn(f"Failed to download config.json from {model_url}. HTTP status code: {response.status_code}")


    pconfig_url = f"https://huggingface.co/{username}/{repository}/resolve/main/preprocessorconfig.json?download=true"
    print(f"Downloading preprocessorconfig.json from {model_url}...")

    response = requests.get(pconfig_url, stream=True)
    config_cache_path = model_folder / "preprocessorconfig.json"
    if response.status_code == 200:
        with open(config_cache_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Preprocessor Configuration file downloaded and saved to {model_cache_path}")
    else:
        warnings.warn(f"Failed to download preprocessorconfig.json from {model_url}. HTTP status code: {response.status_code}")

def clear_model_cache():
    """
    Clears anything in the downloaded model cache.
    """
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

