import os
import requests
import shutil
import warnings
from pathlib import Path
from huggingface_hub import HfApi
import importlib.util
import sys
from textformat import TableFormatter, bcolors, TextFormat
from textformat.progress import Colors

from .internal import (
    _download_with_progress,
    _format_size,
    _load_custom_objects_from_file
)

CACHE_DIR = Path(os.path.expanduser("~/.cache/tensorflowtools"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HUGGINGFACE_DIR = CACHE_DIR / "huggingface"
HUGGINGFACE_DIR.mkdir(parents=True, exist_ok=True)

PACKAGE_DATA_DIR = HUGGINGFACE_DIR

def get_model_folder(username: str, repository: str) -> Path:
    """
    Returns the folder path where the model will be stored.
    """
    return PACKAGE_DATA_DIR / username / repository

def download_model(username: str, repository: str, redownload: bool = False):
    """
    Downloads the full model repository from Hugging Face (without using git or symlinks),
    and displays a summary of all downloaded files in a formatted table.
    """
    model_folder = get_model_folder(username, repository)

    if model_folder.exists():
        if redownload:
            warnings.warn(f"{bcolors.WARNING}Model already exists at {model_folder}. Deleting and re-downloading it.{bcolors.ENDC}")
            shutil.rmtree(model_folder)
        else:
            warnings.warn(f"{bcolors.WARNING}Model already exists. Use redownload=True to refresh the cache.{bcolors.ENDC}")
            return

    api = HfApi()
    repo_id = f"{username}/{repository}"
    try:
        files = api.list_repo_files(repo_id=repo_id)
    except Exception as e:
        raise RuntimeError(f"{bcolors.FAIL}Failed to list files for {repo_id}: {e}{bcolors.ENDC}")

    model_folder.mkdir(parents=True, exist_ok=True)
    downloaded_files = []

    for file in files:
        file_url = f"https://huggingface.co/{username}/{repository}/resolve/main/{file}"
        local_path = model_folder / file
        local_path.parent.mkdir(parents=True, exist_ok=True)

        _download_with_progress(file_url, local_path, download_color=Colors.CYAN, complete_color=Colors.GREEN)

        size = local_path.stat().st_size
        downloaded_files.append([file, _format_size(size)])

    print(f"\n{bcolors.OKGREEN}Download complete: {repo_id} → {model_folder}{bcolors.ENDC}")

    if downloaded_files:
        print(f"\n{bcolors.HEADER}Downloaded Files:{bcolors.ENDC}")
        table = TableFormatter.generate(["Filename", "Size"], downloaded_files)
        print(table)

def load_model(username: str, repository: str, model_name: str, custom_objects: dict | None | bool = False):
    """
    Loads a Tensorflow Keras model from a Hugging Face repo path.
    Supports: .keras, .h5
    Uses custom_objects.py if found.
    
    :param username: Username that the model was downloaded from
    :param repository: Repository that the model was downloaded from
    :param model_name: Model file to load
    :param custom_objects: Allow custom object loading from custom_objects.py
    :return: Loaded TensorFlow model
    """

    repo_path = PACKAGE_DATA_DIR / username / repository
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    model_path = PACKAGE_DATA_DIR / username / repository / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"The model {model_name} is not found in the huggingface package data directory.")
    
    custom_objects_file = repo_path / "custom_objects.py"
    if custom_objects_file.exists():
        custom_objects = _load_custom_objects_from_file(custom_objects_file)
    else: 
        custom_objects = None
    import tensorflow as tf

    return tf.keras.models.load_model(model_path, custom_objects=custom_objects) # type: ignore

def clear_model_cache():
    """
    Clears anything in the downloaded model cache.
    """
    if PACKAGE_DATA_DIR.exists():
        shutil.rmtree(PACKAGE_DATA_DIR)
    PACKAGE_DATA_DIR.mkdir(parents=True, exist_ok=True)

def delete_model(username: str, repository: str):
    """
    Deletes a specific model repository from the local cache.
    
    :param username: The Hugging Face username
    :param repository: The repository name
    """
    model_path = get_model_folder(username, repository)
    
    if model_path.exists():
        shutil.rmtree(model_path)
        print(f"{bcolors.OKGREEN}Deleted model at: {model_path}{bcolors.ENDC}")
    else:
        raise FileNotFoundError(f"{bcolors.FAIL}Model not found: {model_path}{bcolors.ENDC}")


def list_models():
    """
    Lists all downloaded models in a structured table format.
    Shows: Username, Repository, Total Files, Total Size
    """
    if not PACKAGE_DATA_DIR.exists():
        print(f"{bcolors.WARNING}No models downloaded yet.{bcolors.ENDC}")
        return

    headers = ["Username", "Repository", "Files", "Total Size"]
    data = []

    for user_dir in PACKAGE_DATA_DIR.iterdir():
        if not user_dir.is_dir():
            continue
        for repo_dir in user_dir.iterdir():
            if not repo_dir.is_dir():
                continue

            total_size = 0
            file_count = 0
            for root, _, files in os.walk(repo_dir):
                for file in files:
                    file_path = Path(root) / file
                    total_size += file_path.stat().st_size
                    file_count += 1

            size_str = _format_size(total_size)
            data.append([user_dir.name, repo_dir.name, file_count, size_str])

    if data:
        print(TextFormat.style("\nDownloaded Models:", TextFormat.BOLD, TextFormat.COLORS["green"]))
        print(TableFormatter.generate(headers, data))
    else:
        print(f"{bcolors.WARNING}No models found in cache.{bcolors.ENDC}")

def get_cache_dir():
    """Gets the cache dir where user/model folders are kept"""
    return str(HUGGINGFACE_DIR)

#   Copyright 2025 Rihaan Meher
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