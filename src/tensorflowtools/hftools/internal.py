import requests
import importlib.util
import sys
from textformat.progress import DownloadProgressBar
from pathlib import Path

def _download_with_progress(url: str, output_path: Path, download_color, complete_color):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    if response.status_code != 200:
        raise Exception(f"Failed to download from {url}. HTTP status: {response.status_code}")

    progress = DownloadProgressBar(total, prefix=output_path.name, 
                           download_color=download_color, complete_color=complete_color)
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                progress.update(progress.downloaded + len(chunk))
    print()

def _load_custom_objects_from_file(file_path: Path) -> dict:
    """
    Safely imports the `CUSTOM_OBJECTS` dictionary from a custom_objects.py file.
    """
    if not file_path.exists():
        return {}

    module_name = f"custom_objects_{hash(file_path)}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return getattr(module, "CUSTOM_OBJECTS", {})
    except Exception as e:
        raise ImportError(f"Failed to import CUSTOM_OBJECTS from {file_path}: {e}")

def _format_size(size_in_bytes):
    """Format the size in a human-readable format."""
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.1f} PiB"

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