### ``tensorflowtools.hftools`` — Hugging Face Utilities for TensorFlow

Tools for downloading and loading TensorFlow models from Hugging Face repositories without using git or symlinks.

- Cache Directory
Models are stored in:

```
~/.cache/tensorflowtools/huggingface/<username>/<repository>
```

#### Functions

<details><summary>

``download_model``

</summary>

```python
download_model_from_huggingface(username: str, repository: str, redownload=False)
```

Downloads all files from a Hugging Face repo (without symlinks) and stores them locally to be loaded with the ``load_model`` function.

Parameters:

username: Hugging Face username or org.

repository: Name of the model repository.

redownload: If True, deletes and redownloads the model folder if already present
</details>

<details>
<summary>

``load_model``
</summary>

```python
load_model(username: str, repository: str, model_name: str, custom_objects: bool = False)
```

Loads a model using ``tf.keras.models.load_model()`` from tensorflowtools' model download cache from ``download_model``

Automatically uses custom_objects.py if found.

Supports .keras and .h5 formats.

Parameters:

username: Username from cache directory.

repository: Repo name.

model_name: Filename of the model (e.g. ``tf_model.h5``).

custom_objects: If True, looks for ``custom_objects.py``.

</details>

<details>
<summary>

``list_models``
</summary>

```python
list_models()
```

Prints a table of all downloaded models, showing:

Username

Repository

Number of files

Total size

</details>

<details>
<summary>

``clear_model_cache``
</summary>

```python
clear_model_cache()
```
Deletes all cached models in ~/.cache/tensorflowtools/huggingface/.
</details>

📦 Expected custom_objects.py format
Your model repo may contain a custom_objects.py file like this:

```python
from tensorflow.keras.layers import LeakyReLU

CUSTOM_OBJECTS = {
    "LeakyReLU": LeakyReLU
}
```

This will be used automatically during model loading if the appropriate flags are passed in ``load_model``

> [!CAUTION]
> Using ``custom_objects=True`` can result in the execution of untrusted code unsandboxed on your computer. Only do this if you trust the authors and have verified that the model cache directory has not been tampered with.
