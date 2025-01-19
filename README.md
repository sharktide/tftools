# TFtools 0.0.2

### TFtools Latest

    pip install --upgrade tensorflowtools

### TFtools Nightly

    pip install --upgrade -i https://test.pypi.org/simple/ tensorflowtools

This is a small package with various utilites to do with tensorflow.



**Disclaimer**: This package is not affiliated with or endorsed by TensorFlow, Google, or Huggingface. It is an independent open-source project built to integrate TensorFlow with various tools.

## License

Copyright 2025 Rihaan Meher

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

## Dependencies

This project uses the following open-source libraries:

- **TensorFlow**: Licensed under the Apache 2.0 License.
- **Hugging Face Hub**: Licensed under the Apache 2.0 License.
- **setuptools**: Licensed under the MIT License (https://opensource.org/licenses/MIT).
- **pathlib**: Part of Python (Standard Library).
- **warnings** Part of Python (Standard Library)
- **os**: Part of Python (Standard Library). 



# Submodules

## hftools

<details>
<summary>Functions</summary>

#### download_model_from_huggingface(username, repository, model_id)

This downloads a model named tf_model.h5 or tf_model.keras from huggingface to the tensorflowtools data directory. It can be used with the load_from_hf_cache function in the kerastools submodule

##### example

    import tensorflowtools
    tensorflowtools.hftools.download_model_from_huggingface("sharktide", "recyclebot0", "sharktide/recyclebot0")
    model = tensorflowtools.kerastools.load_from_hf_cache("sharktide", "recyclebot0", "tf_model.h5")
    model.summary


#### clear_model_cache()

This clears the model cache; all downloaded models and configuration files will be deleted

    import tensorflowtools
    tensorflowtools.hftools.download_model_from_huggingface("sharktide", "recyclebot0", "sharktide/recyclebot0")
    model = tensorflowtools.kerastools.load_from_hf_cache("sharktide", "recyclebot0", "tf_model.h5")
    model.summary
    tensorflowtools.hftools.clear_model_cache()
    try:
        model = tensorflowtools.kerastools.load_from_hf_cache("sharktide", "recyclebot0", "tf_model.h5")
    except:
        print("It worked!")

</details>