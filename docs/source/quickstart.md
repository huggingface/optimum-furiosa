<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


# Quickstart

🤗 Optimum Furiosa was designed with one goal in mind: **making inference straightforward for 🤗 models while leveraging the complete power of Furiosa NPUs**.


Let's see now how we can apply static quantization with Optimum Furiosa

Static quantization relies on feeding batches of data through the model to estimate the activation quantization parameters ahead of inference time. To support this, 🤗 Optimum allows you to provide a _calibration dataset_. The calibration dataset can be a simple `Dataset` object from the 🤗 Datasets library, or any dataset that's hosted on the Hugging Face Hub. For this example, we'll pick the [`sst2`](https://huggingface.co/datasets/glue/viewer/sst2/test) dataset that the model was originally trained on:

```python
>>> from functools import partial
>>> from optimum.furiosa.configuration import AutoCalibrationConfig

# Define the processing function to apply to each example after loading the dataset
>>> def preprocess_fn(ex, tokenizer):
...     return tokenizer(ex["sentence"])

>>> # Create the calibration dataset
>>> calibration_dataset = quantizer.get_calibration_dataset(
...     "glue",
...     dataset_config_name="sst2",
...     preprocess_function=partial(preprocess_fn, tokenizer=tokenizer),
...     num_samples=50,
...     dataset_split="train",
... )
>>> # Create the calibration configuration containing the parameters related to calibration.
>>> calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
>>> # Perform the calibration step: computes the activations quantization ranges
>>> ranges = quantizer.fit(
...     dataset=calibration_dataset,
...     calibration_config=calibration_config,
...     operators_to_quantize=qconfig.operators_to_quantize,
... )
>>> # Apply static quantization on the model
>>> quantizer.quantize(    # doctest: +IGNORE_RESULT
...     save_dir=save_directory,
...     calibration_tensors_range=ranges,
...     quantization_config=qconfig,
... )
```

In this example, we've quantized a model from the Hugging Face Hub, but it could also be a path to a local model directory. The result from applying the `quantize()` method is a `model_quantized.onnx` file that can be used to run inference. Here's an example of how to load an ONNX Runtime model and generate predictions with it:



## Ready-to-Use Examples

Here are examples for various modalities and tasks that can be used out of the box:
- Images
  - [image classification](https://github.com/huggingface/optimum-furiosa/tree/main/examples/image-classification)