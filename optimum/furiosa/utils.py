#  Copyright 2023 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from pathlib import Path
from typing import List, Union

import numpy as np
from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer

from furiosa.runtime.tensor import DataType
from optimum.exporters.onnx import main_export


ONNX_WEIGHTS_NAME = "model.onnx"
ONNX_WEIGHTS_NAME_STATIC = "model_static.onnx"
FURIOSA_ENF_FILE_NAME = "model.enf"
FURIOSA_QUANTIZED_FILE_NAME = "model_quantized.dfg"

MAX_ONNX_OPSET_2022_2_0 = 10
MAX_ONNX_OPSET = 13
MIN_ONNX_QDQ_OPSET = 13

WARBOY_DEVICE = "warboy"

FURIOSA_DTYPE_TO_NUMPY_DTYPE = {
    DataType.UINT8: np.uint8,
    DataType.INT8: np.int8,
    DataType.FLOAT32: np.float32,
}

_HEAD_TO_AUTOMODELS = {
    "image-classification": "FuriosaAIModelForImageClassification",
}


def export_model_to_onnx(model_id, save_dir, input_shape_dict, output_shape_dict, file_name="model.onnx"):
    task = "image-classification"
    main_export(model_id, save_dir, task=task)

    import onnx
    from onnx import shape_inference
    from onnx.tools import update_model_dims

    save_dir_path = Path(save_dir) / "model.onnx"
    model = onnx.load(save_dir_path)
    updated_model = update_model_dims.update_inputs_outputs_dims(model, input_shape_dict, output_shape_dict)
    inferred_model = shape_inference.infer_shapes(updated_model)

    static_model_path = Path(save_dir_path).parent / file_name
    onnx.save(inferred_model, static_model_path)


def maybe_load_preprocessors(src_name_or_path: Union[str, Path], subfolder: str = "") -> List:
    preprocessors = []
    try:
        preprocessors.append(AutoTokenizer.from_pretrained(src_name_or_path, subfolder=subfolder))
    except Exception:
        pass

    try:
        preprocessors.append(AutoProcessor.from_pretrained(src_name_or_path, subfolder=subfolder))
    except Exception:
        pass

    try:
        preprocessors.append(AutoFeatureExtractor.from_pretrained(src_name_or_path, subfolder=subfolder))
    except Exception:
        pass
    return preprocessors


def maybe_save_preprocessors(src_name_or_path: Union[str, Path], dest_dir: Union[str, Path], src_subfolder: str = ""):
    """
    Saves the tokenizer, the processor and the feature extractor when found in `src_dir` in `dest_dir`.

    Args:
        src_dir (`Union[str, Path]`):
            The source directory from which to copy the files.
        dest_dir (`Union[str, Path]`):
            The destination directory to copy the files to.
        src_subfolder (`str`, defaults to `""`):
            In case the preprocessor files are located inside a subfolder of the model directory / repo on the Hugging
            Face Hub, you can specify the subfolder name here.
    """
    if not isinstance(dest_dir, Path):
        dest_dir = Path(dest_dir)

    dest_dir.mkdir(exist_ok=True)
    for preprocessor in maybe_load_preprocessors(src_name_or_path, subfolder=src_subfolder):
        preprocessor.save_pretrained(dest_dir)
