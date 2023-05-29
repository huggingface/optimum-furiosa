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

import numpy as np

from furiosa.runtime.tensor import DataType
from optimum.exporters import TasksManager
from optimum.exporters.onnx import export


FAI_ENF_FILE_NAME = "furiosa_model.enf"

ONNX_WEIGHTS_NAME = "model.onnx"
ONNX_WEIGHTS_NAME_STATIC = "model_static.onnx"

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
    model = TasksManager.get_model_from_task(task, model_id)

    model_type = model.config.model_type.replace("_", "-")
    model.config.save_pretrained(save_dir)

    onnx_config_class = TasksManager.get_exporter_config_constructor(
        exporter="onnx",
        model=model,
        task=task,
        model_name=model_id,
        model_type=model_type,
    )

    onnx_config = onnx_config_class(model.config)
    save_dir_path = Path(save_dir) / "model_temp.onnx"

    # Export the model to the ONNX format
    export(
        model=model,
        config=onnx_config,
        opset=onnx_config.DEFAULT_ONNX_OPSET,
        output=save_dir_path,
    )

    import onnx
    from onnx import shape_inference
    from onnx.tools import update_model_dims

    model = onnx.load(save_dir_path)
    updated_model = update_model_dims.update_inputs_outputs_dims(model, input_shape_dict, output_shape_dict)
    inferred_model = shape_inference.infer_shapes(updated_model)

    static_model_path = Path(save_dir_path).parent / file_name
    onnx.save(inferred_model, static_model_path)
