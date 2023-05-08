#  Copyright 2022 The HuggingFace Team. All rights reserved.
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

from typing import Dict, List, Optional, Union

from optimum.configuration_utils import BaseConfig


DEFAULT_QUANTIZATION_CONFIG = {}


class FuriosaAIConfig(BaseConfig):
    CONFIG_NAME = "furiosa_config.json"
    FULL_CONFIGURATION_FILE = "furiosa_config.json"

    def __init__(
        self,
        compression: Union[List[Dict], Dict, None] = None,
        input_info: Optional[List] = None,
        save_onnx_model: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.compression = compression or DEFAULT_QUANTIZATION_CONFIG
        self.input_info = input_info
        self.save_onnx_model = save_onnx_model
        self._enable_standard_onnx_export_option()
        self.optimum_version = kwargs.pop("optimum_version", None)
