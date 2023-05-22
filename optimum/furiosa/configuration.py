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

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional, Union

import onnx
from datasets import Dataset

from furiosa.quantizer import CalibrationMethod, Calibrator
from optimum.configuration_utils import BaseConfig


DEFAULT_QUANTIZATION_CONFIG = {}


@dataclass
class CalibrationConfig:
    """
    CalibrationConfig is the configuration class handling all the ONNX Runtime parameters related to the calibration
    step of static quantization.

    Args:
        dataset_name (`str`):
            The name of the calibration dataset.
        dataset_config_name (`str`):
            The name of the calibration dataset configuration.
        dataset_split (`str`):
            Which split of the dataset is used to perform the calibration step.
        dataset_num_samples (`int`):
            The number of samples composing the calibration dataset.
        method (`CalibrationMethod`):
            The method chosen to calculate the activations quantization parameters using the calibration dataset.
    """

    dataset_name: str
    dataset_config_name: str
    dataset_split: str
    dataset_num_samples: int
    method: CalibrationMethod

    def create_calibrator(
        self,
        model: Union[onnx.ModelProto, bytes],
    ) -> Calibrator:
        kwargs = {
            "model": model,
            "calibrate_method": self.method,
        }
        return Calibrator(**kwargs)


class AutoCalibrationConfig:
    @staticmethod
    def minmax(dataset: Dataset, moving_average: bool = False) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.

        Returns:
            The calibration configuration.
        """
        return CalibrationConfig(
            dataset_name=dataset.info.builder_name,
            dataset_config_name=dataset.info.config_name,
            dataset_split=str(dataset.split),
            dataset_num_samples=dataset.num_rows,
            method=CalibrationMethod.MIN_MAX_ASYM,
        )


@dataclass
class QuantizationConfig:
    """
    QuantizationConfig is the configuration class handling all the ONNX Runtime quantization parameters.

    Args:
        activations_symmetric (`bool`, defaults to `False`):
            Whether to apply symmetric quantization on the activations.
        weights_symmetric (`bool`, defaults to `True`):
            Whether to apply symmetric quantization on the weights.
    """

    activations_symmetric: bool = False
    weights_symmetric: bool = True

    @property
    def use_symmetric_calibration(self) -> bool:
        return self.activations_symmetric and self.weights_symmetric


class FuriosaAIConfig(BaseConfig):
    CONFIG_NAME = "furiosa_config.json"
    FULL_CONFIGURATION_FILE = "furiosa_config.json"

    def __init__(
        self,
        opset: Optional[int] = None,
        use_external_data_format: bool = False,
        one_external_file: bool = True,
        quantization: Optional[QuantizationConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.opset = opset
        self.quantization = self.dataclass_to_dict(quantization)
        self.optimum_version = kwargs.pop("optimum_version", None)

    @staticmethod
    def dataclass_to_dict(config) -> dict:
        new_config = {}
        if config is None:
            return new_config
        if isinstance(config, dict):
            return config
        for k, v in asdict(config).items():
            if isinstance(v, Enum):
                v = v.name
            elif isinstance(v, list):
                v = [elem.name if isinstance(elem, Enum) else elem for elem in v]
            new_config[k] = v
        return new_config
