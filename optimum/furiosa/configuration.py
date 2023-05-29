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

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional, Sequence, Union

import onnx
from datasets import Dataset

from furiosa.quantizer import CalibrationMethod, Calibrator
from optimum.configuration_utils import BaseConfig


DEFAULT_QUANTIZATION_CONFIG = {}


@dataclass
class CalibrationConfig:
    """
    CalibrationConfig is the configuration class handling all the FurioaAI parameters related to the calibration
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
        percentage (`Optional[float]`, defaults to `None`):
            The percentage to use when computing the activations quantization ranges when performing the calibration
            step using the Percentile method.
    """

    dataset_name: str
    dataset_config_name: str
    dataset_split: str
    dataset_num_samples: int
    method: CalibrationMethod
    percentage: Optional[float] = None

    def create_calibrator(
        self,
        model: Union[onnx.ModelProto, bytes],
    ) -> Calibrator:
        return Calibrator(model, self.method, percentage=self.percentage)


class AutoCalibrationConfig:
    @staticmethod
    def create_calibration_config(dataset: Dataset, method: CalibrationMethod, percentile: float = None):
        return CalibrationConfig(
            dataset_name=dataset.info.builder_name,
            dataset_config_name=dataset.info.config_name,
            dataset_split=str(dataset.split),
            dataset_num_samples=dataset.num_rows,
            method=method,
            percentage=percentile,
        )

    @staticmethod
    def minmax_asym(dataset: Dataset) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.

        Returns:
            The calibration configuration.
        """
        return AutoCalibrationConfig.create_calibration_config(
            dataset,
            method=CalibrationMethod.MIN_MAX_ASYM,
        )

    def minmax_sym(dataset: Dataset) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.

        Returns:
            The calibration configuration.
        """
        return AutoCalibrationConfig.create_calibration_config(
            dataset,
            method=CalibrationMethod.MIN_MAX_SYM,
        )

    @staticmethod
    def entropy_asym(
        dataset: Dataset,
    ) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.

        Returns:
            The calibration configuration.
        """
        return AutoCalibrationConfig.create_calibration_config(
            dataset,
            method=CalibrationMethod.ENTROPY_ASYM,
        )

    @staticmethod
    def entropy_sym(
        dataset: Dataset,
    ) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.

        Returns:
            The calibration configuration.
        """
        return AutoCalibrationConfig.create_calibration_config(
            dataset,
            method=CalibrationMethod.ENTROPY_SYM,
        )

    @staticmethod
    def percentiles_asym(dataset: Dataset, percentile: float = 99.999) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.
            percentile (`float`):
                The percentile to use when computing the activations quantization ranges.

        Returns:
            The calibration configuration.
        """
        return AutoCalibrationConfig.create_calibration_config(
            dataset,
            method=CalibrationMethod.PERCENTILE_ASYM,
            percentile=percentile,
        )

    @staticmethod
    def percentiles_sym(dataset: Dataset, percentile: float = 99.999) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.
            percentile (`float`):
                The percentile to use when computing the activations quantization ranges.

        Returns:
            The calibration configuration.
        """
        return AutoCalibrationConfig.create_calibration_config(
            dataset,
            method=CalibrationMethod.PERCENTILE_SYM,
            percentile=percentile,
        )

    @staticmethod
    def mse_asym(dataset: Dataset) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.

        Returns:
            The calibration configuration.
        """
        return AutoCalibrationConfig.create_calibration_config(
            dataset,
            method=CalibrationMethod.MSE_ASYM,
        )

    @staticmethod
    def mse_sym(dataset: Dataset) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.

        Returns:
            The calibration configuration.
        """
        return AutoCalibrationConfig.create_calibration_config(
            dataset,
            method=CalibrationMethod.MSE_SYM,
        )

    @staticmethod
    def sqnr_asym(dataset: Dataset) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.

        Returns:
            The calibration configuration.
        """
        return AutoCalibrationConfig.create_calibration_config(
            dataset,
            method=CalibrationMethod.SQNR_ASYM,
        )

    @staticmethod
    def sqnr_sym(dataset: Dataset) -> CalibrationConfig:
        """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.

        Returns:
            The calibration configuration.
        """
        return AutoCalibrationConfig.create_calibration_config(
            dataset,
            method=CalibrationMethod.SQNR_SYM,
        )


@dataclass
class QuantizationConfig:
    """
    QuantizationConfig is the configuration class handling all the FuriosaAI quantization parameters.

     Args:
        with_quantize  (`bool`, defaults to `True`):
            WWhether to put a Quantize operator at the beginning of the resulting model.
        normalized_pixel_outputs (` Sequence[int`, defaults to `None`)::
            A sequence of indices of output tensors in the ONNX model that produce pixel values in a normalized format
            ranging from 0.0 to 1.0. If specified, the corresponding output tensors in the resulting quantized model
            will generate pixel values in an unnormalized format from 0 to 255, represented as unsigned 8-bit integers (uint8).
    """

    with_quantize: bool = True
    normalized_pixel_outputs: Sequence[int] = None


class FuriosaAIConfig(BaseConfig):
    CONFIG_NAME = "furiosa_config.json"
    FULL_CONFIGURATION_FILE = "furiosa_config.json"

    def __init__(
        self,
        opset: Optional[int] = None,
        quantization: Optional[QuantizationConfig] = None,
        calibration: Optional[CalibrationConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.quantization = self.dataclass_to_dict(quantization)
        self.calibration = self.dataclass_to_dict(calibration)
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
