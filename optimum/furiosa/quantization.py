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

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union

import numpy as np
import onnx
import tqdm
from datasets import Dataset, load_dataset
from transformers import AutoConfig

from furiosa.optimizer import optimize_model
from furiosa.quantizer import quantize

from .configuration import CalibrationConfig, FuriosaAIConfig, QuantizationConfig
from .modeling import FuriosaAIModel
from .quantization_base import OptimumQuantizer


if TYPE_CHECKING:
    from transformers import PretrainedConfig

LOGGER = logging.getLogger(__name__)


class FuriosaAICalibrationDataReader:
    __slots__ = ["batch_size", "dataset", "_dataset_iter", "input_datatypes"]

    def __init__(self, dataset: Dataset, input_datatypes, batch_size: int = 1):
        if dataset is None:
            raise ValueError("Provided dataset is None.")

        if input_datatypes is None:
            raise ValueError("Provided input_datatypes is None.")

        if batch_size <= 0:
            raise ValueError(f"Provided batch_size should be >= 1 (got: {batch_size}).")

        self.dataset = dataset
        self.input_datatypes = input_datatypes
        self.batch_size = batch_size

        self._dataset_iter = iter(self.dataset)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __next__(self):
        featurized_samples = None
        try:
            featurized_samples = []
            for _ in range(self.batch_size):
                sample = next(self._dataset_iter)

                input_list = [[] for i in range(len(sample))]
                for i, name in enumerate(sample):
                    input_list[i] += [sample[name]]
                input_list = [
                    np.array(d, onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[self.input_datatypes[i]])
                    for i, d in enumerate(input_list)
                ]

                featurized_samples.append(input_list)

        except StopIteration:
            raise StopIteration

        if len(featurized_samples) > 0:
            return featurized_samples

        raise StopIteration

    def __iter__(self):
        return self


class FuriosaAIQuantizer(OptimumQuantizer):
    """
    Handles the FuriosaAI quantization process for models shared on huggingface.co/models.
    """

    def __init__(self, model_path: Path, config: Optional["PretrainedConfig"] = None):
        """
        Args:
            model_path (`Path`):
                Path to the onnx model files you want to quantize.
            config (`Optional[PretrainedConfig]`, *optional*):
                The configuration of the model.
        """
        super().__init__()
        self.model_path = model_path
        self.config = config
        if self.config is None:
            try:
                self.config = AutoConfig.from_pretrained(self.model_path.parent)
            except OSError:
                LOGGER.warning(
                    f"Could not load the config for {self.model_path} automatically, this might make "
                    "the quantized model harder to use because it will not be able to be loaded by an FuriosaAIModel without "
                    "having to specify the configuration explicitly."
                )
        self._calibrator = None
        self._calibration_config = None

    @classmethod
    def from_pretrained(
        cls,
        model_or_path: Union["FuriosaAIQuantizer", str, Path],
        file_name: Optional[str] = None,
    ) -> "FuriosaAIQuantizer":
        """
        Instantiates a `FuriosaAIQuantizer` from a model path.

        Args:
            model_or_path (`Union[FuriosaAIModel, str, Path]`):
                Can be either:
                    - A path to a saved exported ONNX Intermediate Representation (IR) model, e.g., `./my_model_directory/.
                    - Or an `FuriosaAIModelModelForXX` class, e.g., `FuriosaAIModelModelForImageClassification`.
            file_name(`Optional[str]`, *optional*):
                Overwrites the default model file name from `"model.onnx"` to `file_name`.
                This allows you to load different model files from the same repository or directory.
        Returns:
            An instance of `FuriosaAIQuantizer`.
        """
        furiosa_quantizer_error_message = "FuriosaAIQuantizer does not support multi-file quantization. Please create separate FuriosaAIQuantizer instances for each model/file, by passing the argument `file_name` to FuriosaAIQuantizer.from_pretrained()."

        if isinstance(model_or_path, str):
            model_or_path = Path(model_or_path)

        path = None
        if isinstance(model_or_path, Path) and file_name is None:
            onnx_files = list(model_or_path.glob("*.onnx"))
            if len(onnx_files) == 0:
                raise FileNotFoundError(f"Could not find any model file in {model_or_path}")
            elif len(onnx_files) > 1:
                raise RuntimeError(
                    f"Found too many ONNX model files in {model_or_path}. {furiosa_quantizer_error_message}"
                )
            file_name = onnx_files[0].name

        if isinstance(model_or_path, FuriosaAIModel):
            if path is None:
                if isinstance(model_or_path.model, str) and model_or_path.model.endswith(".onnx"):
                    path = Path(model_or_path.model)
            else:
                raise ValueError(
                    "Currently, quantization of only ONNX files is supported using the optimum-furiosa repository!"
                )
        elif os.path.isdir(model_or_path):
            path = Path(model_or_path) / file_name
        else:
            raise ValueError(f"Unable to load model from {model_or_path}.")
        return cls(path)

    def fit(
        self,
        dataset: Dataset,
        calibration_config: CalibrationConfig,
        batch_size: int = 1,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Performs the calibration step and computes the quantization ranges.

        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.
            calibration_config ([`~CalibrationConfig`]):
                The configuration containing the parameters related to the calibration step.
            batch_size (`int`, *optional*, defaults to 1):
                The batch size to use when collecting the quantization ranges values.

        Returns:
            The dictionary mapping the nodes name to their quantization ranges.
        """
        # If a dataset is provided, then we are in a static quantization mode
        LOGGER.info(
            f"Using static quantization schema ("
            f"dataset: {calibration_config.dataset_name}, method: {calibration_config.method}"
            f")"
        )

        self.partial_fit(
            dataset,
            calibration_config,
            batch_size,
        )
        return self.compute_ranges()

    def _load_model_and_optimize(self):
        model = onnx.load(Path(self.model_path).as_posix())
        self.onnx_model = optimize_model(model)

    def partial_fit(self, dataset: Dataset, calibration_config: CalibrationConfig, batch_size: int = 1):
        """
        Performs the calibration step and collects the quantization ranges without computing them.

        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.
            calibration_config (`CalibrationConfig`):
                The configuration containing the parameters related to the calibration step.
            batch_size (`int`, *optional*, defaults to 1):
                The batch size to use when collecting the quantization ranges values.
        """
        self._calibration_config = calibration_config

        # If no calibrator, then create one
        if calibration_config.method is not None:
            LOGGER.info(f"Creating calibrator: {calibration_config.method}({calibration_config})")
            self._load_model_and_optimize()

            self._calibrator = calibration_config.create_calibrator(
                model=self.onnx_model,
            )

        def get_input_datatypes(model):
            input_datatypes = []

            for input in model.graph.input:
                input_type = input.type.tensor_type.elem_type
                input_datatypes.extend([input_type])

            return input_datatypes

        input_datatypes = get_input_datatypes(self.onnx_model)

        LOGGER.info("Collecting tensors statistics...")
        reader = FuriosaAICalibrationDataReader(dataset, input_datatypes, batch_size)
        for data in tqdm.tqdm(reader):
            self._calibrator.collect_data(data)

    def compute_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Computes the quantization ranges.

        Returns:
            The dictionary mapping the nodes name to their quantization ranges.
        """
        if self._calibrator is None:
            raise ValueError(
                "Calibrator is None, please call `partial_fit` or `fit` method at least ones to compute ranges."
            )

        LOGGER.info("Computing calibration ranges")
        return self._calibrator.compute_range()

    def quantize(
        self,
        quantization_config: QuantizationConfig,
        save_dir: Union[str, Path],
        file_suffix: Optional[str] = "quantized",
        calibration_tensors_range: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Path:
        """
        Quantizes a model given the optimization specifications defined in `quantization_config`.

        Args:
            quantization_config (`QuantizationConfig`):
                The configuration containing the parameters related to quantization.
            save_dir (`Union[str, Path]`):
                The directory where the quantized model should be saved.
            file_suffix (`Optional[str]`, *optional*, defaults to `"quantized"`):
                The file_suffix used to save the quantized model.
            calibration_tensors_range (`Optional[Dict[NodeName, Tuple[float, float]]]`, *optional*):
                The dictionary mapping the nodes name to their quantization ranges, used and required only when applying
                static quantization.

        Returns:
            The path of the resulting quantized model.
        """

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.onnx_model is None:
            self._load_model_and_optimize()

        LOGGER.info("Quantizing model...")
        model_quantized = quantize(
            self.onnx_model,
            calibration_tensors_range,
            with_quantize=quantization_config.with_quantize,
            normalized_pixel_outputs=quantization_config.normalized_pixel_outputs,
        )

        suffix = f"_{file_suffix}" if file_suffix else ""
        quantized_model_path = save_dir.joinpath(f"{self.model_path.stem}{suffix}").with_suffix(".dfg")
        LOGGER.info(f"Saving quantized model at: {save_dir}")
        with open(quantized_model_path.as_posix(), "wb") as f:
            f.write(bytes(model_quantized))

        # Create and save the configuration summarizing all the parameters related to quantization
        furiosa_config = FuriosaAIConfig(quantization=quantization_config, calibration=self._calibration_config)
        furiosa_config.save_pretrained(save_dir)

        if self.config is not None:
            self.config.save_pretrained(save_dir)

        return Path(save_dir)

    def get_calibration_dataset(
        self,
        dataset_name: str,
        num_samples: int = 100,
        dataset_config_name: Optional[str] = None,
        dataset_split: Optional[str] = None,
        preprocess_function: Optional[Callable] = None,
        preprocess_batch: bool = True,
        seed: int = 2016,
        use_auth_token: bool = False,
    ) -> Dataset:
        """
        Creates the calibration `datasets.Dataset` to use for the post-training static quantization calibration step.

        Args:
            dataset_name (`str`):
                The dataset repository name on the Hugging Face Hub or path to a local directory containing data files
                to load to use for the calibration step.
            num_samples (`int`, *optional*, defaults to 100):
                The maximum number of samples composing the calibration dataset.
            dataset_config_name (`Optional[str]`, *optional*):
                The name of the dataset configuration.
            dataset_split (`Optional[str]`, *optional*):
                Which split of the dataset to use to perform the calibration step.
            preprocess_function (`Optional[Callable]`, *optional*):
                Processing function to apply to each example after loading dataset.
            preprocess_batch (`bool`, *optional*, defaults to `True`):
                Whether the `preprocess_function` should be batched.
            seed (`int`, *optional*, defaults to 2016):
                The random seed to use when shuffling the calibration dataset.
            use_auth_token (`bool`, *optional*, defaults to `False`):
                Whether to use the token generated when running `transformers-cli login` (necessary for some datasets
                like ImageNet).
        Returns:
            The calibration `datasets.Dataset` to use for the post-training static quantization calibration
            step.
        """
        calib_dataset = load_dataset(
            dataset_name,
            name=dataset_config_name,
            split=dataset_split,
            use_auth_token=use_auth_token,
        )

        if num_samples is not None:
            num_samples = min(num_samples, len(calib_dataset))
            calib_dataset = calib_dataset.shuffle(seed=seed).select(range(num_samples))

        if preprocess_function is not None:
            processed_calib_dataset = calib_dataset.map(preprocess_function, batched=preprocess_batch)
        else:
            processed_calib_dataset = calib_dataset

        return self.clean_calibration_dataset(processed_calib_dataset)

    def clean_calibration_dataset(self, dataset: Dataset) -> Dataset:
        model = onnx.load(self.model_path)
        model_inputs = {input.name for input in model.graph.input}
        ignored_columns = list(set(dataset.column_names) - model_inputs)
        return dataset.remove_columns(ignored_columns)
