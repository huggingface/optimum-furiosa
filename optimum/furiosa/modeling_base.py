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
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Tuple, Union

import onnx
from huggingface_hub import hf_hub_download
from transformers import PretrainedConfig
from transformers.file_utils import add_start_docstrings

# Import Furiosa SDK
from furiosa import optimizer
from furiosa.runtime import session
from furiosa.tools.compiler.api import compile
from optimum.exporters import TasksManager
from optimum.exporters.onnx import main_export
from optimum.modeling_base import OptimizedModel

from .utils import (
    FURIOSA_ENF_FILE_NAME,
    FURIOSA_QUANTIZED_FILE_NAME,
    ONNX_WEIGHTS_NAME,
    ONNX_WEIGHTS_NAME_STATIC,
    maybe_load_preprocessors,
    maybe_save_preprocessors,
)


logger = logging.getLogger(__name__)


@add_start_docstrings(
    """
    Base FuriosaAIModel class.
    """,
)
class FuriosaAIBaseModel(OptimizedModel):
    _AUTOMODELS_TO_TASKS = {cls_name: task for task, cls_name in TasksManager._TASKS_TO_AUTOMODELS.items()}
    auto_model_class = None
    export_feature = None

    def __init__(
        self,
        model: Union[bytes, str, Path],
        config: PretrainedConfig = None,
        device: str = None,
        furiosa_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        input_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
        output_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
        **kwargs,
    ):
        self.config = config
        self.model_save_dir = model_save_dir
        self.furiosa_config = furiosa_config
        self.preprocessors = kwargs.get("preprocessors", [])
        enable_compilation = kwargs.get("compile", True)

        self.model = model

        if enable_compilation:
            self.model = self.compile(model, input_shape_dict, output_shape_dict)

        self.create_session()

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None, **kwargs):
        dst_path = Path(save_directory) / FURIOSA_ENF_FILE_NAME

        if isinstance(self.model, (str, Path)):
            copyfile(self.model, dst_path)
        else:
            with open(dst_path, "wb") as f:
                f.write(self.model)

    def create_session(self):
        """
        Create a Furiosa runtime session for the model.

        Creates a session object using the Furiosa runtime for executing the model.

        Returns:
            None
        """
        self.sess = session.create(self.model)
        self.input_num = self.sess.input_num
        self.inputs_to_dtype = []
        for i in range(self.input_num):
            self.inputs_to_dtype.append(self.sess.input(i).dtype)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        from_onnx: bool = False,
        from_quantized: bool = False,
        local_files_only: bool = False,
        input_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
        output_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
        **kwargs,
    ):
        """
        Loads a model and its configuration file from a directory or the Hugging Face Hub.

        Args:
            model_id (Union[str, Path]):
                The directory from which to load the model. Can be either:
                    - The model ID of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.
            config (PretrainedConfig):
                The configuration object associated with the model.
            use_auth_token (Union[bool, str, None], defaults to None):
                The token to use as HTTP bearer authorization for remote files. Needed to load models from a private repository.
            revision (Union[str, None], defaults to None):
                The specific model version to use. It can be a branch name, a tag name, or a commit ID.
            force_download (bool, defaults to False):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist.
            cache_dir (str, defaults to None):
                The path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.
            file_name (str, defaults to None):
                The file name of the model to load. Overwrites the default file name and allows one to load the model with a different name.
            subfolder (str, defaults to ""):
                The subfolder to load the model.
            from_onnx (bool, defaults to False):
                Whether the model is being loaded from an ONNX file.
            from_quantized (bool, defaults to False):
                Whether the model is being loaded from a quantized file.
            local_files_only (bool, defaults to False):
                Whether or not to only look at local files (i.e., do not try to download the model).
            input_shape_dict (Dict[str, Tuple[int]], defaults to None):
                A dictionary specifying the input shapes for dynamic models.
            output_shape_dict (Dict[str, Tuple[int]], defaults to None):
                A dictionary specifying the output shapes for dynamic models.
            **kwargs:
                Additional keyword arguments to be passed to the underlying model loading function.

        Returns:
            An instance of the model class loaded from the specified directory or Hugging Face Hub.
        """
        if from_onnx:
            default_file_name = ONNX_WEIGHTS_NAME
        elif from_quantized:
            default_file_name = FURIOSA_QUANTIZED_FILE_NAME
        else:
            default_file_name = FURIOSA_ENF_FILE_NAME

        file_name = file_name or default_file_name

        # Load the model from local directory
        if Path(model_id).is_dir():
            file_path = Path(model_id) / file_name
            model_save_dir = model_id
            preprocessors = maybe_load_preprocessors(model_id)
        # Download the model from the hub
        else:
            file_path = hf_hub_download(
                repo_id=model_id,
                filename=file_name,
                subfolder=subfolder,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )
            model_save_dir = Path(file_path).parent
            preprocessors = maybe_load_preprocessors(model_id, subfolder=subfolder)

        model = cls.load_model(file_path, input_shape_dict, output_shape_dict)

        return cls(
            model, config=config, model_save_dir=model_save_dir, compile=False, preprocessors=preprocessors, **kwargs
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        task: Optional[str] = None,
        **kwargs,
    ):
        """
        Export a vanilla Transformers model into an ONNX model using `transformers.onnx.export_onnx`.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.            save_dir (`str` or `Path`):
                The directory where the exported ONNX model should be saved, default to
                `transformers.file_utils.default_cache_path`, which is the cache directory for transformers.
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        # Export the model to the ONNX format
        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            do_validation=False,
            no_post_process=True,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
        )

        config.save_pretrained(save_dir_path)
        maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            from_onnx=True,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **kwargs,
        )

    @classmethod
    def load_model(
        cls,
        model_path: Union[str, Path],
        input_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
        output_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
    ):
        """
        Loads and processes a model for use with the Furiosa framework.

        Args:
            model_path (Union[str, Path]):
                The path to the model file.
            input_shape_dict (Dict[str, Tuple[int]], defaults to None):
                A dictionary specifying the input shapes for dynamic models.
            output_shape_dict (Dict[str, Tuple[int]], defaults to None):
                A dictionary specifying the output shapes for dynamic models.

        Returns:
            If the model is in the 'onnx' or 'dfg' format, the compiled model in the Furiosa binary format is returned.
            If the model is in the 'enf' format, the model path is returned as-is.

        Raises:
            ValueError: If the model format is not supported or invalid.
        """
        model_path = Path(model_path)
        if model_path.suffix in (".onnx", ".dfg"):
            compiled_model = cls.compile(model_path, input_shape_dict, output_shape_dict)
            return compiled_model
        if model_path.suffix == ".enf":
            return model_path

        raise ValueError("Invalid model types. Supported formats are 'onnx', 'dfg', or 'enf'.")

    @classmethod
    def compile(
        cls,
        model: Union[str, Path, bytes],
        input_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
        output_shape_dict: Optional[Dict[str, Tuple[int]]] = None,
    ):
        """
        Compiles the model to the Furiosa binary format.

        Args:
            model (Union[str, Path]):
                The model to be compiled.
            input_shape_dict (Dict[str, Tuple[int]], defaults to None):
                A dictionary specifying the input shapes for dynamic models.
            output_shape_dict (Dict[str, Tuple[int]], defaults to None):
                A dictionary specifying the output shapes for dynamic models.
        Returns:
            The compiled model in the Furiosa binary format.

        Raises:
            ValueError: If the model format is not supported or invalid.
        """
        if isinstance(model, (str, Path)):
            model = cls._reshape(model, input_shape_dict, output_shape_dict)
            input_bytes = Path(model).read_bytes()
        else:
            input_bytes = model

        logger.info("Compiling the model...")
        compiled_model = compile(input_bytes, target_ir="enf")
        return compiled_model

    @staticmethod
    def _check_is_dynamic(model_path: Union[str, Path]):
        is_dynamic = False
        if Path(model_path).suffix == ".onnx":
            model = onnx.load(model_path)
            is_dynamic = any(any(dim.dim_param for dim in inp.type.tensor_type.shape.dim) for inp in model.graph.input)

        return is_dynamic

    @staticmethod
    def optimize_model(model: onnx.ModelProto) -> Path:
        return optimizer.frontend.onnx.optimize_model(model)

    @staticmethod
    def _update_inputs_outputs_dims(
        model_path: Union[str, Path],
        input_shape_dict: Dict[str, Tuple[int]],
        output_shape_dict: Dict[str, Tuple[int]],
    ) -> onnx.ModelProto:
        from onnx import shape_inference
        from onnx.tools import update_model_dims

        model = onnx.load(model_path)

        updated_model = update_model_dims.update_inputs_outputs_dims(model, input_shape_dict, output_shape_dict)
        return shape_inference.infer_shapes(updated_model)

    @classmethod
    def _reshape(
        cls,
        model_path: Union[str, Path],
        input_shape_dict: Dict[str, Tuple[int]],
        output_shape_dict: Dict[str, Tuple[int]],
    ) -> Union[str, Path]:
        """
        Propagates the given input shapes on the model's layers, fixing the input shapes of the model.

        Args:
            model_path (Union[str, Path]):
                Path to the model.
            input_shape_dict (Dict[str, Tuple[int]]):
                Input shapes for the model.
            output_shape_dict (Dict[str, Tuple[int]]):
                Output shapes for the model.

        Returns:
            Union[str, Path]:
                Path to the model after updating the input shapes.

        Raises:
            ValueError: If the model provided has dynamic axes in input/output and no input/output shape is provided.
        """
        if isinstance(model_path, (str, Path)) and Path(model_path).suffix == ".onnx":
            is_dynamic = cls._check_is_dynamic(model_path)
            if is_dynamic:
                if input_shape_dict is None or output_shape_dict is None:
                    raise ValueError(
                        "The model provided has dynamic axes in input/output. Please provide input and output shapes for compilation."
                    )

                model = cls._update_inputs_outputs_dims(model_path, input_shape_dict, output_shape_dict)
                optimized_model = cls.optimize_model(model)

                static_model_path = Path(model_path).parent / ONNX_WEIGHTS_NAME_STATIC
                onnx.save(optimized_model, static_model_path)

                return static_model_path

        return model_path

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def _auto_model_to_task(cls, auto_model_class):
        """
        Get the task corresponding to a class (for example AutoModelForXXX in transformers).
        """
        return cls._AUTOMODELS_TO_TASKS[auto_model_class.__name__]
