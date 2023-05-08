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

import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Union

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import PretrainedConfig
from transformers.file_utils import add_start_docstrings

# Import Furiosa SDK
from furiosa.runtime import session
from optimum.exporters import TasksManager
from optimum.exporters.onnx import export
from optimum.modeling_base import OptimizedModel

from .utils import ONNX_WEIGHTS_NAME


# if is_transformers_version("<", "4.25.0"):
#     from transformers.generation_utils import GenerationMixin
# else:
#     from transformers.generation import GenerationMixin

logger = logging.getLogger(__name__)

_SUPPORTED_DEVICES = {
    "WARBOY",
}


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
        model,
        config: PretrainedConfig = None,
        device: str = None,
        furiosa_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        self.config = config
        self.model_save_dir = model_save_dir
        self._device = device.upper()
        self.furiosa_config = furiosa_config
        self.preprocessors = kwargs.get("preprocessors", [])
        enable_compilation = kwargs.get("compile", True)

        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(model.inputs)}
        self.model = model
        if enable_compilation:
            self.compile()

    @staticmethod
    def load_model(file_name: Union[str, Path]):
        """
        Loads the model.

        Arguments:
            file_name (`str` or `Path`):
                The path of the model ONNX or XML file.
        """
        if isinstance(file_name, str):
            file_name = Path(file_name)
        file_name.with_suffix(".bin") if file_name.suffix == ".xml" else None
        return 0
        # return core.read_model(file_name, bin_file_name)

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None, **kwargs):
        pass

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
        from_onnx: bool = False,
        local_files_only: bool = False,
        **kwargs,
    ):
        """
        Loads a model and its configuration file from a directory or the HF Hub.

        Arguments:
            model_id (`str` or `Path`):
                The directory from which to load the model.
                Can be either:
                    - The model id of a pretrained model hosted inside a model repo on huggingface.co.
                    - The path to a directory containing the model weights.
            use_auth_token (`str` or `bool`):
                The token to use as HTTP bearer authorization for remote files. Needed to load models from a private
                repository.
            revision (`str`, *optional*):
                The specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Union[str, Path]`, *optional*):
                The path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            file_name(`str`, *optional*):
                The file name of the model to load. Overwrites the default file name and allows one to load the model
                with a different name.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
        """
        default_file_name = ONNX_WEIGHTS_NAME
        file_name = file_name or default_file_name

        # Load the model from local directory
        if os.path.isdir(model_id):
            file_name = os.path.join(model_id, file_name)
            if os.path.isfile(os.path.join(model_id, "ov_model.xml")):
                file_name = os.path.join(model_id, "ov_model.xml")
                logger.warning(
                    "The file names `ov_model.xml` and `ov_model.bin` will be soon deprecated."
                    "Make sure to rename your file to respectively `openvino_model.xml` and `openvino_model.bin`"
                )
            model = cls.load_model(file_name)
            model_save_dir = model_id
        # Download the model from the hub
        else:
            model_file_names = [file_name]
            if not from_onnx:
                model_file_names.append(file_name.replace(".xml", ".bin"))
            file_names = []
            try:
                for file_name in model_file_names:
                    model_cache_path = hf_hub_download(
                        repo_id=model_id,
                        filename=file_name,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                    file_names.append(model_cache_path)
            except EntryNotFoundError:
                file_names = []
            model_save_dir = Path(model_cache_path).parent
            model = cls.load_model(file_names[0])
        return cls(model, config=config, model_save_dir=model_save_dir, **kwargs)

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

        model_kwargs = {
            "revision": revision,
            "use_auth_token": use_auth_token,
            "cache_dir": cache_dir,
            "subfolder": subfolder,
            "local_files_only": local_files_only,
            "force_download": force_download,
        }

        model = TasksManager.get_model_from_task(task, model_id, **model_kwargs)

        model_type = model.config.model_type.replace("_", "-")
        onnx_config_class = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model=model,
            task=task,
            model_name=model_id,
            model_type=model_type,
        )

        onnx_config = onnx_config_class(model.config)
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        # Export the model to the ONNX format
        export(
            model=model,
            config=onnx_config,
            opset=onnx_config.DEFAULT_ONNX_OPSET,
            output=save_dir_path / ONNX_WEIGHTS_NAME,
        )

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

    def compile(self):
        if self.sess is None:
            logger.info("Compiling the model and creating the session ...")
            self.sess = session.create(self.model)

    def _reshape(
        self,
        model,
        batch_size: int,
        sequence_length: int,
        height: int = None,
        width: int = None,
    ):
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            shapes[inputs][0] = batch_size
            shapes[inputs][1] = sequence_length
            if height is not None:
                shapes[inputs][2] = height
            if width is not None:
                shapes[inputs][3] = width
        model.reshape(shapes)
        return model

    def reshape(self, batch_size: int, sequence_length: int, height: int = None, width: int = None):
        """
        Propagates the given input shapes on the model's layers, fixing the inputs shapes of the model.

        Arguments:
            batch_size (`int`):
                The batch size.
            sequence_length (`int`):
                The sequence length or number of channels.
            height (`int`, *optional*):
                The image height.
            width (`int`, *optional*):
                The image width.
        """
        self.is_dynamic = True if batch_size == -1 and sequence_length == -1 else False
        self.model = self._reshape(self.model, batch_size, sequence_length, height, width)
        self.sess = None
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def _auto_model_to_task(cls, auto_model_class):
        """
        Get the task corresponding to a class (for example AutoModelForXXX in transformers).
        """
        return cls._AUTOMODELS_TO_TASKS[auto_model_class.__name__]

    # def can_generate(self) -> bool:
    #     """
    #     Returns whether this model can generate sequences with `.generate()`.
    #     """
    #     if isinstance(self, GenerationMixin):
    #         return True
    #     return False
