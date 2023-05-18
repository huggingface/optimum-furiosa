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
from typing import Union

import numpy as np
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
)
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import (
    ImageClassifierOutput,
)

from .modeling_base import FuriosaAIBaseModel


logger = logging.getLogger(__name__)


_FEATURE_EXTRACTOR_FOR_DOC = "AutoFeatureExtractor"

MODEL_START_DOCSTRING = r"""
    This model inherits from [`optimum.furiosa.FuriosaAIBaseModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)
    Parameters:
        model (`furiosa.runtime.model`): is the main class used to run inference.
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig)
            is the Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the [`~furiosa.modeling.FuriosaAIBaseModel.from_pretrained`] method to load the model weights.
        device (`str`, defaults to `"CPU"`):
            The device type for which the model will be optimized for. The resulting compiled model will contains nodes specific to this device.
        furiosa_config (`Optional[Dict]`, defaults to `None`):
            The dictionnary containing the informations related to the model compilation.
        compile (`bool`, defaults to `True`):
            Disable the model compilation during the loading step when set to `False`.
"""

IMAGE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.Tensor`):
            Pixel values corresponding to the images in the current batch.
            Pixel values can be obtained from encoded images using [`AutoFeatureExtractor`](https://huggingface.co/docs/transformers/autoclass_tutorial#autofeatureextractor).
"""


class FuriosaAIModel(FuriosaAIBaseModel):
    base_model_prefix = "furiosa_model"
    auto_model_class = AutoModel

    def __init__(self, model, config: transformers.PretrainedConfig = None, **kwargs):
        super().__init__(model, config, **kwargs)
        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        self.auto_model_class.register(AutoConfig, self.__class__)
        self.device = torch.device("cpu")

    def to(self, device: str):
        """
        Use the specified `device` for inference. For example: "cpu" or "gpu". `device` can
        be in upper or lower case. To speed up first inference, call `.compile()` after `.to()`.
        """
        self._device = device.upper()
        self.sess = None
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError


IMAGE_CLASSIFICATION_EXAMPLE = r"""
    Example of image classification using `transformers.pipelines`:
    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.furiosa import {model_class}

    >>> preprocessor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True, input_shape_dict="{"pixel_values": [1, 3, 224, 224]}", output_shape_dict="{"logits": [1, 1000]}",)
    >>> pipe = pipeline("image-classification", model=model, feature_extractor=preprocessor)
    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> outputs = pipe(url)
    ```
"""


@add_start_docstrings(
    """
    FuriosaAI Model with a ImageClassifierOutput for image classification tasks.
    """,
    MODEL_START_DOCSTRING,
)
class FuriosaAIModelForImageClassification(FuriosaAIModel):
    export_feature = "image-classification"
    auto_model_class = AutoModelForImageClassification

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)

    @add_start_docstrings_to_model_forward(
        IMAGE_INPUTS_DOCSTRING.format("batch_size, num_channels, height, width")
        + IMAGE_CLASSIFICATION_EXAMPLE.format(
            processor_class=_FEATURE_EXTRACTOR_FOR_DOC,
            model_class="FuriosaAIModelForImageClassification",
            checkpoint="microsoft/resnet50",
        )
    )
    def forward(
        self,
        pixel_values: Union[torch.Tensor, np.ndarray],
        **kwargs,
    ):
        self.compile()

        np_inputs = isinstance(pixel_values, np.ndarray)
        if not np_inputs:
            pixel_values = np.array(pixel_values)

        # Run inference
        outputs = self.sess.run(pixel_values)
        logits = torch.from_numpy(outputs[0].numpy()) if not np_inputs else outputs[0].numpy()
        return ImageClassifierOutput(logits=logits)
