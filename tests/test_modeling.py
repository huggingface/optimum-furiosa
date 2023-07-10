# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import os
import tempfile
import unittest

import numpy as np
import requests
import torch
from parameterized import parameterized
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, PretrainedConfig, pipeline, set_seed

from optimum.furiosa import FuriosaAIModelForImageClassification
from optimum.furiosa.utils import FURIOSA_ENF_FILE_NAME
from optimum.utils import (
    logging,
)


SEED = 42

logger = logging.get_logger()

MODEL_DICT = {
    "mobilenet_v1": ["google/mobilenet_v1_0.75_192", {"pixel_values": [1, 3, 192, 192]}, {"logits": [1, 1001]}],
    "mobilenet_v2": [
        "hf-internal-testing/tiny-random-MobileNetV2Model",
        {"pixel_values": [1, 3, 32, 32]},
        {"logits": [1, 2]},
    ],
    "resnet": ["hf-internal-testing/tiny-random-resnet", {"pixel_values": [1, 3, 224, 224]}, {"logits": [1, 1000]}],
}


TENSOR_ALIAS_TO_TYPE = {
    "pt": torch.Tensor,
    "np": np.ndarray,
}


class FuriosaAIModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MODEL_ID = "mohitsha/furiosa-resnet-tiny-beans"

    def test_load_from_hub_and_save_model(self):
        preprocessor = AutoFeatureExtractor.from_pretrained(self.MODEL_ID)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        loaded_model = FuriosaAIModelForImageClassification.from_pretrained(self.MODEL_ID)
        self.assertIsInstance(loaded_model.config, PretrainedConfig)
        loaded_model_outputs = loaded_model(**inputs)

        with tempfile.TemporaryDirectory() as tmpdirname:
            loaded_model.save_pretrained(tmpdirname)
            del loaded_model
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(FURIOSA_ENF_FILE_NAME in folder_contents)
            model = FuriosaAIModelForImageClassification.from_pretrained(tmpdirname)

        outputs = model(**inputs)
        self.assertTrue(torch.equal(loaded_model_outputs.logits, outputs.logits))


class FuriosaAIModelForImageClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "mobilenet_v1",
        "mobilenet_v2",
        "resnet",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    FuriosaAIMODEL_CLASS = FuriosaAIModelForImageClassification
    TASK = "image-classification"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id, input_shape_dict, output_shape_dict = MODEL_DICT[model_arch]
        set_seed(SEED)
        fai_model = FuriosaAIModelForImageClassification.from_pretrained(
            model_id, export=True, input_shape_dict=input_shape_dict, output_shape_dict=output_shape_dict
        )
        self.assertIsInstance(fai_model.config, PretrainedConfig)
        transformers_model = AutoModelForImageClassification.from_pretrained(model_id)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        for input_type in ["pt", "np"]:
            inputs = preprocessor(images=image, return_tensors=input_type)
            fai_outputs = fai_model(**inputs)
            self.assertIn("logits", fai_outputs)
            self.assertIsInstance(fai_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(fai_outputs.logits), transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id, input_shape_dict, output_shape_dict = MODEL_DICT[model_arch]
        model = FuriosaAIModelForImageClassification.from_pretrained(
            model_id, export=True, input_shape_dict=input_shape_dict, output_shape_dict=output_shape_dict
        )
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("image-classification", model=model, feature_extractor=preprocessor)
        outputs = pipe("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))
        gc.collect()
