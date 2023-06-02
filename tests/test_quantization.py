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

import os
import tempfile
import unittest
from functools import partial
from pathlib import Path

import requests
from parameterized import parameterized
from PIL import Image
from transformers import AutoFeatureExtractor

from optimum.furiosa import (
    AutoCalibrationConfig,
    FuriosaAIConfig,
    FuriosaAIModelForImageClassification,
    FuriosaAIQuantizer,
    QuantizationConfig,
)
from optimum.furiosa.utils import export_model_to_onnx


class FuriosaAIQuantizationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ((FuriosaAIModelForImageClassification, "fxmarty/resnet-tiny-beans"),)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_quantization(self, model_cls, model_name):
        qconfig = QuantizationConfig()

        def preprocess_fn(ex, feature_extractor):
            return feature_extractor(ex["image"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            export_model_to_onnx(
                model_name,
                save_dir=tmp_dir,
                input_shape_dict={"pixel_values": [1, 3, 224, 224]},
                output_shape_dict={"logits": [1, 3]},
                file_name="model.onnx",
            )

            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

            quantizer = FuriosaAIQuantizer.from_pretrained(tmp_dir, file_name="model.onnx")

            calibration_dataset = quantizer.get_calibration_dataset(
                "beans",
                preprocess_function=partial(preprocess_fn, feature_extractor=feature_extractor),
                num_samples=10,
                dataset_split="train",
            )

            calibration_config = AutoCalibrationConfig.mse_asym(calibration_dataset)
            ranges = quantizer.fit(
                dataset=calibration_dataset,
                calibration_config=calibration_config,
            )

            quantizer.quantize(
                save_dir=output_dir,
                calibration_tensors_range=ranges,
                quantization_config=qconfig,
            )

            expected_fai_config = FuriosaAIConfig(quantization=qconfig, calibration=calibration_config)
            fai_config = FuriosaAIConfig.from_pretrained(tmp_dir)
            # Verify the FuriosaAIConfig was correctly created and saved
            self.assertEqual(fai_config.to_dict(), expected_fai_config.to_dict())

            assert os.path.isfile(output_dir.joinpath("model_quantized.dfg")) is True

            fai_model_quantized = model_cls(Path(output_dir) / "model_quantized.dfg")

            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            image = Image.open(requests.get(url, stream=True).raw)
            inputs = feature_extractor(images=image, return_tensors="np")

            fai_outputs = fai_model_quantized(**inputs)
            self.assertIn("logits", fai_outputs)
