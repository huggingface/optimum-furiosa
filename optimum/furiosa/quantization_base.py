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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union


logger = logging.getLogger(__name__)


class OptimumQuantizer(ABC):
    @classmethod
    def from_pretrained(
        cls,
        model_or_path: Union[str, Path],
        file_name: Optional[str] = None,
    ):
        """Overwrite this method in subclass to define how to load your model from pretrained"""
        raise NotImplementedError(
            "Overwrite this method in subclass to define how to load your model from pretrained for quantization"
        )

    @abstractmethod
    def quantize(self, save_dir: Union[str, Path], file_prefix: Optional[str] = None, **kwargs):
        """Overwrite this method in subclass to define how to quantize your model for quantization"""
        raise NotImplementedError(
            "Overwrite this method in subclass to define how to quantize your model for quantization"
        )
