<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Image classification

The script [`run_image_classification.py`](https://github.com/huggingface/optimum-furiosa/blob/main/examples/quantization/image_classification/run_image_classification.py) allows us to apply different quantization using [FuriosaAI SDK](https://furiosa-ai.github.io/docs/latest/en/software/quantization.html) for image classification tasks.

The following example applies quantization on a Resnet model fine-tuned on the beans classification dataset.

```bash
python run_image_classification.py \
    --model_name_or_path eugenecamus/resnet-50-base-beans-demo \
    --dataset_name beans \
    --do_eval \
    --output_dir /tmp/image_classification_resnet_beans
```
