[![Test](https://github.com/huggingface/optimum-furiosa/actions/workflows/test.yml/badge.svg)](https://github.com/huggingface/optimum-furiosa/actions/workflows/test.yml)


# optimum-furiosa
Accelerated inference of 🤗 models using FuriosaAI NPU chips.

## Furiosa SDK setup
A Furiosa SDK environment needs to be enabled to use this library. Please refer to Furiosa's [Installation](https://furiosa-ai.github.io/docs/latest/en/software/installation.html) guide.

## Install
Optimum Furiosa is a fast-moving project, and you may want to install from source.

`pip install git+https://github.com/huggingface/optimum-furiosa.git`

### Installing in developer mode

If you are working on the `optimum-furiosa` code then you should use an editable install
by cloning and installing `optimum` and `optimum-furiosa`:

```
git clone https://github.com/huggingface/optimum
git clone https://github.com/huggingface/optimum-furiosa
pip install -e optimum -e optimum-furiosa
```

Now whenever you change the code, you'll be able to run with those changes instantly.


## How to use it?
To load a model and run inference with Furiosa NPU, you can just replace your `AutoModelForXxx` class with the corresponding `FuriosaAIModelForXxx` class. 

```diff
import requests
from PIL import Image

- from transformers import AutoModelForImageClassification
+ from optimum.furiosa import FuriosaAIModelForImageClassification
from transformers import AutoFeatureExtractor, pipeline

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model_id = "microsoft/resnet-50"
- model = AutoModelForImageClassification.from_pretrained(model_id)
+ model = FuriosaAIModelForImageClassification.from_pretrained(model_id, export=True, input_shape_dict={"pixel_values": [1, 3, 224, 224]}, output_shape_dict={"logits": [1, 1000]},)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
cls_pipe = pipeline("image-classification", model=model, feature_extractor=feature_extractor)
outputs = cls_pipe(image)
```

If you find any issue while using those, please open an issue or a pull request.
