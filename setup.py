import re

from setuptools import find_namespace_packages, setup


# Ensure we match the version set in optimum/furiosa/version.py
try:
    filepath = "optimum/furiosa/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)

INSTALL_REQUIRE = [
    "optimum>=1.8.0",
    "transformers>=4.20.0",
    "datasets>=1.4.0",
    "furiosa-optimizer",
    "furiosa-quantizer",
    "furiosa-sdk",
    "onnx>=1.12.0",
    "sentencepiece",
    "scipy",
]

TESTS_REQUIRE = ["pytest", "parameterized", "Pillow", "evaluate", "diffusers", "py-cpuinfo"]

QUALITY_REQUIRE = ["black~=23.1", "ruff>=0.0.241"]

EXTRA_REQUIRE = {
    "testing": [
        "filelock",
        "GitPython",
        "parameterized",
        "psutil",
        "pytest",
        "pytest-pythonpath",
        "pytest-xdist",
        "librosa",
        "soundfile",
    ],
    "quality": QUALITY_REQUIRE,
}

setup(
    name="optimum-furiosa",
    version=__version__,
    description="Optimum Library is an extension of the Hugging Face Transformers library, providing a framework to "
    "integrate third-party libraries from Hardware Partners and interface with their specific "
    "functionality.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, quantization, pruning, knowledge distillation, optimization, training",
    url="https://huggingface.co/hardware",
    author="HuggingFace Inc. Special Ops Team",
    author_email="hardware@huggingface.co",
    license="Apache",
    packages=find_namespace_packages(include=["optimum*"]),
    install_requires=INSTALL_REQUIRE,
    extras_require=EXTRA_REQUIRE,
    include_package_data=True,
    zip_safe=False,
    entry_points={"console_scripts": ["optimum-cli=optimum.commands.optimum_cli:main"]},
)
