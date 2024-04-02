#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import setuptools

with open("README.md") as f:
    readme = f.read()

setuptools.setup(
    name="apple_pytorch_speech_features",
    version="1.0.0",
    author="Arnav Kundu",
    author_email="a_kundu@apple.com",
    description="Pytorch Feature Extraction",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=["torch", "scipy", "numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
