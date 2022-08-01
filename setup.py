from setuptools import setup
import pathlib
import os

here = pathlib.Path(__file__).parent.resolve()

with open(os.path.join(here, 'requirements.txt')) as f:
    required = f.read().splitlines()

setup(
    name="oadg",
    version="0.0.2",
    description="Order Agnostic Autoregressive diffusion for geostatistical applications",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="Lukas Mosser",
    author_email="lukas.mosser@gmail.com",
    url="https://github.com/LukasMosser/oadg",
    packages=['oadg'],
    python_requires=">=3.8.0",
    install_requires=required,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
