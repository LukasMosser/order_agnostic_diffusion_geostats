# Order-Agnostic Autoregressive Diffusion Models for Geostatistical Applications


## Introduction

This is a short introduction to the reasoning behind this work.   
The introductory notebook provides a full length description and implementation of the methods.  
*Introductory Notebook* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasMosser/order_agnostic_diffusion_geostats/blob/main/notebooks/introduction_and_walkthrough.ipynb)

#### Geostatistical Modeling
Geostatistical models are critical for applications such as mineral resource estimation, 
storage modeling of CO2, and many other geospatial tasks.

Sequential indicator simulation (SIS) (See [Gomez-Hernandez&Srivastasa, 2021](https://link.springer.com/article/10.1007/s11004-021-09926-0) for an excellent review) is an autoregressive model for categorical 
properties that has found widespread adoption due to its flexibility and ability to incorporate existing data.

These features make SIS able to generate stochastic realizations honoring existing observations.

#### (Deep) Autoregressive Generative Models

Autoregressive models (See [Kevin Patrick Murphy](Kevin Patrick Murphy)'s new book [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) chapter 22 for an introduction) using (deep) neural networks have shown a large potential to represent complex data distributions.
In many cases, so-called causal convolutions require data to be generated in very specific patterns (top down, left to right)
which does not allow for sampling of realizations with conditioning data at various locations.

A recent method called order-agnostic [autoregressive diffusion models by Hoogeboom et al.](https://arxiv.org/abs/2110.02037) allows for arbitrary ordering
of the generation steps. 

#### New possibilities for geostatistical modeling with (deep) generative models
This opens up the ability to incorporate spatially distributed conditioning data
to generate geostatistical realizations that honor data. 

Furthermore, the model parameterizes a categorical distribution which allows us to directly compute the 
entropy i.e. uncertainty distribution given the conditioning data.

I hope that these connections between (deep) autoregressive models and sequential geostatistical methods
also interest the reader, and spurns further research at the intersection between the fields of geostatistics and machine learning.

## Disclaimer and a note on publishing

Right now this is a few notebooks, some code, and some models.
I do not have funding necessary to publish in a proper journal, but may consider publishing through [Curvenote](https://curvenote.com/).  

Please if you find this useful or interesting do consider referencing the repository anyway.

As such this article is not peer reviewed, but I am happy to receive comments and will acknowledge these.

Models have been trained on my own cost via [Google Colab Pro+](https://colab.research.google.com/).
Models are hosted on :hug-face: [Huggingface](https://huggingface.co/) Model Repositories and Monitoring was done with [Weights&Biases](https://wandb.ai/site).

## Installation
Installation can be performed via pip:
```bash
pip install git+https://github.com/LukasMosser/order_agnostic_diffusion_geostats@main
```


## Models
| Model Description            | Huggingface Model Hub Link                                                                       | Weights & Biases Logging Run                                                                                                                                     |
|------------------------------|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Channels Dataset at 64x64 px | [Huggingface Model Hub Link](https://huggingface.co/porestar/oadg_mnist_32/tree/main)            | [Weights & Biases Monitoring](https://wandb.ai/lukas-mosser/order-agnostic-autoregressive-diffusion-channels/runs/2swdnaup/overview?workspace=user-lukas-mosser) |
 | MNIST Dataset at 32x32 px    | [Huggingface Model Hub Link](https://huggingface.co/porestar/oadg_mnist_32/tree/main)            | [Weights & Biases Monitoring](https://wandb.ai/lukas-mosser/order-agnostic-autoregressive-diffusion-mnist/runs/xwwwqpgp?workspace=user-lukas-mosser)             |


## Notebooks
| Description                                      | Google Colab Link                                                                                                                                                                                                            |
|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Introduction and Walkthrough (Start Here)        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasMosser/order_agnostic_diffusion_geostats/blob/main/notebooks/introduction_and_walkthrough.ipynb)  |
| Train MNIST and Channel Models with Google Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasMosser/order_agnostic_diffusion_geostats/blob/main/notebooks/train_oadg_models_colab_hf_wb.ipynb) |
|                                                  |                                                                                                                                                                                                                              |
|                                                  |                                                                                                                                                                                                                              |


## Acknowledgments
I would like to thank Emiel Hoogeboom [[Website](https://ehoogeboom.github.io/)] [[Twitter](https://twitter.com/emiel_hoogeboom)]
for clarifications via email on understanding the methodology of the ARDM approach.

There exists an excellent official implementation by Emiel and his co-authors here: [Official Implementation Github](https://github.com/google-research/google-research/tree/master/autoregressive_diffusion)

Furthermore, thanks to [Eric Laloy](https://scholar.google.com/citations?user=QrvhkvQAAAAJ&hl=en) and colleagues for making their [channel training image](https://github.com/elaloy/gan_for_gradient_based_inv) available online.

Finally, a huge thanks to the ML community for making available libraries such as :hug-face: huggingface hub, diffusers, accelerate, pytorch, and many more
without this work couldn't exist.

Please consider citing their work and providing proper attribution.

## Reference
If you've found this useful please consider referencing this repository in your own work
```
@software{order_agnostic_diffusion_geostats,
  author       = {Lukas Mosser},
  title        = {{Order-Agnostic Autoregressive Diffusion Models for Geostatistical Applications}},
  month        = aug,
  year         = 2022,
  url          = {https://github.com/LukasMosser/order_agnostic_diffusion_geostats}
}
```

## License
```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
```