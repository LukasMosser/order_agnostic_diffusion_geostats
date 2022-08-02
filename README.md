# Order-Agnostic Autoregressive Diffusion Models for Geostatistical Applications


## Description


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
| Train MNIST and Channel Models with Google Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasMosser/order_agnostic_diffusion_geostats/blob/main/notebooks/train_oadg_models_colab_hf_wb.ipynb) |
|                                                  |                                                                                                                                                                                                                              |
|                                                  |                                                                                                                                                                                                                              |


## Acknowledgments
I would like to thank Emiel Hoogeboom [[Website](https://ehoogeboom.github.io/)] [[Twitter](https://twitter.com/emiel_hoogeboom)]
for clarifications via email on understanding the methodology of the paper.

There exists an excellent official implementation by Emiel and his co-authors here: [Official Implementation Github](https://github.com/google-research/google-research/tree/master/autoregressive_diffusion)

Furthermore, thanks to Eric Laloy and colleagues for making their channel training image available online.


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