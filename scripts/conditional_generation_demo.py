import gradio as gr
import torch
from diffusers.models import UNet2DModel
from huggingface_hub import hf_hub_url, cached_download


config_file_url = hf_hub_url(repo_id="porestar/oadg_channels_64", filename="model.pt", revision="main")
cached_download(config_file_url)

model = UNet2DModel(
    sample_size=64,
    in_channels=2,
    out_channels=2,
    layers_per_block=2,
    block_out_channels=(64, 64, 128, 128),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

model.load_state_dict(torch.load("./oadg_channels_64/model.pt"))


def classify_image(inp):
    return {"lol": 0}


img = gr.Image(image_mode="L", source="canvas", shape=(32, 32), invert_colors=False)
label = gr.Label(num_top_classes=3)

demo = gr.Interface(
    fn=classify_image, inputs=img, outputs=label, interpretation="default"
)

demo.launch()
