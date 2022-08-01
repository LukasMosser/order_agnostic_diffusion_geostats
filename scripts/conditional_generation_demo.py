import gradio as gr
import torch
from diffusers.models import UNet2DModel
from huggingface_hub import hf_hub_download
from oadg.sampling import sample, make_conditional_paths_and_realization, initalize_empty_realizations_and_paths

image_size = 32
batch_size = 16
device = 'cpu'

path = hf_hub_download(repo_id="porestar/oadg_channels_64", filename="model.pt")

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

model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

model = model.to(device)


def sample_image(img):
    if img is None:
        t_range_start, sigma_conditioned, realization = initalize_empty_realizations_and_paths(batch_size, image_size, image_size, device=device)
    else:
        t_range_start, sigma_conditioned, realization = make_conditional_paths_and_realization(img, device=device)

    img = sample(model, batch_size=batch_size, image_size=image_size,
                 realization=realization, t_range_start=t_range_start, sigma_conditioned=sigma_conditioned, device=device)
    img = img.reshape(4*image_size, 4*image_size)*255
    return img


img = gr.Image(image_mode="L", source="canvas", shape=(image_size, image_size), invert_colors=True)
out = gr.Image(image_mode="L", shape=(image_size, image_size), invert_colors=True)

demo = gr.Interface(fn=sample_image, inputs=img, outputs=out)

demo.launch()
