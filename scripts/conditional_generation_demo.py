import gradio as gr
import torch
from diffusers.models import UNet2DModel
from huggingface_hub import hf_hub_download
from oadg.sampling import sample, make_conditional_paths_and_realization, initialize_empty_realizations_and_paths
from oadg.sampling import evaluate_entropy

image_size = 64
batch_size = 1
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
        idx_start, random_paths, realization = initialize_empty_realizations_and_paths(batch_size, image_size, image_size, device=device)
    else:
        img = (img > 0).astype(int)
        idx_start, random_paths, realization = make_conditional_paths_and_realization(img, batch_size=batch_size, device=device)

    img = sample(model, batch_size=batch_size, image_size=image_size,
                 realization=realization, idx_start=idx_start, random_paths=random_paths, device=device)
    img = img.reshape(image_size, image_size) * 255

    entropy = evaluate_entropy(model, batch_size=batch_size, image_size=image_size,
                               realization=realization, idx_start=idx_start, random_paths=random_paths, device=device)
    entropy = (entropy.reshape(image_size, image_size) * 255).astype(int)

    return entropy, img


img = gr.Image(image_mode="L", source="canvas", shape=(image_size, image_size), invert_colors=True, label="Drawing Canvas")
out_realization = gr.Image(image_mode="L", shape=(image_size, image_size), invert_colors=True, label="Sample Realization")
out_entropy = gr.Image(image_mode="L", shape=(image_size, image_size), invert_colors=True, label="Entropy of Drawn Data")

demo = gr.Interface(fn=sample_image, inputs=img, outputs=[out_entropy, out_realization],
                    title="Order Agnostic Autoregressive Diffusion Channels Demo",
                    description="""Sample conditional or unconditional images by drawing into the canvas.
                    Outputs a random sampled realization and predicted entropy under the trained model for the conditioning data.""")


demo.launch()
