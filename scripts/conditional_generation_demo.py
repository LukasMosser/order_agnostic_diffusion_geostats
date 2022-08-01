import gradio as gr
from diffusers.models import UNet2DModel

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

pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")

def predict(image):
  predictions = pipeline(image)
  return {p["label"]: p["score"] for p in predictions}

gr.Interface(
    predict,
    inputs=gr.inputs.Image(label="Upload hot dog candidate", type="filepath"),
    outputs=gr.outputs.Label(num_top_classes=2),
    title="Hot Dog? Or Not?",
).launch()

def classify_image(inp):
    print(img.shape)
    print(img.value)
    plt.imshow(img)
    plt.show()
    return {"lol": 0}


img = gr.Image(image_mode="L", source="canvas", shape=(32, 32), invert_colors=False)
label = gr.Label(num_top_classes=3)

demo = gr.Interface(
    fn=classify_image, inputs=img, outputs=label, interpretation="default"
)

demo.launch()

#