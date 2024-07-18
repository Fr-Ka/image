

#Text to image generator
#Change runtime type to GPU

!pip install transformers accelerate diffusers invisible_watermark safetensors

from diffusers import DiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load the model
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.to("cuda")

#Define a function to generate and display images based on text prompts
def generate_and_display_image(prompt):
    # Generate the image
    images = pipe(prompt=prompt).images[0]

    # Display the image
    plt.imshow(images)
    plt.axis('off')  # Hide the axis
    plt.show()

