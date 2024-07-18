

#Text to image generator
#Change runtime type to GPU

!pip install transformers accelerate diffusers invisible_watermark safetensors streamlit

import streamlit as st
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io

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
    return image

# Streamlit app
st.title("Text to Image Generator")

prompt = st.text_input("Enter a text prompt:", value="A beautiful landscape with mountains and a river")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = generate_image(prompt)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.image(byte_im, caption='Generated Image', use_column_width=True)
