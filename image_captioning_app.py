import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):

    #convert numpy array to PIL image and convert to RGB
    raw_image = Image.fromarray(input_immage).convert('RBG')

    #process image

    inputs = processor(raw_image, return_tensors = "pt")

    #generate caption

    out = model.generate(**inputs, max_length=50)

    #Decode generated tokens

    caption = processor.decode(out[0], skip_special_tokens = True)

    return caption

iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)

iface.launch()