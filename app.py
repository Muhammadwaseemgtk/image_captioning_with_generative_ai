import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import gradio as gr

# loading the pretrained processor and model 
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def image_captioner(img):
    # loading the image for further processing
    image = Image.fromarray(img).convert("RGB")

    # preparing the inputs in the form of pytorch tensors
    inputs = processor(images = image, return_tensors='pt')

    # the resulting output from the model
    outputs = model.generate(**inputs, max_length=50)

    # decoding the output from the model in human readable text
    caption = processor.decode(outputs[0], skip_special_tokens = True)

    return caption


iface = gr.Interface(
    fn = image_captioner,
    inputs = gr.Image(),
    outputs= 'text',
    title = "Image Captioning App",
    description = "Upload the image to get the caption for it with the help of AI model"
)

iface.launch()