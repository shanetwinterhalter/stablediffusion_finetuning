from diffusers import StableDiffusionPipeline, DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
from torch.cuda import is_available as cuda_available
from torch.backends.mps import is_available as mps_available
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
import requests
import torch


def get_device():
    # Set GPU device
    if cuda_available():
        device = 'cuda'
    elif mps_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device


def plot_img(image):
    imgplot = plt.imshow(image)
    plt.axis('off')
    imgplot.axes.get_xaxis().set_visible(False)
    imgplot.axes.get_yaxis().set_visible(False)


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def image_grid(imgs, rows=2, cols=2):                                                                                                                                                                                                         
    w, h = imgs[0].size                                                                                                                                                                                                                       
    grid = Image.new('RGB', size=(cols*w, rows*h))                                                                                                                                                                                            
                                                                                                                                                                                                                                              
    for i, img in enumerate(imgs):                                                                                                                                                                                                            
        grid.paste(img, box=(i%cols*w, i//cols*h))                                                                                                                                                                                            
    return grid


def gen_images(no_imgs, prompt, n_prompt="", num_inference_steps=75, guidance_scale=20, seed=None, weights_folder_name=None, model_id=None, text_encoder=None, unet=None):
    weights_root_folder = "./models/shane/stable_diffusion_weights"
    if weights_folder_name is not None:
        finetuned_model = weights_root_folder + "/" + weights_folder_name
        pipe = StableDiffusionPipeline.from_pretrained(finetuned_model, torch_dtype=torch.float16).to("cuda")
    else:
        unet = UNet2DConditionModel.from_pretrained(weights_root_folder+"/"+unet+"/unet", torch_dtype=torch.float16)
        text_encoder = CLIPTextModel.from_pretrained(weights_root_folder+"/"+text_encoder+"/text_encoder", torch_dtype=torch.float16)
        pipe = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, torch_dtype=torch.float16).to("cuda")
    
    generator = torch.Generator("cuda")
    if seed is not None:
        generator.manual_seed(seed)
    image = [0] * no_imgs
    for i in range(no_imgs):
        image[i] = pipe(prompt, negative_prompt=n_prompt, generator=generator, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return image