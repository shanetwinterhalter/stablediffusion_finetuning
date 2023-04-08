from diffusers import StableDiffusionPipeline, DiffusionPipeline
from diffusers import UNet2DConditionModel, StableDiffusionUpscalePipeline
from transformers import CLIPTextModel
from torch.cuda import is_available as cuda_available
from torch.backends.mps import is_available as mps_available
from PIL import Image
from io import BytesIO
from hashlib import md5

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


def image_grid(imgs):
    cols = min(5, len(imgs))
    rows = len(imgs) // cols + 1 if len(imgs) % cols else len(imgs) // cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def gen_images(no_imgs, prompt, n_prompt="", num_inference_steps=15,
               guidance_scale=20, seed=None, weights_folder_name=None,
               model_id=None, text_encoder=None, unet=None, device="cuda"):
    weights_root_folder = "./models/shane/stable_diffusion_weights"
    if weights_folder_name is not None:
        finetuned_model = weights_root_folder + "/" + weights_folder_name
        pipe = StableDiffusionPipeline.from_pretrained(
            finetuned_model, torch_dtype=torch.float16).to("cuda")
    else:
        unet = UNet2DConditionModel.from_pretrained(
            weights_root_folder+"/"+unet+"/unet", torch_dtype=torch.float16)
        text_encoder = CLIPTextModel.from_pretrained(
            weights_root_folder+"/"+text_encoder+"/text_encoder",
            torch_dtype=torch.float16)
        pipe = DiffusionPipeline.from_pretrained(
            model_id, unet=unet, text_encoder=text_encoder,
            torch_dtype=torch.float16).to(device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device).manual_seed(seed)
    image = [0] * no_imgs
    for i in range(no_imgs):
        image[i] = pipe(prompt, negative_prompt=n_prompt, generator=generator,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale).images[0]
    return image


# Only cuda supported for upscaling images
def upscale_images(images, prompt, n_prompt="", num_inference_steps=15,
                   model_id="stabilityai/stable-diffusion-x4-upscaler",
                   device="cuda"):
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, torch_dtype=torch.float16).to(device)
    pipe.enable_xformers_memory_efficient_attention()

    return [
        pipe(prompt=prompt,
             negative_prompt=n_prompt,
             num_inference_steps=num_inference_steps,
             image=img.resize((256, 256))).images[0]
        for img in images
    ]


def save_images(images, folder):
    for img in images:
        img.save(folder + "/" + md5(img.tobytes()).hexdigest() + ".png")
