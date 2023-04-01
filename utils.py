from torch.cuda import is_available as cuda_available
from torch.backends.mps import is_available as mps_available
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
import requests


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
