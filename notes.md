# Things to try

- Improve finetuning prompt - generate prompts based on dataset & use those to generate class pics. Then train with those prompts
- Test DDIM scheduler instead of whatever is currently being used
- Keep improving dataset


# Done items

- Find out why all generated images are the wrong orientation - likely because source images are also the wrong orientation?
    - Make sure all images are correct orientation
    - Crop each image so it is square
    - Resize to 768x768
- Improve dataset, no 2 pics in same position/clothes/background
- Test training UNET on 8GB GPU and add pre-trained text encoder

