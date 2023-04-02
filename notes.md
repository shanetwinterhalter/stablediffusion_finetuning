# Things to try

- Improve finetuning prompt - generate prompts based on dataset & use those to generate class pics. Then train with those prompts
- Try and get text encoder training working on 8GB GPU
- Try and get good results on 8GB GPU
- Test DDIM scheduler instead of whatever is currently being used


# Done items

- Find out why all generated images are the wrong orientation - likely because source images are also the wrong orientation?
    - Make sure all images are correct orientation
    - Crop each image so it is square
    - Resize to 768x768
- Improve dataset, no 2 pics in same position/clothes/background
