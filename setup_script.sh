#!/bin/bash

echo "Installing diffusers from source"
wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth.py

echo "Installing other requirements"
pip install -U pip
pip install -r requirements.txt
pip install -U numexpr

echo "Remember to run accelerate config to finish setting up the environment"