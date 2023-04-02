#!/bin/bash

MODEL_NAME="stabilityai/stable-diffusion-2-1"
FINETUNE_MODEL_NAME="shane5"
#MODEL_NAME = "CompVis/stable-diffusion-v1-4"
TOKEN_NAME="shane"
INSTANCE_DIR="./models/$TOKEN_NAME/data/$TOKEN_NAME"
CLASS_NAME="man"
# TODO: Test if we can find a better prompt by generating prompt from images & using that to ensure generated images are as close as possible to uploaded images
CLASS_PROMPT="a photo of a man"
INSTANCE_PROMPT="a photo of a $TOKEN_NAME man"
CLASS_DIR="./models/$TOKEN_NAME/data/$CLASS_NAME"
OUTPUT_DIR="./models/$TOKEN_NAME/stable_diffusion_weights/$FINETUNE_MODEL_NAME"
PRIOR_LOSS_WEIGHT=1.0
RESOLUTION=512
TRAIN_BATCH_SIZE=1
SAMPLE_BATCH_SIZE=1
GRAD_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-6
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0
NUM_CLASS_IMAGES=200
MAX_TRAIN_STEPS=1200 #200
# Checkpointing seems to cause failure so set to never trigger
CHECKPOINTING_STEPS=9999

mkdir -p $INSTANCE_DIR
mkdir -p $OUTPUT_DIR

# TODO: Test DDIM scheduler
accelerate launch train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation \
    --prior_loss_weight=$PRIOR_LOSS_WEIGHT \
    --instance_prompt="$INSTANCE_PROMPT" \
    --class_prompt="$CLASS_PROMPT" \
    --resolution=$RESOLUTION \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
    --learning_rate=$LEARNING_RATE \
    --lr_scheduler=$LR_SCHEDULER \
    --lr_warmup_steps=$LR_WARMUP_STEPS \
    --num_class_images=$NUM_CLASS_IMAGES \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --sample_batch_size=$SAMPLE_BATCH_SIZE \
    --checkpointing_steps=$CHECKPOINTING_STEPS \
    --mixed_precision='fp16' \
    --enable_xformers_memory_efficient_attention \
    --set_grads_to_none \
    --use_8bit_adam \
    #--train_text_encoder
