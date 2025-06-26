#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "E:/Data/Models(huggingface)/CogVideoX1-5-2B"
    --model_name "cogvideox1.5-t2v"  # ["cogvideox-t2v"]
    --model_type "t2v"
    --training_type "lora"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "E:/PythonLearn/work/SSH_Connect/Autodl/under2postgraudate/Video-Generation-field/CogVideoX/CogVideo/output_dir"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "E:/Data/datasets/Video_Datasets/Disney-VideoGeneration-Dataset"    # root path
    --caption_column "prompt.txt"                     # caption txt path
    --video_column "videos.txt"                       # video txt path
    --train_resolution "81x768x1360"  # (frames x height x width), frames should be 8N+1
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 10 # number of training epochs
    --seed 42 # random seed
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "fp16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 10 # save checkpoint every x steps
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
#    --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation false  # ["true", "false"]  # 是否验证
    --validation_dir "/absolute/path/to/your/validation_set"  # 验证集路径
    --validation_steps 20                   # 每20个batch验证一次
    --validation_prompts "prompts.txt"      # 验证集prompts
    --gen_fps 16
)

# Combine all arguments and launch training
accelerate launch train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"


# accelerate launch train.py --model_path "THUDM/CogVideoX1.5-2B" --model_name "cogvideox1.5-t2v" --model_type "t2v" --training_type "lora" --output_dir "E:/PythonLearn/work/SSH_Connect/Autodl/under2postgraudate/Video-Generation-field/CogVideoX/CogVideo/output_dir" --report_to "tensorboard" --data_root "E:/Data/datasets/Video_Datasets/Disney-VideoGeneration-Dataset" --caption_column "prompt.txt" --video_column "videos.txt" --train_resolution "81x768x1360" --train_epochs 10 --seed 42 --batch_size 1 --gradient_accumulation_steps 1 --mixed_precision "bf16" --num_workers 8 --pin_memory True --nccl_timeout 1800 --checkpointing_steps 10 --checkpointing_limit 2 --do_validation false --validation_dir "/absolute/path/to/your/validation_set" --validation_steps 20 --validation_prompts "prompts.txt" --gen_fps 16
