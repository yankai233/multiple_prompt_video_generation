#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Checkpointing Configuration
CHECKPOINT_ARGS=(
     --resume_from_checkpoint ""      # XXX/model.pth
)
# Output Configuration
OUTPUT_ARGS=(
    --log_dir "../../excluded_dir/output_dir/logs/train_feature_extraction"
    --output_dir "../../excluded_dir/output_dir/logs/train_feature_extraction/ckpt"
)

# configs Configuration
CONFIGS_ARGS=(
    --configs "../../configs/Feature_Extraction_Module_T5.yaml"
)

# Training Configuration
TRAIN_ARGS=(
    --device "cuda"
    --mixed_precision "fp16"
)


# Combine all arguments and launch training
python train_feature_extraction.py \
    "${CHECKPOINT_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${CONFIGS_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \

# accelerate launch train.py --model_path "THUDM/CogVideoX1.5-2B" --model_name "cogvideox1.5-t2v" --model_type "t2v" --training_type "lora" --output_dir "E:/PythonLearn/work/SSH_Connect/Autodl/under2postgraudate/Video-Generation-field/CogVideoX/CogVideo/output_dir" --report_to "tensorboard" --data_root "E:/Data/datasets/Video_Datasets/Disney-VideoGeneration-Dataset" --caption_column "prompt.txt" --video_column "videos.txt" --train_resolution "81x768x1360" --train_epochs 10 --seed 42 --batch_size 1 --gradient_accumulation_steps 1 --mixed_precision "bf16" --num_workers 8 --pin_memory True --nccl_timeout 1800 --checkpointing_steps 10 --checkpointing_limit 2 --do_validation false --validation_dir "/absolute/path/to/your/validation_set" --validation_steps 20 --validation_prompts "prompts.txt" --gen_fps 16
