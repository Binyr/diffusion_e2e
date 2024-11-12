#!/bin/bash # accelerate launch training/train.py \
CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 accelerate launch --config_file 'configs/deep_speed/ds.yaml' \
  --num_machines 1 --num_processes 8 --main_process_port 23456 --mixed_precision 'fp16' training/train.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-2" \
  --modality "normals" \
  --noise_type "zeros" \
  --max_train_steps 20000 \
  --checkpointing_steps 4000 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --learning_rate 3e-05 \
  --lr_total_iter_length 20000 \
  --lr_exp_warmup_steps 100 \
  --mixed_precision "no" \
  --output_dir "outputs/stable_diffusion_e2e_ft_normals" \
  --enable_xformers_memory_efficient_attention \
  --dataloader_num_workers 8 \
  "$@"