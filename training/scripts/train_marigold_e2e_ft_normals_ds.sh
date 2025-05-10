#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file 'configs/deep_speed/ds_acc4.yaml' \
  --num_machines 1 --num_processes 3 --main_process_port 22456 training/train.py \
  --pretrained_model_name_or_path "GonzaloMG/marigold-normals" \
  --modality "normals" \
  --noise_type "zeros" \
  --max_train_steps 20000 \
  --checkpointing_steps 20000 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --learning_rate 3e-05 \
  --lr_total_iter_length 20000 \
  --lr_exp_warmup_steps 100 \
  --mixed_precision "fp16" \
  --output_dir "model-finetuned/marigold_e2e_ft_normals" \
  --enable_xformers_memory_efficient_attention \
  "$@"