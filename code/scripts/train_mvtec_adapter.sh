#!/bin/bash

# Adapter-only fine-tuning: only trains TextPromptAdapter (~263K params)
# Loads existing AnomalyGPT checkpoint and freezes everything else
# Expected training time: ~4-5 hours on 2x RTX 5090

deepspeed --include localhost:0,1 --master_port 28400 train_mvtec_adapter.py \
    --model openllama_peft \
    --stage 1 \
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth \
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/ \
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt \
    --anomalygpt_ckpt_path ./ckpt/train_mvtec/pytorch_model.pt \
    --max_tgt_len 512 \
    --data_path ../data/pandagpt4_visual_instruction_data.json \
    --image_root_path /workspace/Master-Thesis/data/images/ \
    --save_path ./ckpt/train_mvtec_adapter/ \
    --log_path ./ckpt/train_mvtec_adapter/log/ \
    --epochs 3
