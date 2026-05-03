"""Train visual_adapter + text_adapter using pure PyTorch DDP (no DeepSpeed).

Usage:
    torchrun --nproc_per_node=2 train_adapter_ddp.py \
        --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth \
        --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/ \
        --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt \
        --anomalygpt_ckpt_path ./ckpt/train_mvtec/pytorch_model.pt \
        --save_path ./ckpt/train_adapter_ddp/ \
        --epochs 3 \
        --max_steps 0
"""

import os, sys, time, json, argparse, random, logging, datetime, types
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Minimal imports needed
from config import load_config
from datasets import load_mvtec_dataset, load_sft_dataset
from model import OpenLLAMAPEFTModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--imagebind_ckpt_path', type=str, required=True)
    p.add_argument('--vicuna_ckpt_path', type=str, required=True)
    p.add_argument('--delta_ckpt_path', type=str, required=True)
    p.add_argument('--anomalygpt_ckpt_path', type=str, default='')
    p.add_argument('--save_path', type=str, default='./ckpt/train_adapter_ddp/')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--max_steps', type=int, default=0)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--batch_size', type=int, default=3)
    p.add_argument('--grad_accum', type=int, default=2)
    p.add_argument('--max_tgt_len', type=int, default=512)
    p.add_argument('--save_every', type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()

    # DDP init
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    is_main = (local_rank == 0)

    # Build config (reuse existing config system)
    cfg = {
        'model': 'openllama_peft',
        'stage': 1,
        'root_dir': '../',
        'mode': 'train',
        'local_rank': local_rank,
        'world_size': world_size,
        'imagebind_ckpt_path': args.imagebind_ckpt_path,
        'vicuna_ckpt_path': args.vicuna_ckpt_path,
        'delta_ckpt_path': args.delta_ckpt_path,
        'max_tgt_len': args.max_tgt_len,
        'data_path': '../data/pandagpt4_visual_instruction_data.json',
        'image_root_path': '/workspace/Master-Thesis/data/images/',
        'train_micro_batch_size_per_gpu': args.batch_size,
        'layers': [7, 15, 23, 31],
    }
    yaml_config = load_config(cfg)
    cfg.update(yaml_config)

    # Mock dschf for dataset loader compatibility
    class _MockDschf:
        config = {"train_micro_batch_size_per_gpu": args.batch_size}
    cfg["dschf"] = _MockDschf()

    # Load datasets
    train_data, train_iter, sampler = load_mvtec_dataset(cfg)
    train_data_sft, train_iter_sft, _ = load_sft_dataset(cfg)

    # Patch LLM to load in bf16 (saves ~15GB per GPU)
    import transformers
    _orig_from_pretrained = transformers.LlamaForCausalLM.from_pretrained
    @classmethod  
    def _bf16_from_pretrained(cls, *a, **kw):
        kw.setdefault("torch_dtype", torch.bfloat16)
        kw.setdefault("low_cpu_mem_usage", True)
        return _orig_from_pretrained.__func__(cls, *a, **kw)
    transformers.LlamaForCausalLM.from_pretrained = _bf16_from_pretrained

    # Build model
    model = OpenLLAMAPEFTModel(**cfg)

    # Load checkpoints
    delta_ckpt = torch.load(args.delta_ckpt_path, map_location='cpu')
    model.load_state_dict(delta_ckpt, strict=False)
    del delta_ckpt

    if args.anomalygpt_ckpt_path:
        anomaly_ckpt = torch.load(args.anomalygpt_ckpt_path, map_location='cpu')
        model.load_state_dict(anomaly_ckpt, strict=False)
        if is_main:
            print(f'[!] Loaded AnomalyGPT checkpoint from {args.anomalygpt_ckpt_path}')
        del anomaly_ckpt

    # Freeze all, then unfreeze adapters
    for p in model.parameters():
        p.requires_grad = False

    for p in model.text_adapter.parameters():
        p.requires_grad = True
    for p in model.visual_adapter.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f'[Adapter DDP] trainable: {trainable:,} / {total:,} ({trainable/total*100:.4f}%)')
        va_params = sum(p.numel() for p in model.visual_adapter.parameters())
        ta_params = sum(p.numel() for p in model.text_adapter.parameters())
        print(f'  visual_adapter: {va_params:,}, text_adapter: {ta_params:,}')

    # Move to device in bf16 to fit in 32GB
    model = model.bfloat16().to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.001)

    # Training loop
    os.makedirs(args.save_path, exist_ok=True)
    steps_per_epoch = len(train_data) // (args.batch_size * world_size)
    total_steps = 2 * args.epochs * steps_per_epoch  # x2 for mvtec + sft alternating

    if is_main:
        print(f'[!] Starting training: {args.epochs} epochs, ~{steps_per_epoch} steps/epoch, total ~{total_steps} steps')
        pbar = tqdm(total=total_steps if args.max_steps == 0 else args.max_steps * 2)
    
    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        for batch, batch_sft in zip(train_iter, train_iter_sft):
            # MVTec batch
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, acc = model(batch)
            loss = loss / args.grad_accum
            loss.backward()

            # SFT batch
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss_sft, acc_sft = model(batch_sft)
            loss_sft = loss_sft / args.grad_accum
            loss_sft.backward()

            global_step += 1

            if global_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if is_main:
                pbar.set_description(f'loss: {loss.item()*args.grad_accum:.4f}/{loss_sft.item()*args.grad_accum:.4f} acc: {acc*100:.1f}/{acc_sft*100:.1f}')
                pbar.update(2)

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            if global_step % args.save_every == 0 and is_main:
                save_adapter(model, args.save_path, global_step)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Final save
    if is_main:
        save_adapter(model, args.save_path, global_step)
        pbar.close()
        print(f'[!] Training complete. {global_step} steps.')

    dist.destroy_process_group()


def save_adapter(model, path, step):
    """Save only adapter weights."""
    from collections import OrderedDict
    m = model.module if hasattr(model, 'module') else model
    ckpt = OrderedDict()
    for k, v in m.named_parameters():
        if v.requires_grad:
            ckpt[k] = v.data.cpu().clone()
    
    # Also check visual_adapter specifically
    va_keys = [k for k in ckpt if 'visual_adapter' in k]
    ta_keys = [k for k in ckpt if 'text_adapter' in k]
    
    save_file = f'{path}/adapter_model_step{step}.pt'
    torch.save(ckpt, save_file)
    print(f'[SAVE] {save_file} | VA keys: {len(va_keys)}, TA keys: {len(ta_keys)}, total: {len(ckpt)}')
    
    # Verify VA weights changed from init
    for k in va_keys:
        if 'up_proj.weight' in k:
            nonzero = (ckpt[k] != 0).sum().item()
            total = ckpt[k].numel()
            print(f'  [CHECK] {k}: nonzero={nonzero}/{total}')
            break


if __name__ == '__main__':
    main()
