"""Test different anomaly map computation strategies.

Variants:
  A) Baseline (current): max cosine sim, equal layer weights
  B) Top-k soft matching: average top-k similar patches instead of just max
  C) Weighted layers: different fixed weight schemes
  D) Exponential distance: use exp(-d) instead of 1-sim for sharper maps
  E) Combined: top-k + weighted layers + exp distance

Usage:
    python3 test_anomaly_map_variants.py 2>&1 | tee /workspace/Master-Thesis/logs/anomaly_map_variants.log
"""

import os, sys, argparse, glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from header import *
from model import OpenLLAMAPEFTModel

# Import helpers from v6 test
# We only need select_spread_references; describles not needed for this test
def select_spread_references(all_paths, n_refs, model_ref):
    if len(all_paths) <= n_refs:
        return all_paths
    from model.ImageBind import data
    from model.ImageBind.models.imagebind_model import ModalityType
    all_features = []
    for i in range(0, len(all_paths), 16):
        batch_paths = all_paths[i:i+16]
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(batch_paths, "cuda")}
        inputs = {key: inputs[key].to(model_ref.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = model_ref.visual_encoder(inputs)
            cls_features = embeddings["vision"][0]
        all_features.append(cls_features)
    all_features = torch.cat(all_features, dim=0)
    all_features = F.normalize(all_features, dim=-1)
    mean_feat = F.normalize(all_features.mean(dim=0, keepdim=True), dim=-1)
    dists_to_mean = 1 - (all_features @ mean_feat.T).squeeze()
    first_idx = dists_to_mean.argmin().item()
    selected = [first_idx]
    for _ in range(n_refs - 1):
        selected_feats = all_features[selected]
        sims = all_features @ selected_feats.T
        max_sims, _ = sims.max(dim=1)
        for idx in selected:
            max_sims[idx] = 999
        next_idx = max_sims.argmin().item()
        selected.append(next_idx)
    selected_paths = [all_paths[i] for i in selected]
    print(f"    Selected refs: {[os.path.basename(p) for p in selected_paths]}")
    return selected_paths

ALL_CLASS_NAMES = ["bottle", "cable", "capsule", "carpet", "grid",
                   "hazelnut", "leather", "metal_nut", "pill", "screw",
                   "tile", "toothbrush", "transistor", "wood", "zipper"]


# ============ VARIANT IMPLEMENTATIONS ============

def compute_baseline(query_pt, normal_pt):
    """A) Current: max cosine sim, equal layer average"""
    sims = []
    for i in range(len(query_pt)):
        q = F.normalize(query_pt[i], dim=-1)
        n = F.normalize(normal_pt[i], dim=-1)
        sim_matrix = torch.mm(q.view(-1, q.shape[-1]), n.view(-1, n.shape[-1]).T)
        sim_max, _ = torch.max(sim_matrix, dim=1)
        sims.append(sim_max)
    sim = torch.mean(torch.stack(sims, dim=0), dim=0).reshape(1, 1, 16, 16)
    sim = F.interpolate(sim, size=224, mode='bilinear', align_corners=True)
    return 1 - sim


def compute_topk(query_pt, normal_pt, k=5):
    """B) Top-k soft matching"""
    sims = []
    for i in range(len(query_pt)):
        q = F.normalize(query_pt[i], dim=-1)
        n = F.normalize(normal_pt[i], dim=-1)
        sim_matrix = torch.mm(q.view(-1, q.shape[-1]), n.view(-1, n.shape[-1]).T)
        topk_vals, _ = torch.topk(sim_matrix, k=min(k, sim_matrix.shape[1]), dim=1)
        sims.append(topk_vals.mean(dim=1))
    sim = torch.mean(torch.stack(sims, dim=0), dim=0).reshape(1, 1, 16, 16)
    sim = F.interpolate(sim, size=224, mode='bilinear', align_corners=True)
    return 1 - sim


def compute_weighted(query_pt, normal_pt, weights=None):
    """C) Weighted layer combination"""
    if weights is None:
        weights = [0.1, 0.2, 0.3, 0.4]
    sims = []
    for i in range(len(query_pt)):
        q = F.normalize(query_pt[i], dim=-1)
        n = F.normalize(normal_pt[i], dim=-1)
        sim_matrix = torch.mm(q.view(-1, q.shape[-1]), n.view(-1, n.shape[-1]).T)
        sim_max, _ = torch.max(sim_matrix, dim=1)
        sims.append(sim_max * weights[i])
    sim = torch.sum(torch.stack(sims, dim=0), dim=0).reshape(1, 1, 16, 16)
    sim = F.interpolate(sim, size=224, mode='bilinear', align_corners=True)
    return 1 - sim


def compute_exp(query_pt, normal_pt, temp=10.0):
    """D) Exponential distance for sharper maps"""
    sims = []
    for i in range(len(query_pt)):
        q = F.normalize(query_pt[i], dim=-1)
        n = F.normalize(normal_pt[i], dim=-1)
        sim_matrix = torch.mm(q.view(-1, q.shape[-1]), n.view(-1, n.shape[-1]).T)
        sim_max, _ = torch.max(sim_matrix, dim=1)
        sims.append(sim_max)
    sim = torch.mean(torch.stack(sims, dim=0), dim=0).reshape(1, 1, 16, 16)
    sim = F.interpolate(sim, size=224, mode='bilinear', align_corners=True)
    return 1 - torch.exp(temp * (sim - 1))


def compute_combined(query_pt, normal_pt, k=5, weights=None, temp=10.0):
    """E) Combined: top-k + weighted + exp"""
    if weights is None:
        weights = [0.1, 0.2, 0.3, 0.4]
    sims = []
    for i in range(len(query_pt)):
        q = F.normalize(query_pt[i], dim=-1)
        n = F.normalize(normal_pt[i], dim=-1)
        sim_matrix = torch.mm(q.view(-1, q.shape[-1]), n.view(-1, n.shape[-1]).T)
        topk_vals, _ = torch.topk(sim_matrix, k=min(k, sim_matrix.shape[1]), dim=1)
        sims.append(topk_vals.mean(dim=1) * weights[i])
    sim = torch.sum(torch.stack(sims, dim=0), dim=0).reshape(1, 1, 16, 16)
    sim = F.interpolate(sim, size=224, mode='bilinear', align_corners=True)
    return 1 - torch.exp(temp * (sim - 1))


# All variants to test
VARIANTS = {
    'A_baseline':      lambda q, n: compute_baseline(q, n),
    'B_topk3':         lambda q, n: compute_topk(q, n, k=3),
    'B_topk5':         lambda q, n: compute_topk(q, n, k=5),
    'B_topk10':        lambda q, n: compute_topk(q, n, k=10),
    'C_wt_late':       lambda q, n: compute_weighted(q, n, [0.1, 0.2, 0.3, 0.4]),
    'C_wt_early':      lambda q, n: compute_weighted(q, n, [0.4, 0.3, 0.2, 0.1]),
    'D_exp5':          lambda q, n: compute_exp(q, n, temp=5.0),
    'D_exp10':         lambda q, n: compute_exp(q, n, temp=10.0),
    'D_exp20':         lambda q, n: compute_exp(q, n, temp=20.0),
    'E_combined':      lambda q, n: compute_combined(q, n, k=5, weights=[0.1, 0.2, 0.3, 0.4], temp=10.0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_shot', type=int, default=2)
    parser.add_argument('--round', type=int, default=3)
    args = parser.parse_args()

    # Load model
    model_args = {
        'model': 'openllama_peft',
        'stage': 1,
        'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
        'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
        'max_tgt_len': 512,
        'lora_r': 32,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
    }
    model = OpenLLAMAPEFTModel(**model_args)
    delta_ckpt = torch.load('../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt', map_location='cpu')
    model.load_state_dict(delta_ckpt, strict=False)
    delta_ckpt = torch.load('./ckpt/train_mvtec/pytorch_model.pt', map_location='cpu')
    model.load_state_dict(delta_ckpt, strict=False)
    model = model.eval().half().cuda()
    print('[!] Model loaded')

    root_dir = '/workspace/Master-Thesis/data/mvtec_anomaly_detection'
    mask_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # Per-class per-variant results
    results_per_class = {v: {} for v in VARIANTS}

    for c_name in ALL_CLASS_NAMES:
        print(f'\n=== {c_name} ===')
        normal_dir = os.path.join(root_dir, c_name, "train", "good")
        k_shot_override = {"screw": 1, "cable": 1}
        k = k_shot_override.get(c_name, args.k_shot)

        SPREAD_CLASSES = {"capsule", "cable", "toothbrush", "grid", "hazelnut"}
        if c_name in SPREAD_CLASSES:
            all_normal_paths = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
            normal_img_paths = select_spread_references(all_normal_paths, max(k, 4), model)[:k]
            print(f"    Using SPREAD selection")
        else:
            normal_img_paths = [os.path.join(normal_dir, str(args.round * 4 + i).zfill(3) + ".png") for i in range(4)][:k]
            print(f"    Using ORIGINAL selection")

        # Pre-compute normal patch tokens
        with torch.no_grad():
            normal_patch_tokens = model.encode_image_for_one_shot_with_aug(normal_img_paths)

        # Collect test images
        test_images = []
        for root, dirs, files in os.walk(os.path.join(root_dir, c_name, "test")):
            for file in sorted(files):
                if not file.endswith('.png'):
                    continue
                file_path = os.path.join(root, file)
                is_normal = 'good' in file_path.split('/')[-2]
                if not is_normal:
                    mask_path = file_path.replace('test', 'ground_truth').replace('.png', '_mask.png')
                else:
                    mask_path = None
                test_images.append((file_path, is_normal, mask_path))

        # Init per-class storage
        for v in VARIANTS:
            results_per_class[v][c_name] = {'i_preds': [], 'i_labels': [], 'p_preds': [], 'p_labels': []}

        for img_idx, (file_path, is_normal, mask_path) in enumerate(test_images):
            with torch.no_grad():
                query_patch_tokens = model.encode_image_for_one_shot([file_path])

            if is_normal:
                img_mask = np.zeros((224, 224))
            else:
                img_mask = mask_transform(Image.open(mask_path).convert('L')).squeeze().numpy()
                img_mask[img_mask > 0.1] = 1
                img_mask[img_mask <= 0.1] = 0

            i_label = 0 if is_normal else 1

            for v_name, v_func in VARIANTS.items():
                with torch.no_grad():
                    amap = v_func(query_patch_tokens, normal_patch_tokens)
                am_np = amap.squeeze().detach().cpu().numpy()
                d = results_per_class[v_name][c_name]
                d['i_preds'].append(am_np.max())
                d['i_labels'].append(i_label)
                d['p_preds'].append(am_np)
                d['p_labels'].append(img_mask)

            if (img_idx + 1) % 50 == 0:
                print(f'  {img_idx+1}/{len(test_images)} done')

        # Per-class summary
        print(f'\n  {c_name} results:')
        print(f'  {"Variant":<16} {"i_AUROC":>8} {"p_AUROC":>8} {"diff_i":>7} {"diff_p":>7}')
        print(f'  {"-"*50}')
        baseline_i, baseline_p = None, None
        for v_name in VARIANTS:
            d = results_per_class[v_name][c_name]
            try:
                i_auc = roc_auc_score(d['i_labels'], d['i_preds']) * 100
                p_auc = roc_auc_score(np.array(d['p_labels']).flatten(), np.array(d['p_preds']).flatten()) * 100
            except:
                i_auc = p_auc = 0.0
            if v_name == 'A_baseline':
                baseline_i, baseline_p = i_auc, p_auc
            di = i_auc - baseline_i if baseline_i is not None else 0
            dp = p_auc - baseline_p if baseline_p is not None else 0
            mark = " <--" if (di > 0.5 or dp > 0.5) else ""
            print(f'  {v_name:<16} {i_auc:>8.2f} {p_auc:>8.2f} {di:>+7.2f} {dp:>+7.2f}{mark}')

    # ============ OVERALL ============
    print(f'\n{"="*70}')
    print(f'OVERALL RESULTS (mean across 15 classes)')
    print(f'{"="*70}')
    print(f'{"Variant":<16} {"i_AUROC":>8} {"p_AUROC":>8} {"diff_i":>7} {"diff_p":>7}')
    print(f'{"-"*50}')

    baseline_avg_i, baseline_avg_p = None, None
    for v_name in VARIANTS:
        i_aucs, p_aucs = [], []
        for c_name in ALL_CLASS_NAMES:
            d = results_per_class[v_name][c_name]
            try:
                i_aucs.append(roc_auc_score(d['i_labels'], d['i_preds']) * 100)
                p_aucs.append(roc_auc_score(np.array(d['p_labels']).flatten(), np.array(d['p_preds']).flatten()) * 100)
            except:
                pass
        avg_i = np.mean(i_aucs)
        avg_p = np.mean(p_aucs)
        if v_name == 'A_baseline':
            baseline_avg_i, baseline_avg_p = avg_i, avg_p
        di = avg_i - baseline_avg_i if baseline_avg_i is not None else 0
        dp = avg_p - baseline_avg_p if baseline_avg_p is not None else 0
        mark = " ***" if (di > 0.3 or dp > 0.3) else ""
        print(f'{v_name:<16} {avg_i:>8.2f} {avg_p:>8.2f} {di:>+7.2f} {dp:>+7.2f}{mark}')

    print(f'\n[!] All done.')


if __name__ == '__main__':
    main()
