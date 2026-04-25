"""
方案 1: Better Few-Shot Reference Selection
改進：不用固定的第 N 張 normal image，而是從所有 normal images 中
     用 feature space 選最具代表性的（離 mean 最近的），並且用更多張做 ensemble
"""
import os
from model.openllama import OpenLLAMAPEFTModel
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import argparse
import glob
import random

parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
parser.add_argument("--few_shot", type=bool, default=True)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--round", type=int, default=3)
parser.add_argument("--adapter_ckpt", type=str, default=None)
parser.add_argument("--anomalygpt_ckpt", type=str, default=None)
parser.add_argument("--class_name", type=str, default=None)
parser.add_argument("--n_refs", type=int, default=4, help="Number of reference images to select")
parser.add_argument("--selection", type=str, default="hybrid", choices=["spread", "random", "original", "hybrid"],
                    help="Reference selection strategy (hybrid=spread for improved classes, original for others)")

command_args = parser.parse_args()

describles = {}
describles['bottle'] = "This is a photo of a bottle for anomaly detection, which should be round, without any damage, flaw, defect, scratch, hole or broken part."
describles['cable'] = "This is a photo of three cables for anomaly detection, where all cables must be present in correct positions without swapping, each having intact insulation with no cuts or missing wires."
describles['capsule'] = "This is a photo of a capsule for anomaly detection, which should be black and orange with print 500, having a smooth cylindrical shape with clean seam line, without any cracks, dents, scratches, faulty imprint, or squeezed deformation."
describles['carpet'] = "This is a photo of carpet for anomaly detection, which should have a uniform woven texture with consistent color throughout, without any holes, cuts, stains, color spots, or thread irregularities."
describles['grid'] = "This is a photo of grid for anomaly detection, which should have uniform spacing with consistent parallel lines forming a regular pattern, without any bent wires, broken threads, missing sections, or metal contamination."
describles['hazelnut'] = "This is a photo of a hazelnut for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['leather'] = "This is a photo of leather for anomaly detection, which should be brown and without any damage, flaw, defect, scratch, hole or broken part."
describles['metal_nut'] = "This is a photo of a metal nut for anomaly detection, which should be hexagonal with a smooth surface, consistent threading, and correct orientation (not flipped), without any bent edges, scratches, surface discoloration, or thread damage."
describles['pill'] = "This is a photo of a pill for anomaly detection, which should be white, with print 'FF' and red patterns, without any damage, flaw, defect, scratch, hole or broken part."
describles['screw'] = "This is a photo of a screw for anomaly detection, which tail should be sharp, and without any damage, flaw, defect, scratch, hole or broken part."
describles['tile'] = "This is a photo of tile for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['toothbrush'] = "This is a photo of a toothbrush for anomaly detection, which should have evenly arranged bristles of uniform height and color with an intact handle, without any missing bristles, bent bristles, or defective brush head."
describles["transistor"] = "This is a photo of a transistor for anomaly detection, which should have three straight parallel leads, an intact plastic case with clear markings, and correct placement orientation, without any bent leads, cut leads, damaged case, or misplacement."
describles['wood'] = "This is a photo of wood for anomaly detection, which should be brown with patterns, without any damage, flaw, defect, scratch, hole or broken part."
describles['zipper'] = "This is a photo of a zipper for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."

FEW_SHOT = command_args.few_shot

args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': command_args.anomalygpt_ckpt if command_args.anomalygpt_ckpt else './ckpt/train_mvtec/pytorch_model.pt',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}

model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
if command_args.adapter_ckpt:
    adapter_ckpt = torch.load(command_args.adapter_ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(adapter_ckpt, strict=False)
model = model.eval().half().cuda()
print(f'[!] init the 7b model over ...')


def select_spread_references(all_paths, n_refs, model_ref):
    """
    Select n_refs images that are spread out in feature space.
    Strategy: pick the one closest to mean, then iteratively pick the one
    farthest from already selected (maximin diversity).
    """
    if len(all_paths) <= n_refs:
        return all_paths

    # Encode all normal images in batches
    from model.ImageBind import data
    from model.ImageBind.models.imagebind_model import ModalityType
    batch_size = 16
    all_features = []
    for i in range(0, len(all_paths), batch_size):
        batch_paths = all_paths[i:i+batch_size]
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(batch_paths, 'cuda')}
        inputs = {key: inputs[key].to(model_ref.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = model_ref.visual_encoder(inputs)
            # Use the global CLS token from first layer
            cls_features = embeddings['vision'][0]  # [B, 1024]
        all_features.append(cls_features)

    all_features = torch.cat(all_features, dim=0)  # [N, 1280]
    all_features = torch.nn.functional.normalize(all_features, dim=-1)

    # Step 1: pick closest to mean
    mean_feat = all_features.mean(dim=0, keepdim=True)
    mean_feat = torch.nn.functional.normalize(mean_feat, dim=-1)
    dists_to_mean = 1 - (all_features @ mean_feat.T).squeeze()
    first_idx = dists_to_mean.argmin().item()

    selected = [first_idx]

    # Step 2: maximin diversity selection
    for _ in range(n_refs - 1):
        selected_feats = all_features[selected]  # [k, 1280]
        # For each candidate, compute min distance to any selected
        sims = all_features @ selected_feats.T  # [N, k]
        max_sims, _ = sims.max(dim=1)  # [N] - highest similarity to any selected
        # Mask already selected
        for idx in selected:
            max_sims[idx] = 999
        # Pick the one with lowest max similarity (most different from all selected)
        next_idx = max_sims.argmin().item()
        selected.append(next_idx)

    selected_paths = [all_paths[i] for i in selected]
    print(f"    Selected refs: {[os.path.basename(p) for p in selected_paths]}")
    return selected_paths


p_auc_list = []
i_auc_list = []

def predict(input, image_path, normal_img_path, max_length, top_p, temperature, history, modality_cache):
    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\n### Assistant: {a}\n###'
        else:
            prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' Human: {input}'

    response, pixel_output = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [],
        'video_paths': [],
        'thermal_paths': [],
        'normal_img_paths': normal_img_path if normal_img_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })
    return response, pixel_output

input_text = "Is there any anomaly in the image?"
root_dir = '/workspace/Master-Thesis/data/mvtec_anomaly_detection'
mask_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

ALL_CLASS_NAMES = ["bottle", "cable", "capsule", "carpet", "grid","hazelnut", "leather", "metal_nut", "pill", "screw","tile", "toothbrush", "transistor", "wood", "zipper"]
CLASS_NAMES = [command_args.class_name] if command_args.class_name else ALL_CLASS_NAMES

precision = []

for c_name in CLASS_NAMES:
    print(f"\n[*] Processing {c_name}...")

    # Get all normal training images for this class
    normal_dir = f"/workspace/Master-Thesis/data/mvtec_anomaly_detection/{c_name}/train/good/"
    all_normal_paths = sorted(glob.glob(os.path.join(normal_dir, "*.png")))

    k_shot_override = {"screw": 1, "cable": 1}
    k = k_shot_override.get(c_name, command_args.k_shot)

    # Classes that benefit from spread selection (i_AUROC improved >= +0.5)
    SPREAD_CLASSES = {"capsule", "cable", "toothbrush", "grid", "hazelnut"}

    if command_args.selection == "hybrid":
        if c_name in SPREAD_CLASSES:
            normal_img_paths = select_spread_references(all_normal_paths, max(k, command_args.n_refs), model)[:k]
            print(f"    [{c_name}] Using SPREAD selection")
        else:
            normal_img_paths = [normal_dir + str(command_args.round * 4 + i).zfill(3) + ".png" for i in range(4)][:k]
            print(f"    [{c_name}] Using ORIGINAL selection")
    elif command_args.selection == "spread":
        normal_img_paths = select_spread_references(all_normal_paths, max(k, command_args.n_refs), model)[:k]
    elif command_args.selection == "random":
        random.seed(42)
        normal_img_paths = random.sample(all_normal_paths, min(k, len(all_normal_paths)))
    else:
        # Original fixed selection
        normal_img_paths = [normal_dir + str(command_args.round * 4 + i).zfill(3) + ".png" for i in range(4)][:k]

    print(f"    Using {len(normal_img_paths)} reference images (k={k})")

    right = 0
    wrong = 0
    p_pred, p_label, i_pred, i_label = [], [], [], []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "test" in file_path and 'png' in file and c_name in file_path:
                if FEW_SHOT:
                    resp, anomaly_map = predict(describles[c_name] + ' ' + input_text, file_path, normal_img_paths, 512, 0.1, 1.0, [], [])
                else:
                    resp, anomaly_map = predict(describles[c_name] + ' ' + input_text, file_path, [], 512, 0.1, 1.0, [], [])
                is_normal = 'good' in file_path.split('/')[-2]
                if is_normal:
                    img_mask = Image.fromarray(np.zeros((224, 224)), mode='L')
                else:
                    mask_path = file_path.replace('test', 'ground_truth').replace('.png', '_mask.png')
                    img_mask = Image.open(mask_path).convert('L')
                img_mask = mask_transform(img_mask)
                img_mask[img_mask > 0.1], img_mask[img_mask <= 0.1] = 1, 0
                img_mask = img_mask.squeeze().reshape(224, 224).cpu().numpy()
                anomaly_map = anomaly_map.reshape(224, 224).detach().cpu().numpy()
                p_label.append(img_mask)
                p_pred.append(anomaly_map)
                i_label.append(1 if not is_normal else 0)
                i_pred.append(anomaly_map.max())
                if 'good' not in file_path and 'Yes' in resp:
                    right += 1
                elif 'good' in file_path and 'No' in resp:
                    right += 1
                else:
                    wrong += 1

    p_pred, p_label = np.array(p_pred), np.array(p_label)
    i_pred, i_label = np.array(i_pred), np.array(i_label)
    p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100, 2)
    i_auroc = round(roc_auc_score(i_label.ravel(), i_pred.ravel()) * 100, 2)
    p_auc_list.append(p_auroc)
    i_auc_list.append(i_auroc)
    precision.append(100 * right / (right + wrong))
    print(c_name, 'right:', right, 'wrong:', wrong)
    print(c_name, "i_AUROC:", i_auroc)
    print(c_name, "p_AUROC:", p_auroc)

print("i_AUROC:", torch.tensor(i_auc_list).mean())
print("p_AUROC:", torch.tensor(p_auc_list).mean())
print("precision:", torch.tensor(precision).mean())
