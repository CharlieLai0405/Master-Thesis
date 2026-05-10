"""
VisA Best Combo: per-class optimal k + spread/original selection
"""
import os
from model.openllama import OpenLLAMAPEFTModel
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import csv
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
parser.add_argument("--few_shot", type=bool, default=True)
parser.add_argument("--round", type=int, default=14)
parser.add_argument("--n_refs", type=int, default=4)
parser.add_argument("--anomalygpt_ckpt", type=str, default='./ckpt/train_visa/pytorch_model.pt')
command_args = parser.parse_args()

# Per-class optimal config
CLASS_CONFIG = {
    'candle':     {'k': 1, 'selection': 'original'},
    'capsules':   {'k': 2, 'selection': 'spread'},
    'cashew':     {'k': 2, 'selection': 'spread'},
    'chewinggum': {'k': 2, 'selection': 'original'},
    'fryum':      {'k': 1, 'selection': 'spread'},
    'macaroni1':  {'k': 1, 'selection': 'original'},
    'macaroni2':  {'k': 2, 'selection': 'original'},
    'pcb1':       {'k': 2, 'selection': 'original'},
    'pcb2':       {'k': 1, 'selection': 'original'},
    'pcb3':       {'k': 1, 'selection': 'spread'},
    'pcb4':       {'k': 2, 'selection': 'original'},
    'pipe_fryum': {'k': 1, 'selection': 'original'},
}

describles = {}
describles['candle'] = "This is a photo of 4 candles for anomaly detection, every candle should be round, without any damage, flaw, defect, scratch, hole or broken part."
describles['capsules'] = "This is a photo of many small capsules for anomaly detection, every capsule is green, should be without any damage, flaw, defect, scratch, hole or broken part."
describles['cashew'] = "This is a photo of a cashew for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['chewinggum'] = "This is a photo of a chewinggom for anomaly detection, which should be white, without any damage, flaw, defect, scratch, hole or broken part."
describles['fryum'] = "This is a photo of a fryum for anomaly detection on green background, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['macaroni1'] = "This is a photo of 4 macaronis for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['macaroni2'] = "This is a photo of 4 macaronis for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles['pcb1'] = "This is a photo of a printed circuit board (PCB) for anomaly detection, which should have clean solder joints, intact traces, and properly placed components without any missing parts, bent pins, scratches, or contamination."
describles['pcb2'] = "This is a photo of a printed circuit board (PCB) for anomaly detection, which should have clean solder joints, intact traces, and properly placed components without any missing parts, bent pins, scratches, or contamination."
describles['pcb3'] = "This is a photo of a printed circuit board (PCB) for anomaly detection, which should have clean solder joints, intact traces, and properly placed components without any missing parts, bent pins, scratches, or contamination."
describles['pcb4'] = "This is a photo of a printed circuit board (PCB) for anomaly detection, which should have clean solder joints, intact traces, and properly placed components without any missing parts, bent pins, scratches, or contamination."
describles['pipe_fryum'] = "This is a photo of a pipe fryum for anomaly detection, which should be tubular shaped without any damage, flaw, defect, scratch, hole or broken part."

FEW_SHOT = command_args.few_shot

args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': command_args.anomalygpt_ckpt,
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
model = model.eval().half().cuda()
print(f'[!] init the 7b model over ...')


def select_spread_references(all_paths, n_refs, model_ref):
    if len(all_paths) <= n_refs:
        return all_paths
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
            cls_features = embeddings['vision'][0]
        all_features.append(cls_features)
    all_features = torch.cat(all_features, dim=0)
    all_features = torch.nn.functional.normalize(all_features, dim=-1)
    mean_feat = all_features.mean(dim=0, keepdim=True)
    mean_feat = torch.nn.functional.normalize(mean_feat, dim=-1)
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


def predict(input_text, image_path, normal_img_path, max_length, top_p, temperature, history, modality_cache):
    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\n### Assistant: {a}\n###'
        else:
            prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
    if len(history) == 0:
        prompt_text += f'{input_text}'
    else:
        prompt_text += f' Human: {input_text}'
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
root_dir = '../data/VisA'
mask_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
datas_csv_path = '../data/VisA/split_csv/1cls.csv'

CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

# Load all paths from CSV
file_paths = {c: [] for c in CLASS_NAMES}
all_normal_paths = {c: [] for c in CLASS_NAMES}
original_normal_all = {c: [] for c in CLASS_NAMES}

with open(datas_csv_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[0] in CLASS_NAMES:
            if row[1] == 'test':
                file_paths[row[0]].append(os.path.join(root_dir, row[3]))
            elif row[1] == 'train' and row[2] == 'normal':
                all_normal_paths[row[0]].append(os.path.join(root_dir, row[3]))

# Original refs (for non-spread classes)
with open(datas_csv_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[0] in CLASS_NAMES and row[1] == 'train':
            original_normal_all[row[0]].append(os.path.join(root_dir, row[3]))

# Get original refs at round offset
original_normal = {}
for c in CLASS_NAMES:
    max_k = 2
    needed = command_args.round * 4 + max_k
    original_normal[c] = original_normal_all[c][command_args.round * 4:needed] if len(original_normal_all[c]) >= needed else original_normal_all[c][-max_k:]

p_auc_list = []
i_auc_list = []
precision = []

for c_name in CLASS_NAMES:
    cfg = CLASS_CONFIG[c_name]
    k = cfg['k']
    sel = cfg['selection']
    print(f"\n[*] Processing {c_name} (k={k}, {sel})...")

    if sel == 'spread':
        normal_img_path = select_spread_references(all_normal_paths[c_name], max(k, command_args.n_refs), model)[:k]
    else:
        normal_img_path = original_normal[c_name][:k]

    print(f"    Using {len(normal_img_path)} reference images")

    right = 0
    wrong = 0
    p_pred, p_label, i_pred, i_label = [], [], [], []

    for file_path in tqdm(file_paths[c_name]):
        if FEW_SHOT:
            resp, anomaly_map = predict(describles[c_name] + ' ' + input_text, file_path, normal_img_path, 512, 0.01, 1.0, [], [])
        else:
            resp, anomaly_map = predict(describles[c_name] + ' ' + input_text, file_path, None, 512, 0.01, 1.0, [], [])

        is_normal = 'Normal' in file_path.split('/')[-2]
        if is_normal:
            img_mask = Image.fromarray(np.zeros((224, 224)), mode='L')
        else:
            mask_path = file_path.replace('Images', 'Masks').replace('.JPG', '.png')
            img_mask = Image.open(mask_path).convert('L')

        img_mask = mask_transform(img_mask)
        threshold = img_mask.max() / 100
        img_mask[img_mask > threshold], img_mask[img_mask <= threshold] = 1, 0
        img_mask = img_mask.squeeze().reshape(224, 224).cpu().numpy()
        anomaly_map = anomaly_map.reshape(224, 224).detach().cpu().numpy()

        p_label.append(img_mask)
        p_pred.append(anomaly_map)
        i_label.append(1 if not is_normal else 0)
        i_pred.append(anomaly_map.max())

        if 'Normal' not in file_path and 'Yes' in resp:
            right += 1
        elif 'Normal' in file_path and 'No' in resp:
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

print("\n" + "="*50)
print("i_AUROC:", torch.tensor(i_auc_list).mean())
print("p_AUROC:", torch.tensor(p_auc_list).mean())
print("precision:", torch.tensor(precision).mean())
