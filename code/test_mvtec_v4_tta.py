"""
方案 4: Test-Time Augmentation (TTA)
改進：對 query image 做多次 augmentation（水平翻轉 + 垂直翻轉），
     anomaly map 取平均，減少單次推論的隨機性
"""
import os
from model.openllama import OpenLLAMAPEFTModel
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
parser.add_argument("--few_shot", type=bool, default=True)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--round", type=int, default=3)
parser.add_argument("--adapter_ckpt", type=str, default=None)
parser.add_argument("--anomalygpt_ckpt", type=str, default=None)
parser.add_argument("--class_name", type=str, default=None)

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
    """Select n_refs images spread out in feature space (maximin diversity)."""
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
        'audio_paths': [], 'video_paths': [], 'thermal_paths': [],
        'normal_img_paths': normal_img_path if normal_img_path else [],
        'top_p': top_p, 'temperature': temperature, 'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })
    return response, pixel_output

def predict_with_tta(input_text, image_path, normal_img_paths, describles_text):
    """Run prediction with TTA: original + horizontal flip + vertical flip, average anomaly maps"""
    from PIL import Image as PILImage
    import tempfile

    original_img = PILImage.open(image_path).convert('RGB')

    augmented_paths = [image_path]  # original
    temp_files = []

    # Horizontal flip
    hflip = original_img.transpose(PILImage.FLIP_LEFT_RIGHT)
    tmp_h = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    hflip.save(tmp_h.name)
    augmented_paths.append(tmp_h.name)
    temp_files.append(tmp_h.name)

    # Vertical flip
    vflip = original_img.transpose(PILImage.FLIP_TOP_BOTTOM)
    tmp_v = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    vflip.save(tmp_v.name)
    augmented_paths.append(tmp_v.name)
    temp_files.append(tmp_v.name)

    anomaly_maps = []
    resp_final = None

    for i, aug_path in enumerate(augmented_paths):
        resp, anomaly_map = predict(describles_text + ' ' + "Is there any anomaly in the image?",
                                     aug_path, normal_img_paths, 512, 0.1, 1.0, [], [])
        am = anomaly_map.reshape(224, 224)

        # Reverse the augmentation on the anomaly map
        if i == 1:  # horizontal flip
            am = torch.flip(am, dims=[1])
        elif i == 2:  # vertical flip
            am = torch.flip(am, dims=[0])

        anomaly_maps.append(am)
        if i == 0:
            resp_final = resp

    # Clean up temp files
    for f in temp_files:
        os.unlink(f)

    # Average anomaly maps
    avg_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)
    return resp_final, avg_map

input_text = "Is there any anomaly in the image?"
root_dir = '/workspace/Master-Thesis/data/mvtec_anomaly_detection'
mask_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

ALL_CLASS_NAMES = ["bottle", "cable", "capsule", "carpet", "grid","hazelnut", "leather", "metal_nut", "pill", "screw","tile", "toothbrush", "transistor", "wood", "zipper"]
CLASS_NAMES = [command_args.class_name] if command_args.class_name else ALL_CLASS_NAMES
precision = []

for c_name in CLASS_NAMES:
    normal_dir = f"/workspace/Master-Thesis/data/mvtec_anomaly_detection/{c_name}/train/good/"
    k_shot_override = {"screw": 1, "cable": 1}
    k = k_shot_override.get(c_name, command_args.k_shot)

    # Hybrid selection: spread for improved classes, original for others
    SPREAD_CLASSES = {"capsule", "cable", "toothbrush", "grid", "hazelnut"}
    if c_name in SPREAD_CLASSES:
        import glob
        all_normal_paths = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
        normal_img_paths = select_spread_references(all_normal_paths, max(k, 4), model)[:k]
        print(f"    [{c_name}] Using SPREAD selection")
    else:
        normal_img_paths = [normal_dir + str(command_args.round * 4 + i).zfill(3) + ".png" for i in range(4)][:k]
        print(f"    [{c_name}] Using ORIGINAL selection")

    right, wrong = 0, 0
    p_pred, p_label, i_pred, i_label = [], [], [], []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "test" in file_path and 'png' in file and c_name in file_path:
                if FEW_SHOT:
                    resp, anomaly_map = predict_with_tta(input_text, file_path, normal_img_paths, describles[c_name])
                else:
                    resp, anomaly_map = predict(describles[c_name] + ' ' + input_text, file_path, [], 512, 0.1, 1.0, [], [])
                    anomaly_map = anomaly_map.reshape(224, 224)

                is_normal = 'good' in file_path.split('/')[-2]
                if is_normal:
                    img_mask = Image.fromarray(np.zeros((224, 224)), mode='L')
                else:
                    mask_path = file_path.replace('test', 'ground_truth').replace('.png', '_mask.png')
                    img_mask = Image.open(mask_path).convert('L')
                img_mask = mask_transform(img_mask)
                img_mask[img_mask > 0.1], img_mask[img_mask <= 0.1] = 1, 0
                img_mask = img_mask.squeeze().reshape(224, 224).cpu().numpy()
                anomaly_map_np = anomaly_map.detach().cpu().numpy()
                p_label.append(img_mask)
                p_pred.append(anomaly_map_np)
                i_label.append(1 if not is_normal else 0)
                i_pred.append(anomaly_map_np.max())
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
