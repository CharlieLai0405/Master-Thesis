"""
方案 5: Improved Similarity Calculation
改進：把 few-shot comparison 的 max pooling 改成 top-k average，更 robust
這個改動在 model 的 extract_multimodal_feature 裡，所以需要 monkey-patch
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
parser.add_argument("--topk", type=int, default=5, help="Top-k for similarity averaging")

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

# === Monkey-patch extract_multimodal_feature to use top-k similarity ===
import types
from model.openllama import encode_text_with_prompt_ensemble, CLASS_NAMES as MODEL_CLASS_NAMES
from model.ImageBind import data as ib_data
from model.ImageBind.models.imagebind_model import ModalityType

TOPK = command_args.topk

def extract_multimodal_feature_topk(self, inputs, web_demo):
    features = []
    if inputs['image_paths']:
        prompt = inputs['prompt']
        c_name = 'object'
        for name in MODEL_CLASS_NAMES:
            if name in prompt:
                c_name = name
                break

        if not web_demo:
            image_embeds, _, patch_tokens = self.encode_image(inputs['image_paths'])
            feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, [c_name], self.device, self.text_adapter)
        else:
            image_embeds, _, patch_tokens = self.encode_image_for_web_demo(inputs['image_paths'])
            feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, [c_name], self.device, self.text_adapter)

        anomaly_maps = []
        for layer in range(len(patch_tokens)):
            patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)
            anomaly_map = (100.0 * patch_tokens[layer] @ feats_text_tensor.transpose(-2, -1))
            B, L, C = anomaly_map.shape
            H = int(np.sqrt(L))
            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                        size=224, mode='bilinear', align_corners=True)
            anomaly_map = torch.softmax(anomaly_map, dim=1)
            anomaly_maps.append(anomaly_map[:, 1, :, :])

        anomaly_map_ret = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1, keepdim=True)

        if inputs['normal_img_paths']:
            query_patch_tokens = self.encode_image_for_one_shot(inputs['image_paths'])
            if 'mvtec' in inputs['normal_img_paths'][0]:
                normal_patch_tokens = self.encode_image_for_one_shot_with_aug(inputs['normal_img_paths'])
            else:
                normal_patch_tokens = self.encode_image_for_one_shot(inputs['normal_img_paths'])
            sims = []

            for i in range(len(query_patch_tokens)):
                q_norm = F.normalize(query_patch_tokens[i], dim=-1)
                n_norm = F.normalize(normal_patch_tokens[i], dim=-1)
                cosine_similarity_matrix = torch.mm(q_norm.view(-1, q_norm.shape[-1]), n_norm.view(-1, n_norm.shape[-1]).transpose(0, 1))
                # === KEY CHANGE: top-k average instead of max ===
                k = min(TOPK, cosine_similarity_matrix.shape[1])
                topk_vals, _ = torch.topk(cosine_similarity_matrix, k=k, dim=1)
                sim_topk_avg = topk_vals.mean(dim=1)
                sims.append(sim_topk_avg)

            sim = torch.mean(torch.stack(sims, dim=0), dim=0).reshape(1, 1, 16, 16)
            sim = F.interpolate(sim, size=224, mode='bilinear', align_corners=True)
            anomaly_map_ret = 1 - sim

        features.append(image_embeds)
    if inputs['audio_paths']:
        audio_embeds, _ = self.encode_audio(inputs['audio_paths'])
        features.append(audio_embeds)
    if inputs['video_paths']:
        video_embeds, _ = self.encode_video(inputs['video_paths'])
        features.append(video_embeds)
    if inputs['thermal_paths']:
        thermal_embeds, _ = self.encode_thermal(inputs['thermal_paths'])
        features.append(thermal_embeds)

    feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
    return feature_embeds, anomaly_map_ret

model.extract_multimodal_feature = types.MethodType(extract_multimodal_feature_topk, model)
print(f'[!] Monkey-patched extract_multimodal_feature with top-{TOPK} similarity')

# === Standard test loop (same as original) ===
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

input_text = "Is there any anomaly in the image?"
root_dir = '/workspace/Master-Thesis/data/mvtec_anomaly_detection'
mask_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

ALL_CLASS_NAMES = ["bottle", "cable", "capsule", "carpet", "grid","hazelnut", "leather", "metal_nut", "pill", "screw","tile", "toothbrush", "transistor", "wood", "zipper"]
CLASS_NAMES = [command_args.class_name] if command_args.class_name else ALL_CLASS_NAMES
precision = []

for c_name in CLASS_NAMES:
    normal_img_paths = ["/workspace/Master-Thesis/data/mvtec_anomaly_detection/"+c_name+"/train/good/"+str(command_args.round * 4 + i).zfill(3)+".png" for i in range(4)]
    k_shot_override = {"screw": 1, "cable": 1}
    k = k_shot_override.get(c_name, command_args.k_shot)
    normal_img_paths = normal_img_paths[:k]

    right, wrong = 0, 0
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
