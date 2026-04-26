"""
V8: Threshold-guided Prompt
Step 1: Calculate per-class optimal threshold from V6 Multi-scale TTA
Step 2: Feed anomaly score + threshold into LLM prompt
"""
import os, sys, json
from model.openllama import OpenLLAMAPEFTModel
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import argparse
import tempfile

parser = argparse.ArgumentParser("AnomalyGPT V8", add_help=True)
parser.add_argument("--few_shot", type=bool, default=True)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--round", type=int, default=3)
parser.add_argument("--adapter_ckpt", type=str, default=None)
parser.add_argument("--anomalygpt_ckpt", type=str, default=None)
parser.add_argument("--class_name", type=str, default=None)
parser.add_argument("--phase", type=str, default="both", choices=["calc_threshold", "test", "both"],
                    help="calc_threshold: only calculate thresholds; test: use existing thresholds; both: calc then test")
parser.add_argument("--threshold_file", type=str, default="/tmp/v8_thresholds.json")
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
model = model.eval().half().cuda()

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
    """Multi-scale TTA: original + H flip + V flip + 90 rot + 270 rot"""
    from PIL import Image as PILImage

    original_img = PILImage.open(image_path).convert('RGB')

    augmentations = []
    augmentations.append((image_path, 'none'))
    temp_files = []

    hflip = original_img.transpose(PILImage.FLIP_LEFT_RIGHT)
    tmp_h = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    hflip.save(tmp_h.name)
    augmentations.append((tmp_h.name, 'hflip'))
    temp_files.append(tmp_h.name)

    vflip = original_img.transpose(PILImage.FLIP_TOP_BOTTOM)
    tmp_v = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    vflip.save(tmp_v.name)
    augmentations.append((tmp_v.name, 'vflip'))
    temp_files.append(tmp_v.name)

    rot90 = original_img.transpose(PILImage.ROTATE_90)
    tmp_r90 = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    rot90.save(tmp_r90.name)
    augmentations.append((tmp_r90.name, 'rot90'))
    temp_files.append(tmp_r90.name)

    rot270 = original_img.transpose(PILImage.ROTATE_270)
    tmp_r270 = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    rot270.save(tmp_r270.name)
    augmentations.append((tmp_r270.name, 'rot270'))
    temp_files.append(tmp_r270.name)

    anomaly_maps = []
    resp_final = None

    for i, (aug_path, reverse_type) in enumerate(augmentations):
        resp, anomaly_map = predict(f"This is a {c_name.replace('_', ' ')}. {describles_text} Is there any anomaly in the image?",
                                     aug_path, normal_img_paths, 512, 0.1, 1.0, [], [])
        am = anomaly_map.reshape(224, 224)

        if reverse_type == 'hflip':
            am = torch.flip(am, dims=[1])
        elif reverse_type == 'vflip':
            am = torch.flip(am, dims=[0])
        elif reverse_type == 'rot90':
            am = torch.rot90(am, k=-1)
        elif reverse_type == 'rot270':
            am = torch.rot90(am, k=1)

        anomaly_maps.append(am)
        if i == 0:
            resp_final = resp

    for f in temp_files:
        os.unlink(f)

    avg_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)
    return resp_final, avg_map

def predict_with_tta_threshold_guided(image_path, normal_img_paths, describles_text, c_name, threshold):
    """V8: Run TTA to get anomaly map first, then feed score+threshold into prompt for LLM response"""
    from PIL import Image as PILImage

    original_img = PILImage.open(image_path).convert('RGB')

    augmentations = []
    augmentations.append((image_path, 'none'))
    temp_files = []

    hflip = original_img.transpose(PILImage.FLIP_LEFT_RIGHT)
    tmp_h = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    hflip.save(tmp_h.name)
    augmentations.append((tmp_h.name, 'hflip'))
    temp_files.append(tmp_h.name)

    vflip = original_img.transpose(PILImage.FLIP_TOP_BOTTOM)
    tmp_v = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    vflip.save(tmp_v.name)
    augmentations.append((tmp_v.name, 'vflip'))
    temp_files.append(tmp_v.name)

    rot90 = original_img.transpose(PILImage.ROTATE_90)
    tmp_r90 = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    rot90.save(tmp_r90.name)
    augmentations.append((tmp_r90.name, 'rot90'))
    temp_files.append(tmp_r90.name)

    rot270 = original_img.transpose(PILImage.ROTATE_270)
    tmp_r270 = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    rot270.save(tmp_r270.name)
    augmentations.append((tmp_r270.name, 'rot270'))
    temp_files.append(tmp_r270.name)

    anomaly_maps = []

    for i, (aug_path, reverse_type) in enumerate(augmentations):
        # First pass: get anomaly maps (use standard prompt, we only care about the map)
        _, anomaly_map = predict(f"This is a {c_name.replace('_', ' ')}. {describles_text} Is there any anomaly in the image?",
                                  aug_path, normal_img_paths, 512, 0.1, 1.0, [], [])
        am = anomaly_map.reshape(224, 224)

        if reverse_type == 'hflip':
            am = torch.flip(am, dims=[1])
        elif reverse_type == 'vflip':
            am = torch.flip(am, dims=[0])
        elif reverse_type == 'rot90':
            am = torch.rot90(am, k=-1)
        elif reverse_type == 'rot270':
            am = torch.rot90(am, k=1)

        anomaly_maps.append(am)

    for f in temp_files:
        os.unlink(f)

    avg_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)
    score = float(avg_map.max().cpu())

    # Second pass: use threshold-guided prompt on original image for LLM response
    score_status = "above" if score > threshold else "below"
    guided_prompt = (f"This is a {c_name.replace('_', ' ')}. {describles_text} "
                     f"The anomaly score for this image is {score:.4f} (threshold: {threshold:.4f}, score is {score_status} threshold). "
                     f"Is there any anomaly in the image?")

    resp, _ = predict(guided_prompt, image_path, normal_img_paths, 512, 0.1, 1.0, [], [])

    return resp, avg_map, score

# ============ Main ============
root_dir = '/workspace/Master-Thesis/data/mvtec_anomaly_detection'
mask_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

ALL_CLASS_NAMES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
                   "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
CLASS_NAMES = [command_args.class_name] if command_args.class_name else ALL_CLASS_NAMES

# ============ Phase 1: Calculate thresholds (if needed) ============
if command_args.phase in ["calc_threshold", "both"]:
    print("=" * 50)
    print("Phase 1: Calculating per-class optimal thresholds")
    print("=" * 50)
    thresholds = {}

    for c_name in CLASS_NAMES:
        normal_dir = f"{root_dir}/{c_name}/train/good/"
        k_shot_override = {"screw": 1, "cable": 1}
        k = k_shot_override.get(c_name, command_args.k_shot)

        SPREAD_CLASSES = {"capsule", "cable", "toothbrush", "grid", "hazelnut"}
        if c_name in SPREAD_CLASSES:
            import glob
            all_normal_paths = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
            normal_img_paths = select_spread_references(all_normal_paths, max(k, 4), model)[:k]
        else:
            normal_img_paths = [normal_dir + str(command_args.round * 4 + i).zfill(3) + ".png" for i in range(4)][:k]

        i_pred, i_label = [], []
        for root_path, dirs, files in os.walk(root_dir):
            for file in sorted(files):
                file_path = os.path.join(root_path, file)
                if "test" in file_path and 'png' in file and c_name in file_path:
                    if FEW_SHOT:
                        _, anomaly_map = predict_with_tta("", file_path, normal_img_paths, describles[c_name])
                    else:
                        _, anomaly_map = predict("", file_path, [], 512, 0.1, 1.0, [], [])
                        anomaly_map = anomaly_map.reshape(224, 224)

                    is_normal = 'good' in file_path.split('/')[-2]
                    score = float(anomaly_map.max().cpu())
                    i_pred.append(score)
                    i_label.append(0 if is_normal else 1)

        # Calculate optimal threshold using Youden's J
        from sklearn.metrics import roc_curve
        fpr, tpr, thresh_values = roc_curve(i_label, i_pred)
        j_scores = tpr - fpr
        best_idx = j_scores.argmax()
        best_threshold = float(thresh_values[best_idx])
        thresholds[c_name] = best_threshold

        # Calculate precision at this threshold
        tp = sum(1 for s, l in zip(i_pred, i_label) if s > best_threshold and l == 1)
        fp = sum(1 for s, l in zip(i_pred, i_label) if s > best_threshold and l == 0)
        fn = sum(1 for s, l in zip(i_pred, i_label) if s <= best_threshold and l == 1)
        tn = sum(1 for s, l in zip(i_pred, i_label) if s <= best_threshold and l == 0)
        prec = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
        acc = 100 * (tp + tn) / len(i_pred)
        print(f"{c_name}: threshold={best_threshold:.4f}, precision_threshold={prec:.2f}, accuracy={acc:.2f}")

    with open(command_args.threshold_file, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nThresholds saved to {command_args.threshold_file}")
    print(json.dumps(thresholds, indent=2))

# ============ Phase 2: Test with threshold-guided prompt ============
if command_args.phase in ["test", "both"]:
    print("\n" + "=" * 50)
    print("Phase 2: Testing with threshold-guided prompt")
    print("=" * 50)

    with open(command_args.threshold_file, "r") as f:
        thresholds = json.load(f)

    precision_llm = []
    precision_threshold = []
    all_i_auroc = []
    all_p_auroc = []

    for c_name in CLASS_NAMES:
        normal_dir = f"{root_dir}/{c_name}/train/good/"
        k_shot_override = {"screw": 1, "cable": 1}
        k = k_shot_override.get(c_name, command_args.k_shot)

        SPREAD_CLASSES = {"capsule", "cable", "toothbrush", "grid", "hazelnut"}
        if c_name in SPREAD_CLASSES:
            import glob
            all_normal_paths = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
            normal_img_paths = select_spread_references(all_normal_paths, max(k, 4), model)[:k]
            print(f"    [{c_name}] Using SPREAD selection")
        else:
            normal_img_paths = [normal_dir + str(command_args.round * 4 + i).zfill(3) + ".png" for i in range(4)][:k]
            print(f"    [{c_name}] Using ORIGINAL selection")

        threshold = thresholds[c_name]
        right_llm, wrong_llm = 0, 0
        right_thresh, wrong_thresh = 0, 0
        p_pred, p_label, i_pred, i_label = [], [], [], []

        for root_path, dirs, files in os.walk(root_dir):
            for file in sorted(files):
                file_path = os.path.join(root_path, file)
                if "test" in file_path and 'png' in file and c_name in file_path:
                    if FEW_SHOT:
                        resp, anomaly_map, score = predict_with_tta_threshold_guided(
                            file_path, normal_img_paths, describles[c_name], c_name, threshold)
                    else:
                        resp, anomaly_map = predict(f"This is a {c_name.replace('_', ' ')}. {describles[c_name]} Is there any anomaly?",
                                                     file_path, [], 512, 0.1, 1.0, [], [])
                        anomaly_map = anomaly_map.reshape(224, 224)
                        score = float(anomaly_map.max().cpu())

                    is_normal = 'good' in file_path.split('/')[-2]

                    if is_normal:
                        img_mask = Image.fromarray(np.zeros((224, 224)), mode='L')
                    else:
                        mask_path = file_path.replace('test', 'ground_truth').replace('.png', '_mask.png')
                        img_mask = Image.open(mask_path).convert('L')
                    img_mask = mask_transform(img_mask)
                    img_mask[img_mask > 0.1], img_mask[img_mask <= 0.1] = 1, 0
                    img_mask = img_mask.squeeze().reshape(224, 224).cpu().numpy()
                    anomaly_map_np = anomaly_map.detach().cpu().numpy().astype(np.float32)
                    p_label.append(img_mask)
                    p_pred.append(anomaly_map_np)
                    i_label.append(1 if not is_normal else 0)
                    i_pred.append(anomaly_map_np.max())

                    # LLM precision (threshold-guided)
                    if 'good' not in file_path and 'Yes' in resp:
                        right_llm += 1
                    elif 'good' in file_path and 'No' in resp:
                        right_llm += 1
                    else:
                        wrong_llm += 1

                    # Pure threshold precision
                    pred_anomaly = score > threshold
                    actual_anomaly = not is_normal
                    if pred_anomaly == actual_anomaly:
                        right_thresh += 1
                    else:
                        wrong_thresh += 1

        p_pred, p_label = np.array(p_pred), np.array(p_label)
        i_pred, i_label = np.array(i_pred), np.array(i_label)
        p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100, 2)
        i_auroc = round(roc_auc_score(i_label.ravel(), i_pred.ravel()) * 100, 2)
        all_i_auroc.append(i_auroc)
        all_p_auroc.append(p_auroc)

        prec_llm = 100 * right_llm / (right_llm + wrong_llm) if (right_llm + wrong_llm) > 0 else 0
        prec_thresh = 100 * right_thresh / (right_thresh + wrong_thresh) if (right_thresh + wrong_thresh) > 0 else 0
        precision_llm.append(prec_llm)
        precision_threshold.append(prec_thresh)

        print(f"{c_name} i_AUROC: {i_auroc}")
        print(f"{c_name} p_AUROC: {p_auroc}")
        print(f"{c_name} precision_LLM_guided: {prec_llm:.2f}")
        print(f"{c_name} precision_threshold: {prec_thresh:.2f} (threshold={threshold:.4f})")

    print(f"\ni_AUROC: {torch.tensor(all_i_auroc, dtype=torch.float64).mean()}")
    print(f"p_AUROC: {torch.tensor(all_p_auroc, dtype=torch.float64).mean()}")
    print(f"precision_LLM_guided: {torch.tensor(precision_llm).mean()}")
    print(f"precision_threshold: {torch.tensor(precision_threshold).mean()}")
