import os
import torch
import numpy as np
from model.openllama import OpenLLAMAPEFTModel
from collections import defaultdict

args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': './ckpt/train_mvtec/pytorch_model.pt',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2, 'max_tgt_len': 128, 'lora_r': 32, 'lora_alpha': 32, 'lora_dropout': 0.1,
}
model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()
print('[!] model loaded')

describles = {
    'screw': "This is a photo of a screw for anomaly detection, which tail should be sharp, and without any damage, flaw, defect, scratch, hole or broken part.",
    'capsule': "This is a photo of a capsule for anomaly detection, which should be black and orange with print 500, having a smooth cylindrical shape with clean seam line, without any cracks, dents, scratches, faulty imprint, or squeezed deformation.",
    'cable': "This is a photo of three cables for anomaly detection, where all cables must be present in correct positions without swapping, each having intact insulation with no cuts or missing wires.",
}

# k_shot config (same as main test)
k_config = {'screw': 1, 'cable': 1, 'capsule': 2}
round_num = 3

def predict(input_text, image_path, normal_paths):
    response, pixel_output = model.generate({
        'prompt': input_text,
        'image_paths': [image_path],
        'audio_paths': [], 'video_paths': [], 'thermal_paths': [],
        'normal_img_paths': normal_paths,
        'top_p': 0.1, 'temperature': 1.0, 'max_tgt_len': 512,
        'modality_embeds': []
    })
    return response, pixel_output

data_root = '/workspace/Master-Thesis/data/mvtec_anomaly_detection'

for c_name in ['screw', 'capsule', 'cable']:
    print(f"\n{'='*60}")
    print(f"=== Analyzing: {c_name} ===")
    print(f"{'='*60}")
    
    k = k_config[c_name]
    normal_img_paths = [
        f"{data_root}/{c_name}/train/good/{str(round_num * 4 + i).zfill(3)}.png"
        for i in range(k)
    ]
    
    test_root = f"{data_root}/{c_name}/test"
    results = defaultdict(list)
    
    for subdir in sorted(os.listdir(test_root)):
        subdir_path = os.path.join(test_root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for f in sorted(os.listdir(subdir_path)):
            if not f.endswith('.png'):
                continue
            fpath = os.path.join(subdir_path, f)
            prompt = f"This is a {c_name.replace('_', ' ')}. {describles[c_name]} Is there any anomaly in the image?"
            resp, amap = predict(prompt, fpath, normal_img_paths)
            amap_np = amap.reshape(224,224).detach().cpu().numpy()
            score = float(amap_np.max())
            is_normal = subdir == 'good'
            pred_anomaly = 'Yes' in resp
            correct = (is_normal and not pred_anomaly) or (not is_normal and pred_anomaly)
            results[subdir].append({
                'file': f, 'score': score, 'correct': correct, 
                'pred_anomaly': pred_anomaly, 'is_normal': is_normal
            })
            status = '✓' if correct else '✗'
            print(f"[{status}] {c_name}/{subdir}/{f} score={score:.4f} pred={'Yes' if pred_anomaly else 'No'}")
    
    print(f"\n--- {c_name} Summary by anomaly type ---")
    all_normal_scores = []
    all_anomaly_scores = []
    for atype in sorted(results.keys()):
        items = results[atype]
        total = len(items)
        correct = sum(1 for x in items if x['correct'])
        scores = [x['score'] for x in items]
        if atype == 'good':
            all_normal_scores.extend(scores)
        else:
            all_anomaly_scores.extend(scores)
        print(f"  {atype}: {correct}/{total} ({100*correct/total:.1f}%) | score: mean={np.mean(scores):.4f} std={np.std(scores):.4f} min={np.min(scores):.4f} max={np.max(scores):.4f}")
        wrong = [x for x in items if not x['correct']]
        for w in wrong[:5]:  # show up to 5 wrong cases
            print(f"    WRONG: {w['file']} score={w['score']:.4f} pred={'Yes' if w['pred_anomaly'] else 'No'}")
    
    if all_normal_scores and all_anomaly_scores:
        overlap_min = max(min(all_anomaly_scores), min(all_normal_scores))
        overlap_max = min(max(all_anomaly_scores), max(all_normal_scores))
        print(f"\n  Normal scores:  mean={np.mean(all_normal_scores):.4f} range=[{np.min(all_normal_scores):.4f}, {np.max(all_normal_scores):.4f}]")
        print(f"  Anomaly scores: mean={np.mean(all_anomaly_scores):.4f} range=[{np.min(all_anomaly_scores):.4f}, {np.max(all_anomaly_scores):.4f}]")
        if overlap_min < overlap_max:
            n_overlap = sum(1 for s in all_normal_scores if overlap_min <= s <= overlap_max) + sum(1 for s in all_anomaly_scores if overlap_min <= s <= overlap_max)
            print(f"  Score OVERLAP zone: [{overlap_min:.4f}, {overlap_max:.4f}] ({n_overlap} samples in overlap)")
        else:
            print(f"  No score overlap (clean separation)")
        
        # Separability metric
        sep = (np.mean(all_anomaly_scores) - np.mean(all_normal_scores)) / (np.std(all_normal_scores) + np.std(all_anomaly_scores) + 1e-8)
        print(f"  Separability (Cohen's d approx): {sep:.2f}")

print("\n[!] Analysis complete")
