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

describles_screw = "This is a photo of a screw for anomaly detection, which tail should be sharp, and without any damage, flaw, defect, scratch, hole or broken part."

# k=1 for screw
normal_img_paths = ["/workspace/Master-Thesis/data/mvtec_anomaly_detection/screw/train/good/012.png"]

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

root = '/workspace/Master-Thesis/data/mvtec_anomaly_detection/screw/test'
results = defaultdict(list)

for subdir in sorted(os.listdir(root)):
    subdir_path = os.path.join(root, subdir)
    if not os.path.isdir(subdir_path):
        continue
    for f in sorted(os.listdir(subdir_path)):
        if not f.endswith('.png'):
            continue
        fpath = os.path.join(subdir_path, f)
        prompt = f"This is a screw. {describles_screw} Is there any anomaly in the image?"
        resp, amap = predict(prompt, fpath, normal_img_paths)
        score = amap.reshape(224,224).detach().cpu().numpy().max()
        is_normal = subdir == 'good'
        pred_anomaly = 'Yes' in resp
        correct = (is_normal and not pred_anomaly) or (not is_normal and pred_anomaly)
        results[subdir].append({
            'file': f, 'score': float(score), 'resp': resp[:80],
            'correct': correct, 'pred_anomaly': pred_anomaly
        })
        status = '✓' if correct else '✗'
        print(f"[{status}] {subdir}/{f} score={score:.4f} pred={'Yes' if pred_anomaly else 'No'}")

print("\n=== Summary by anomaly type ===")
for atype, items in results.items():
    total = len(items)
    correct = sum(1 for x in items if x['correct'])
    scores = [x['score'] for x in items]
    print(f"{atype}: {correct}/{total} correct ({100*correct/total:.1f}%), score mean={np.mean(scores):.4f}, std={np.std(scores):.4f}, min={np.min(scores):.4f}, max={np.max(scores):.4f}")
    # Show wrong cases
    wrong = [x for x in items if not x['correct']]
    for w in wrong:
        print(f"  WRONG: {w['file']} score={w['score']:.4f} pred={'Yes' if w['pred_anomaly'] else 'No'}")
