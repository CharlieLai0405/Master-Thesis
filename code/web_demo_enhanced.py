"""
Enhanced AnomalyGPT Web Demo
- Modern Gradio UI (ChatGPT-like)
- Class-specific prompts with dropdown
- Multi-reference support (k=1/2/4)
- Hybrid Reference Selection + Multi-scale TTA
- Anomaly map overlay on original image
"""
import gradio as gr
from model.openllama import OpenLLAMAPEFTModel
from model.prompts import get_prompts
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image as PILImage
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tempfile
import os

# ============================================================
# Model Init
# ============================================================
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': './ckpt/train_mvtec/pytorch_model_v6_best.pt',
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
print('[!] Model loaded successfully.')

# ============================================================
# Class-specific describles
# ============================================================
MVTEC_CLASSES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
                 'transistor', 'wood', 'zipper']
VISA_CLASSES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
ALL_CLASSES = MVTEC_CLASSES + VISA_CLASSES

describles = {
    'bottle': "This is a photo of a bottle for anomaly detection, which should be round, without any damage, flaw, defect, scratch, hole or broken part.",
    'cable': "This is a photo of three cables for anomaly detection, where all cables must be present in correct positions without swapping, each having intact insulation with no cuts or missing wires.",
    'capsule': "This is a photo of a capsule for anomaly detection, which should be black and orange with print 500, having a smooth cylindrical shape with clean seam line, without any cracks, dents, scratches, faulty imprint, or squeezed deformation.",
    'carpet': "This is a photo of carpet for anomaly detection, which should have a uniform woven texture with consistent color throughout, without any holes, cuts, stains, color spots, or thread irregularities.",
    'grid': "This is a photo of grid for anomaly detection, which should have uniform spacing with consistent parallel lines forming a regular pattern, without any bent wires, broken threads, missing sections, or metal contamination.",
    'hazelnut': "This is a photo of a hazelnut for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    'leather': "This is a photo of leather for anomaly detection, which should be brown and without any damage, flaw, defect, scratch, hole or broken part.",
    'metal_nut': "This is a photo of a metal nut for anomaly detection, which should be hexagonal with a smooth surface, consistent threading, and correct orientation (not flipped), without any bent edges, scratches, surface discoloration, or thread damage.",
    'pill': "This is a photo of a pill for anomaly detection, which should be white, with print 'FF' and red patterns, without any damage, flaw, defect, scratch, hole or broken part.",
    'screw': "This is a photo of a screw for anomaly detection, which tail should be sharp, and without any damage, flaw, defect, scratch, hole or broken part.",
    'tile': "This is a photo of tile for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    'toothbrush': "This is a photo of a toothbrush for anomaly detection, which should have evenly arranged bristles of uniform height and color with an intact handle, without any missing bristles, bent bristles, or defective brush head.",
    'transistor': "This is a photo of a transistor for anomaly detection, which should have three straight parallel leads, an intact plastic case with clear markings, and correct placement orientation, without any bent leads, cut leads, damaged case, or misplacement.",
    'wood': "This is a photo of wood for anomaly detection, which should be brown with patterns, without any damage, flaw, defect, scratch, hole or broken part.",
    'zipper': "This is a photo of a zipper for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    'candle': "This is a photo of a candle for anomaly detection, which should have a smooth wax surface with consistent color and no foreign objects.",
    'capsules': "This is a photo of capsules for anomaly detection, which should have intact shells with proper shape and no cracks or deformation.",
    'cashew': "This is a photo of a cashew for anomaly detection, which should have a smooth surface with natural color and no damage.",
    'chewinggum': "This is a photo of chewing gum for anomaly detection, which should have a uniform rectangular shape with smooth surface.",
    'fryum': "This is a photo of fryum for anomaly detection, which should have a consistent puffed shape with uniform color.",
    'macaroni1': "This is a photo of macaroni for anomaly detection, which should have a uniform tubular shape with smooth surface.",
    'macaroni2': "This is a photo of macaroni for anomaly detection, which should have a uniform tubular shape with smooth surface.",
    'pcb1': "This is a photo of a PCB for anomaly detection, which should have clean solder joints, intact traces, and properly placed components.",
    'pcb2': "This is a photo of a PCB for anomaly detection, which should have clean solder joints, intact traces, and properly placed components.",
    'pcb3': "This is a photo of a PCB for anomaly detection, which should have clean solder joints, intact traces, and properly placed components.",
    'pcb4': "This is a photo of a PCB for anomaly detection, which should have clean solder joints, intact traces, and properly placed components.",
    'pipe_fryum': "This is a photo of pipe fryum for anomaly detection, which should have a uniform tubular shape with consistent color.",
}


# ============================================================
# Spread Selection (for Hybrid Reference)
# ============================================================
def select_spread_references(all_paths, n_refs):
    """Select n_refs images spread in feature space (maximin diversity)."""
    if len(all_paths) <= n_refs:
        return all_paths
    from model.ImageBind import data
    from model.ImageBind.models.imagebind_model import ModalityType
    
    batch_size = 16
    all_features = []
    for i in range(0, len(all_paths), batch_size):
        batch_paths = all_paths[i:i+batch_size]
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(batch_paths, 'cuda')}
        inputs = {key: inputs[key].to(model.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = model.visual_encoder(inputs)
            cls_features = embeddings['vision'][0]
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
    
    return [all_paths[i] for i in selected]


# ============================================================
# TTA
# ============================================================
def predict_with_tta(prompt_text, image_path, normal_img_paths, use_tta=True):
    """Run prediction with optional Multi-scale TTA."""
    
    def run_single(img_path, normals):
        response, pixel_output = model.generate({
            'prompt': prompt_text,
            'image_paths': [img_path],
            'normal_img_paths': normals,
            'audio_paths': [], 'video_paths': [], 'thermal_paths': [],
            'top_p': 0.01, 'temperature': 1.0, 'max_tgt_len': 128,
            'modality_embeds': []
        }, web_demo=True)
        return response, pixel_output
    
    if not use_tta or not normal_img_paths:
        return run_single(image_path, normal_img_paths)
    
    original_img = PILImage.open(image_path).convert('RGB')
    augmentations = [(image_path, 'none')]
    temp_files = []
    
    # H-flip
    hflip = original_img.transpose(PILImage.FLIP_LEFT_RIGHT)
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    hflip.save(tmp.name); temp_files.append(tmp.name)
    augmentations.append((tmp.name, 'hflip'))
    
    # V-flip
    vflip = original_img.transpose(PILImage.FLIP_TOP_BOTTOM)
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    vflip.save(tmp.name); temp_files.append(tmp.name)
    augmentations.append((tmp.name, 'vflip'))
    
    # Rot90
    rot90 = original_img.transpose(PILImage.ROTATE_90)
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    rot90.save(tmp.name); temp_files.append(tmp.name)
    augmentations.append((tmp.name, 'rot90'))
    
    # Rot270
    rot270 = original_img.transpose(PILImage.ROTATE_270)
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    rot270.save(tmp.name); temp_files.append(tmp.name)
    augmentations.append((tmp.name, 'rot270'))
    
    maps = []
    response_text = None
    for aug_path, aug_type in augmentations:
        resp, pix = run_single(aug_path, normal_img_paths)
        if response_text is None:
            response_text = resp
        
        m = pix.to(torch.float32).reshape(224, 224)
        if aug_type == 'hflip':
            m = m.flip(1)
        elif aug_type == 'vflip':
            m = m.flip(0)
        elif aug_type == 'rot90':
            m = torch.rot90(m, -1, [0, 1])
        elif aug_type == 'rot270':
            m = torch.rot90(m, 1, [0, 1])
        maps.append(m)
    
    for f in temp_files:
        os.unlink(f)
    
    avg_map = torch.stack(maps).mean(dim=0)
    return response_text, avg_map.unsqueeze(0)


# ============================================================
# Overlay anomaly map on original image
# ============================================================
def create_overlay(image_path, anomaly_map_tensor, alpha=0.5):
    """Create a heatmap overlay on the original image."""
    original = PILImage.open(image_path).convert('RGB').resize((224, 224))
    original_np = np.array(original)
    
    amap = anomaly_map_tensor.to(torch.float32).reshape(224, 224).detach().cpu().numpy()
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    
    heatmap = plt.cm.jet(amap)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    overlay = (original_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return PILImage.fromarray(overlay)


# ============================================================
# Main predict function
# ============================================================
def predict(
    query_image,
    ref_images,
    class_name,
    user_message,
    use_tta,
    use_spread,
    chat_history,
):
    if query_image is None:
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": user_message or "..."})
        chat_history.append({"role": "assistant", "content": "⚠️ Please upload a query image first."})
        return chat_history, None, None
    
    # Build prompt
    c_name = class_name if class_name != "auto-detect" else "object"
    desc = describles.get(c_name, describles.get('object', f"This is a photo of {c_name} for anomaly detection."))
    
    if not user_message:
        user_message = f"{desc} Is there any anomaly in the image?"
    
    # Build prompt text from history
    prompt_text = ''
    if chat_history:
        for msg in chat_history:
            if msg['role'] == 'user':
                if prompt_text == '':
                    prompt_text += f'{msg["content"]}\n### Assistant:'
                else:
                    prompt_text += f' Human: {msg["content"]}\n### Assistant:'
            elif msg['role'] == 'assistant':
                prompt_text += f' {msg["content"]}\n###'
    
    if prompt_text == '':
        prompt_text = user_message
    else:
        prompt_text += f' Human: {user_message}'
    
    # Process reference images
    normal_paths = []
    if ref_images:
        if isinstance(ref_images, list):
            normal_paths = ref_images
        else:
            normal_paths = [ref_images]
        
        # Hybrid reference selection
        if use_spread and len(normal_paths) > 4:
            normal_paths = select_spread_references(normal_paths, k=min(4, len(normal_paths)))
    
    # Run inference
    response, pixel_output = predict_with_tta(
        prompt_text, query_image, normal_paths, use_tta=use_tta
    )
    
    # Create visualizations
    overlay = create_overlay(query_image, pixel_output)
    
    # Anomaly map only
    amap = pixel_output.to(torch.float32).reshape(224, 224).detach().cpu().numpy()
    plt.figure(figsize=(3, 3))
    plt.imshow(amap, cmap='jet')
    plt.axis('off')
    tmp_map = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(tmp_map.name, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    anomaly_map_img = PILImage.open(tmp_map.name)
    
    # Update chat
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": response})
    
    return chat_history, overlay, anomaly_map_img


def clear_all():
    return [], None, None, None, None, "auto-detect", "", True, True


# ============================================================
# Gradio UI
# ============================================================
custom_css = """
.gradio-container { max-width: 1400px !important; margin: auto !important; }
.chat-window { min-height: 400px !important; }
#title-bar { text-align: center; padding: 20px 0 10px 0; }
#title-bar h1 { font-size: 2em; font-weight: 700; margin: 0; }
#title-bar p { color: #666; margin: 5px 0 0 0; }
"""

with gr.Blocks(css=custom_css, title="AnomalyGPT Enhanced") as demo:
    
    gr.HTML("""
    <div id="title-bar">
        <h1>🔍 AnomalyGPT — Industrial Anomaly Detection</h1>
        <p>Upload a query image, select the product category, and optionally provide normal reference images.</p>
    </div>
    """)
    
    with gr.Row():
        # Left panel: inputs
        with gr.Column(scale=1):
            class_name = gr.Dropdown(
                choices=["auto-detect"] + ALL_CLASSES,
                value="auto-detect",
                label="📋 Product Category",
                info="Select the product type for class-specific detection"
            )
            
            query_image = gr.Image(
                type="filepath",
                label="🖼️ Query Image",
                height=250,
            )
            
            ref_images = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="📷 Normal Reference Images (optional, k=1/2/4)",
            )
            
            with gr.Row():
                use_tta = gr.Checkbox(value=True, label="🔄 Multi-Scale TTA")
                use_spread = gr.Checkbox(value=True, label="📊 Spread Selection")
        
        # Right panel: chat + results
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="💬 Conversation",
                height=350,
                type="messages",
            )
            
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Ask about the image... (leave empty for auto-detection)",
                    label="Your Message",
                    scale=4,
                )
                submit_btn = gr.Button("🚀 Analyze", variant="primary", scale=1)
            
            with gr.Row():
                overlay_output = gr.Image(
                    label="🎯 Anomaly Overlay",
                    height=250,
                )
                anomaly_map_output = gr.Image(
                    label="🗺️ Anomaly Heatmap",
                    height=250,
                )
            
            clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
    
    # Event handlers
    submit_btn.click(
        predict,
        inputs=[query_image, ref_images, class_name, user_input, use_tta, use_spread, chatbot],
        outputs=[chatbot, overlay_output, anomaly_map_output],
    )
    
    user_input.submit(
        predict,
        inputs=[query_image, ref_images, class_name, user_input, use_tta, use_spread, chatbot],
        outputs=[chatbot, overlay_output, anomaly_map_output],
    )
    
    clear_btn.click(
        clear_all,
        outputs=[chatbot, query_image, overlay_output, anomaly_map_output,
                 ref_images, class_name, user_input, use_tta, use_spread],
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
