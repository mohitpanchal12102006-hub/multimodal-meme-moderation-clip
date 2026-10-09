"""
Multimodal Content Moderation Web Demo
Powered by OpenAI CLIP (Contrastive Language-Image Pretraining)
"""

import os
import sys

# Check for required packages gracefully before importing
missing_packages = []
try:
    import torch
except ImportError:
    missing_packages.append("torch")

try:
    import torchvision
except ImportError:
    missing_packages.append("torchvision")

try:
    import numpy as np
except ImportError:
    missing_packages.append("numpy")

try:
    from PIL import Image
except ImportError:
    missing_packages.append("pillow")

try:
    import gradio as gr
except ImportError:
    missing_packages.append("gradio")

try:
    import joblib
except ImportError:
    missing_packages.append("joblib")

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    missing_packages.append("transformers")

if missing_packages:
    print("=" * 70)
    print("[!] MISSING PYTHON PACKAGES DETECTED FOR LOCAL EXECUTION")
    print("=" * 70)
    print("The following required libraries are not installed in your Python environment:")
    for pkg in missing_packages:
        print(f"  - {pkg}")
    print("\nTo fix this and run the app locally, install them using:")
    print("    pip install -r requirements.txt")
    print("\nOr install them specifically for Python 3.11 (recommended for PyTorch on Windows):")
    print("    py -3.11 -m pip install -r requirements.txt")
    print("    py -3.11 app.py")
    print("=" * 70)
    sys.exit(1)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Starting Meme Moderation Demo on device: {device}")

# Model paths and defaults
MODEL_BUNDLE_PATH = "clip_classifier.pkl"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

print(f"[*] Loading pretrained Vision-Language Model: {CLIP_MODEL_NAME}...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()
print("[+] CLIP Model loaded successfully.")

# Load trained classifier head if present
classifier_bundle = None
if os.path.exists(MODEL_BUNDLE_PATH):
    try:
        classifier_bundle = joblib.load(MODEL_BUNDLE_PATH)
        print(f"[+] Loaded trained classifier head from {MODEL_BUNDLE_PATH}")
    except Exception as e:
        print(f"[!] Warning: Could not load {MODEL_BUNDLE_PATH}: {e}")
else:
    print(f"[*] Note: '{MODEL_BUNDLE_PATH}' not found. Using Zero-Shot CLIP classification mode.")

CANDIDATE_PROMPTS = [
    "a harmless, non-offensive, humorous meme",
    "an offensive, hateful, derogatory, or harmful meme"
]
CLASSES = ["Not Offensive", "Offensive"]

def classify_meme(image, meme_text):
    if image is None:
        return "<b>Please upload an image first.</b>", {}, "N/A", "Please provide a meme image to analyze."

    text_input = str(meme_text).strip() if meme_text and str(meme_text).strip() else "meme"

    # Convert image format
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")

    # Extract CLIP Embeddings
    inputs = clip_processor(
        text=[text_input],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        img_norm = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_norm = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)

        # Cross-modal cosine similarity (-1 to 1)
        cross_modal_sim = float((img_norm * txt_norm).sum().cpu().item())
        fused = torch.cat([img_norm, txt_norm], dim=-1).cpu().numpy()

    # Classification logic
    if classifier_bundle is not None:
        clf = classifier_bundle.get("classifier", classifier_bundle)
        bundle_classes = classifier_bundle.get("classes", CLASSES)

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(fused)[0]
        else:
            decision = clf.decision_function(fused)[0]
            p1 = 1 / (1 + np.exp(-decision))
            probs = np.array([1 - p1, p1])

        clean_classes = [c.replace('_', ' ').title() for c in bundle_classes]
        conf_dict = {clean_classes[i]: float(probs[i]) for i in range(len(clean_classes))}
        pred_idx = int(np.argmax(probs))
        pred_label = clean_classes[pred_idx]
        conf_score = float(probs[pred_idx])
        mode_used = "Trained Classifier Head (Linear/MLP Probe)"
    else:
        # Zero-shot CLIP fallback
        zero_inputs = clip_processor(
            text=CANDIDATE_PROMPTS,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            zero_out = clip_model(**zero_inputs)
            logits_per_image = zero_out.logits_per_image
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        conf_dict = {
            "Not Offensive": float(probs[0]),
            "Offensive": float(probs[1])
        }
        pred_idx = int(np.argmax(probs))
        pred_label = CLASSES[pred_idx]
        conf_score = float(probs[pred_idx])
        mode_used = "Zero-Shot CLIP Classification (No .pkl needed)"

    # Verdict formatting
    is_offensive = "offensive" in pred_label.lower() and "not" not in pred_label.lower()
    if is_offensive:
        verdict_badge = f"""
        <div style="background-color: #ffebee; border-left: 6px solid #d32f2f; padding: 15px; border-radius: 8px;">
            <h2 style="color: #c62828; margin: 0 0 8px 0;">FLAGGED AS OFFENSIVE / HARMFUL</h2>
            <p style="margin: 0; color: #37474f; font-size: 16px;">
                <b>Policy Violation:</b> This meme was classified as <b>{pred_label}</b> with <b>{conf_score*100:.1f}%</b> confidence.
            </p>
        </div>
        """
    else:
        verdict_badge = f"""
        <div style="background-color: #e8f5e9; border-left: 6px solid #2e7d32; padding: 15px; border-radius: 8px;">
            <h2 style="color: #2e7d32; margin: 0 0 8px 0;">APPROVED (NOT OFFENSIVE)</h2>
            <p style="margin: 0; color: #37474f; font-size: 16px;">
                <b>Content Safe:</b> Classified as <b>{pred_label}</b> with <b>{conf_score*100:.1f}%</b> confidence.
            </p>
        </div>
        """

    # Cross-modal alignment interpretation
    if cross_modal_sim > 0.25:
        alignment_note = f"High Image-Text Alignment ({cross_modal_sim:.3f}). Caption directly mirrors the visual depiction."
    elif cross_modal_sim < 0.10:
        alignment_note = f"Dissonant Image-Text Alignment ({cross_modal_sim:.3f}). Suggests heavy sarcasm, irony, or metaphorical juxtaposition."
    else:
        alignment_note = f"Moderate Alignment ({cross_modal_sim:.3f}). Standard meme contextual pairing."

    technical_details = f"""
    - **Inference Mode:** {mode_used}
    - **Vision-Language Model:** OpenAI CLIP (ViT-B/32)
    - **Image Embedding Dimension:** 512
    - **Text Embedding Dimension:** 512
    - **Multimodal Fused Dimension:** 1024
    - **Cross-Modal Cosine Alignment:** `{cross_modal_sim:.4f}` ({alignment_note})
    """

    return verdict_badge, conf_dict, f"{cross_modal_sim:.3f}", technical_details

# Custom CSS for modern UI
custom_css = """
body { font-family: 'Inter', -apple-system, sans-serif; }
.header-box { text-align: center; margin-bottom: 20px; }
.header-box h1 { font-size: 2.2rem; margin-bottom: 5px; color: #1e293b; }
.header-box p { font-size: 1.05rem; color: #64748b; }
"""

with gr.Blocks(title="Meme VLM Content Moderator", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
    <div class="header-box">
        <h1>Multimodal Meme Moderation System</h1>
        <p>Vision-Language AI for Harmful & Offensive Content Detection in Memes (OpenAI CLIP + Classifier Head)</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Meme Image", height=320)
            input_text = gr.Textbox(
                label="Meme Caption / OCR Text",
                placeholder="Enter the meme text or caption overlay...",
                lines=3
            )
            analyze_btn = gr.Button("Classify Meme", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_verdict = gr.HTML(label="Moderation Verdict")
            output_probs = gr.Label(label="Class Probability Distribution", num_top_classes=4)
            with gr.Accordion("Cross-Modal Explainability & Technical Diagnostics", open=True):
                output_alignment = gr.Textbox(label="Image-Text Cosine Alignment Score (-1 to 1)", interactive=False)
                output_details = gr.Markdown()

    analyze_btn.click(
        fn=classify_meme,
        inputs=[input_image, input_text],
        outputs=[output_verdict, output_probs, output_alignment, output_details]
    )

    gr.Markdown("---")
    gr.Markdown("### 💡 Demonstration & Usage Tips:")
    gr.Markdown("""
    * **Multimodal Context:** Test benign text with an incongruous image (or vice versa) to observe how multimodal fusion detects violations that unimodal filters miss.
    * **Model Loading:** If `clip_classifier.pkl` (trained in `meme_vlm_project.ipynb`) is present, it will automatically use the trained classifier head. Otherwise, it defaults to CLIP Zero-Shot classification.
    """)

if __name__ == "__main__":
    demo.launch(share=False)
