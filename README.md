# 🛡️ Multimodal Content Moderation with Vision-Language Models (CLIP)

> **Multimodal Machine Learning Pipeline** | Powered by OpenAI CLIP & Memotion 7k

---

## 📌 Project Overview
Memes pose a unique challenge for automated content moderation because they rely on **cross-modal interaction**:
- A meme's text might be harmless in isolation (e.g., *"Look at this creature"*).
- The image alone might be harmless (e.g., a photo of an animal).
- **Together**, their juxtaposition can become derogatory or offensive.

This project implements an end-to-end multimodal classification pipeline using **OpenAI's CLIP (Contrastive Language-Image Pretraining)** vision-language model (`clip-vit-base-patch32`) to jointly encode image and text into a 1024-dimensional feature vector, paired with a trained classifier head.

---

## 🚀 Key Fixes & Enhancements Made to the Pipeline

| Section | Issue Addressed | Solution Implemented |
|---|---|---|
| **Path Resolution** | Hardcoded paths failed with `FileNotFoundError` | Added automatic path detection for both local and Colab environments |
| **Label Formulation** | Severe class imbalance in 4-class labels | Implemented standard binary content moderation (`not_offensive` vs. `offensive`) triage |
| **Feature Extraction** | Slow single-sample loop and truncated image errors | Implemented batched GPU extraction (32 images/batch) with corrupted image handling |
| **Classifier Heads** | Basic MLP without balancing | Added both **Linear Probe (Logistic Regression)** and **MLP Classifier** with class weighting |
| **Evaluation & Visuals** | Unlabeled confusion matrix | Added annotated Confusion Matrices, Metric Comparison Bar Charts, and **Visual Error Analysis** |
| **Explainability** | Lack of semantic dissonance metrics | Added **Image-Text Cross-Modal Cosine Similarity** analysis |
| **Model Export** | Incomplete classifier saving | Saves complete bundle (`classifier`, `label_encoder`, `classes`, metadata) in `clip_classifier.pkl` |
| **Interactive Demo** | No user-facing application | Built standalone Gradio web app (`app.py`) with zero-shot fallback |

---

## 📂 Project Structure

```
Meme Classification/
│
├── meme_vlm_project.ipynb       # Complete, runnable Jupyter notebook (Training & Evaluation)
├── app.py                       # Standalone Gradio web demo (loads trained model or zero-shot CLIP)
├── demo_test_images/            # Curated test images representing multimodal edge cases
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ How to Run

### Option A: Run in Google Colab (Recommended for Training)
1. Upload [meme_vlm_project.ipynb](file:///c:/Users/Mohit/Meme%20Classification/meme_vlm_project.ipynb) to Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run all cells sequentially.
4. The notebook trains the model, generates visualizations, and launches an embedded demo.
5. Download `clip_classifier.pkl` from the Files panel in Colab.

### Option B: Run the Web Demo Locally (`app.py`)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place `clip_classifier.pkl` (downloaded from Colab) into the root folder. *(If not present, `app.py` automatically falls back to Zero-Shot CLIP mode!)*
3. Run the application:
   ```bash
   python app.py
   ```
4. Open the displayed local URL (e.g. `http://127.0.0.1:7860`) in your browser.

---

## 💡 Architecture & Key Design Decisions

1. **Why CLIP over separate CNN and BERT?**
   - Traditional pipelines train image and text encoders independently without aligning their vector spaces. CLIP was pretrained on 400M (image, text) pairs via contrastive learning, creating a **shared geometric space** where image and text embeddings directly interact.

2. **Why freeze the CLIP backbone instead of full fine-tuning?**
   - Fine-tuning a 150M parameter model on ~7,000 memes easily leads to catastrophic overfitting. Freezing CLIP preserves rich, generalizable representations, while training a linear probe or MLP head takes seconds and requires minimal compute.

3. **Why binary moderation instead of 4 classes?**
   - In real-world content moderation, systems first perform rapid binary triage (safe vs flagged). Additionally, severe class imbalance in extreme labels is mitigated, ensuring reliable precision and recall.

4. **Multimodal Vector Construction:**
   - Image embedding $v_i \in \mathbb{R}^{512}$ (ViT) and Text embedding $v_t \in \mathbb{R}^{512}$ (Transformer) are $L_2$-normalized and concatenated into a unified $v_{fused} \in \mathbb{R}^{1024}$ representation.
