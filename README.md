# 🌿 LeafLens — Explainable Plant Disease Detection

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.15-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Accuracy](https://img.shields.io/badge/Accuracy-98.64%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

> AI-powered plant leaf disease detection with Explainable Deep Learning (Grad-CAM) — upload a leaf image and get an instant diagnosis with visual explanation.

🔗 **Live Demo:** [leaf-lens-5tka.onrender.com](https://leaf-lens-5tka.onrender.com)
📓 **Kaggle Notebook:** [View full training notebook](https://www.kaggle.com/code/saaswatichinni/notebook2a9d676101)

---

## What it does

Upload any plant leaf image and LeafLens tells you which plant it is, what disease it has or if it is healthy, how confident the model is, and where on the leaf the model looked using Grad-CAM heatmaps.

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 98.64% |
| Macro F1 Score | 0.9801 |
| Weighted F1 | 0.9863 |
| ROC-AUC | 0.9999 |
| Best Val Accuracy | 99.04% |

Evaluated on 8,145 test images across 38 disease classes.

---

## Model Architecture

- Backbone: EfficientNet-B3 pretrained on ImageNet
- Custom head: BatchNorm + Dropout + Linear(512) + ReLU + Linear(38)
- Fine-tuning: Last 2 blocks unfrozen with layer-wise learning rates
- XAI: Grad-CAM implemented from scratch in pure PyTorch

---

## Training Details

| Component | Choice |
|---|---|
| Backbone | EfficientNet-B3 |
| Optimizer | AdamW |
| LR Schedule | Cosine Annealing |
| Loss | CrossEntropy + Label Smoothing 0.1 |
| Regularization | Dropout + BatchNorm |
| Backbone LR | 1e-5 |
| Head LR | 1e-4 |
| Early Stopping | Patience 5 |
| Epochs | 20 |

---

## Dataset

PlantVillage Dataset — 54,305 color images, 38 classes, 14 plant species including Tomato, Apple, Corn, Grape, Potato, Pepper and more. Split: 70% train, 15% val, 15% test.

---

## Explainability

This project implements Grad-CAM from scratch in pure PyTorch without any external library. Grad-CAM answers the question: which part of the leaf made the model predict this disease? Red and warm areas mean high importance to the prediction. Blue and cool areas mean low importance.

---

## Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Apple Cedar Rust | 1.0000 | 1.0000 | 1.0000 |
| Tomato Late Blight | 0.9488 | 0.9653 | 0.9570 |
| Corn Northern Blight | 0.9272 | 0.9524 | 0.9396 |
| Orange Greening | 0.9963 | 1.0000 | 0.9981 |
| Tomato Mosaic Virus | 0.9783 | 0.9783 | 0.9783 |

---

## Tech Stack

ML: PyTorch, timm, scikit-learn
Backend: FastAPI, Uvicorn
Frontend: HTML, CSS, JavaScript
Deployment: Render, Git LFS

---

## Limitations and Future Work

PlantVillage uses controlled lab backgrounds. Real farm images with natural backgrounds show lower confidence due to domain gap. Future work includes training on PlantDoc field images, two-stage YOLO plus EfficientNet pipeline, and background removal using Segment Anything Model.

---

## Run Locally

git clone https://github.com/Saaswati/leaf-lens.git
cd leaf-lens
pip install -r requirements.txt
uvicorn app.main:app --reload

Open http://127.0.0.1:8000

---

## References

- Mohanty et al. 2016 — Using deep learning for image-based plant disease detection
- Selvaraju et al. 2017 — Grad-CAM Visual Explanations from Deep Networks
- Tan and Le 2019 — EfficientNet Rethinking Model Scaling for CNNs

---

## Author

Saaswati — B.Tech CSE AI/ML, Lovely Professional University
Portfolio: saaswati.github.io
GitHub: Saaswati

Built for CSE471 — Deep Learning for Computer Vision
