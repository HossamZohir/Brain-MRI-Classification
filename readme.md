# 🧠 Brain MRI Tumor Classification with EfficientNetV2, ConvNeXt, and ViT

This project presents a comparative deep learning pipeline for multi-class classification of brain MRI scans using three powerful pretrained models:

- ✅ EfficientNetV2B2 (TensorFlow/Keras)
- ✅ ConvNeXt-Tiny (PyTorch)
- ✅ ViT-B/16 (Vision Transformer, HuggingFace/Transformers)

With integrated **Grad-CAM explainability** and performance metrics on the **7023-sample** brain tumor dataset (4 classes: Glioma, Meningioma, Pituitary, Healthy).

---

## 📁 Dataset

- Source: Brain MRI dataset with 7023 images
- Classes: `Glioma`, `Meningioma`, `Pituitary`, `Healthy`
- Preprocessing:
  - Resized to 224×224 and 300×300
  - Stratified split: 70% Train / 15% Val / 15% Test
  - Data augmentation applied to training set

---

## 🧠 Deep Learning Models

| Model             | Framework     | Backbone Details       | Accuracy | Macro F1 |
|------------------|---------------|-------------------------|----------|----------|
| ConvNeXt Tiny     | PyTorch       | GAP + FC + Linear       | 99.05%   | 0.99     |
| EfficientNetV2B2  | Keras         | Dropout + Dense         | 98.86%   | 0.99     |
| ViT-B/16          | Transformers  | CLS Token + Linear Head | 98.86%   | 0.99     |

- All models pretrained on ImageNet
- Fine-tuned on brain MRI dataset

---

## 🔍 Explainability via Grad-CAM

Grad-CAM visualizations are integrated to:
- Highlight tumor regions
- Reveal model attention
- Assist clinical interpretation

<Insert Grad-CAM images or link to visuals>

---

## 📊 Performance Highlights

- All models achieved **balanced performance** across all classes
- ROC AUC & PRC > 0.99 for all categories
- Designed for potential **real-time** and **clinical integration**

---