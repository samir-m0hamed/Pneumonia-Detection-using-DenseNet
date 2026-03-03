# Pneumonia Detection via DenseNet

Deep learning-based medical image classification system for detecting Pneumonia from Chest X-ray images using **DenseNet121** with Transfer Learning and Fine-Tuning.

---

## 📌 Project Overview

This project builds a high-performance binary classifier to distinguish between:

- NORMAL
- PNEUMONIA

The model leverages **DenseNet121 pretrained on ImageNet**, followed by:

1. Feature Extraction Phase  
2. Fine-Tuning Phase  
3. ROC & AUC Evaluation  
4. Threshold Optimization  
5. Confusion Matrix Analysis  

---

## 🧠 Model Architecture

- Base Model: DenseNet121 (Pretrained on ImageNet)
- Input Size: 224×224
- Global Average Pooling
- Fully Connected Layer
- Sigmoid Activation (Binary Classification)

---

## 🚀 Training Strategy

### Phase 1 — Feature Extraction
- Frozen DenseNet backbone
- Train classifier head only

### Phase 2 — Fine-Tuning
- Unfreeze upper DenseNet layers
- Lower learning rate
- Improve representation learning

---

# 📊 Results

---

## 🔹 ROC Curve

![ROC Curve](Pneumonia%20Detection%20via%20DenseNet/AUC.png)

### ROC Performance

| Metric | Value |
|--------|--------|
| AUC Score | **0.992** |

The model demonstrates excellent class separability.

---

## 🔹 Confusion Matrix

![Confusion Matrix](Pneumonia%20Detection%20via%20DenseNet/confusion%20matrix.png)

### Confusion Matrix Values

| Actual \ Predicted | NORMAL | PNEUMONIA |
|--------------------|--------|-----------|
| NORMAL             | 81     | 1         |
| PNEUMONIA          | 33     | 192       |

### Derived Metrics (Threshold = 0.5)

| Metric     | Value |
|------------|--------|
| Accuracy   | 0.89 |
| Precision  | 0.99 |
| Recall     | 0.85 |
| F1 Score   | 0.91 |

---

## 🔹 Phase 1 — Feature Extraction Results

![Feature Extraction](Pneumonia%20Detection%20via%20DenseNet/feature%20extraction%20results.png)

### Final Epoch Metrics

| Metric | Train | Validation |
|--------|--------|------------|
| Loss   | ~0.09 | ~0.26 |
| AUC    | ~0.995 | ~0.988 |
| Accuracy | ~0.965 | ~0.90 |

---

## 🔹 Phase 2 — Fine-Tuning Results

![Fine Tuning](Pneumonia%20Detection%20via%20DenseNet/fine%20tuning%20results.png)

### Final Epoch Metrics

| Metric | Train | Validation |
|--------|--------|------------|
| Loss   | ~0.08 | ~0.27 |
| AUC    | ~0.996 | ~0.985 |
| Accuracy | ~0.968 | ~0.90 |

---

## 🔹 Threshold Optimization

![Threshold Optimization](Pneumonia%20Detection%20via%20DenseNet/threashold%20fine%20tuning.png)

### Threshold vs Metrics

| Threshold | Recall | Precision | F1 |
|-----------|--------|-----------|-----|
| 0.3 | High | Slightly Lower | Strong |
| 0.5 | Balanced | Very High | Optimal |
| 0.7 | Lower | Extremely High | Drops |

Recommended Threshold: **0.3**

---

# 📈 Key Insights

- AUC = 0.992 indicates near-perfect class separability
- Very high Precision (minimal false positives)
- Moderate Recall (some false negatives remain)
- Clinically promising but recall optimization is recommended

---

# 🛠️ Tech Stack

- Python
- keras - Tensorflow
- NumPy
- Scikit-Learn
- Matplotlib
- Plotly

---

# 📂 Folder Structure

```
Pneumonia Detection via DenseNet/
│
├── Pneumonia_Detection_via_DenseNet.ipynb
│
├── AUC.png
├── confusion matrix.png
├── feature extraction results.png
├── fine tuning results.png
├── threashold fine tuning.png
│
└── README.md
```

---

# ⚠️ Notes

- Dataset: Chest X-Ray Pneumonia Dataset
- Images resized to 224×224
- Normalized using ImageNet statistics
- Default threshold = 0.5

---

# 👨‍💻 Author

**Samir Mohamed, AI & Computer Vision Engineer** .
