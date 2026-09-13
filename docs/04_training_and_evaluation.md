# 04. Training and Evaluation — SignalScope

## 1. Training Setup & Hyperparameters

Forensic execution details from the model's original training process:

- **Batch Size**: 32
- **Initial Learning Rate**: $1 \times 10^{-4}$
- **Optimizer**: `AdamW` (weight decay $1 \times 10^{-4}$)
- **Scheduler**: `CosineAnnealingLR` (T_max set to total epoch count)
- **Loss Function**: `CrossEntropyLoss`
- **Training Epochs**: 5 Epochs
- **Input Dimensions**: $224 \times 224 \times 3$ (Native $32 \times 32$ CIFAR resolution pipeline)
- **Training Hardware**: GPU Accelerated (PyTorch Native)
- **Best Model Selection**: Saved best ResNet-50 checkpoint based on validation loss/accuracy.

---

## 2. Benchmark Test Metrics & Performance Table

Evaluated on the official untouched **CIFAKE test set** (20,000 images):

| Metric | Recalculated Score | Benchmark Value |
| :--- | :---: | :---: |
| **Test Accuracy** | `0.974150` | **97.42%** |
| **Precision (Macro)** | `0.974183` | **0.9742** |
| **Recall (Macro)** | `0.974150` | **97.42%** |
| **Macro F1-Score** | `0.974150` | **0.9741** |
| **ROC-AUC** | `0.996407` | **0.9964** |
| **PR-AUC** | `0.996648` | **0.9966** |
| **Sensitivity (Recall REAL)** | `0.978700` | **97.87%** |
| **Specificity (Recall FAKE)** | `0.978300` | **97.83%** |
| **Balanced Accuracy** | `0.974150` | **97.42%** |
| **MCC** | `0.948333` | **0.9483** |
| **Cohen's Kappa** | `0.948300` | **0.9483** |
| **Average Confidence** | `0.977998` | **97.80%** |
| **Test Loss** | `0.076865` | **0.0769** |

---

## 3. Metric Formulas & Calculations

- **Positive Class**: Class 1 (`REAL`)
- **Negative Class**: Class 0 (`FAKE`)

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} = \frac{9787 + 9786}{20000} = \frac{19573}{20000} = 0.97415$$

$$\text{Precision (REAL)} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{9787}{9787 + 214} = 0.97866$$

$$\text{Recall / Sensitivity (REAL)} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{9787}{9787 + 213} = 0.97870$$

$$\text{Specificity (FAKE)} = \frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{9786}{9786 + 214} = 0.97860$$

$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \approx 0.97414$$

---

## 4. Confusion Matrix Breakdown

```
                       Confusion Matrix (CIFAKE Test Set)
                       
                         Predicted FAKE (0)    Predicted REAL (1)
    True FAKE (Class 0)       9,786 (TN)           214 (FP)
    True REAL (Class 1)         213 (FN)         9,787 (TP)
```

- **True Positives (TP)**: 9,787 (REAL photos correctly predicted as REAL)
- **True Negatives (TN)**: 9,786 (FAKE images correctly predicted as FAKE)
- **False Positives (FP)**: 214 (FAKE images misclassified as REAL)
- **False Negatives (FN)**: 213 (REAL photos misclassified as FAKE)

> [!NOTE]
> All metrics reported above are benchmark evaluations obtained from `evaluation/test_predictions.csv` on the CIFAKE test dataset and do not represent guaranteed accuracy on unrestricted real-world digital camera photographs.
