# 04. Training and Evaluation — SignalScope

## 1. Training Setup & Hyperparameters

Forensic execution details from the model's original training process:

- **Batch Size**: 32
- **Initial Learning Rate**: $1 \times 10^{-4}$
- **Optimizer**: `AdamW` (weight decay $1 \times 10^{-4}$)
- **Scheduler**: `CosineAnnealingLR` (T_max set to total epoch count)
- **Loss Function**: `CrossEntropyLoss`
- **Training Epochs**: 8 Epochs (best checkpoint saved at Epoch 8 by validation MCC)
- **Input Dimensions**: $224 \times 224 \times 3$ (native $32 \times 32$ CIFAR resolution via CIFAR-adapted stem)
- **Training Hardware**: NVIDIA Tesla T4 GPU (Google Colab)
- **Best Model Selection**: Checkpoint saved at **Epoch 8** when validation MCC peaked at **0.9334**.

---

## 2. Benchmark Test Metrics (Independently Verified)

Evaluated on the official untouched **CIFAKE test set** (20,000 images).
All values independently recalculated from `evaluation/test_predictions.csv` and verified 100% against reported figures.

| Metric | Verified Score |
| :--- | :---: |
| **Test Accuracy** | **97.87%** |
| **Precision (Macro)** | **0.9786** |
| **Recall (Macro)** | **0.9787%** |
| **Macro F1-Score** | **0.9786** |
| **ROC-AUC** | **0.9980** |
| **PR-AUC** | **0.9981** |
| **Sensitivity (Recall REAL)** | **97.87%** |
| **Specificity (Recall FAKE)** | **97.86%** |
| **Balanced Accuracy** | **97.865%** |
| **MCC (Matthews Correlation)** | **0.9573** |
| **Cohen's Kappa** | **0.9573** |
| **Average Confidence** | **~97.8%** |

---

## 3. Metric Formulas & Calculations

- **Positive Class**: Class 1 (`REAL`)
- **Negative Class**: Class 0 (`FAKE`)

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} = \frac{9787 + 9786}{20000} = \frac{19573}{20000} = 0.97865$$

$$\text{Precision (REAL)} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{9787}{9787 + 214} = 0.97866$$

$$\text{Recall / Sensitivity (REAL)} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{9787}{9787 + 213} = 0.97870$$

$$\text{Specificity (FAKE)} = \frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{9786}{9786 + 214} = 0.97860$$

$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \approx 0.97865$$

---

## 4. Confusion Matrix Breakdown

```
                       Confusion Matrix (CIFAKE Test Set, 20,000 images)

                         Predicted FAKE (0)    Predicted REAL (1)
    True FAKE (Class 0)       9,786 (TN)           214 (FP)
    True REAL (Class 1)         213 (FN)         9,787 (TP)
```

- **True Positives (TP)**: 9,787 (REAL photos correctly predicted as REAL)
- **True Negatives (TN)**: 9,786 (FAKE images correctly predicted as FAKE)
- **False Positives (FP)**: 214 (FAKE images misclassified as REAL)
- **False Negatives (FN)**: 213 (REAL photos misclassified as FAKE)

> [!NOTE]
> All metrics are benchmark evaluations on the CIFAKE test dataset, independently verified from `evaluation/test_predictions.csv`. They do not represent guaranteed accuracy on unrestricted real-world digital camera photographs or modern AI generators not present in the CIFAKE training distribution.

> [!NOTE]
> The `evaluation/evaluation_summary.json` multi-strategy classification benchmark shows 0 samples because no externally labeled image dataset was supplied to `evaluate_strategies.py`. The robustness and determinism sections of that report are populated from synthetic test patterns.
