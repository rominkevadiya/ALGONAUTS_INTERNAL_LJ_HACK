# 04. Training and Evaluation — SignalScope

## 1. Training Setup & Hyperparameters

Forensic execution details extracted from `notebook/SignalScope_CIFAKE_ResNet50_Native32.ipynb`:

- **Batch Size**: 32
- **Initial Learning Rate**: $1 \times 10^{-4}$
- **Optimizer**: `AdamW` (weight decay $1 \times 10^{-4}$)
- **Scheduler**: `ReduceLROnPlateau` (factor=0.1, patience=2 on validation loss)
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
| **Sensitivity (Recall REAL)** | `0.970000` | **97.00%** |
| **Specificity (Recall FAKE)** | `0.978300` | **97.83%** |
| **Test Loss** | `0.076865` | **0.0769** |

---

## 3. Metric Formulas & Calculations

- **Positive Class**: Class 1 (`REAL`)
- **Negative Class**: Class 0 (`FAKE`)

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} = \frac{9700 + 9783}{20000} = 0.97415$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{9700}{9700 + 217} = 0.97812$$

$$\text{Recall (Sensitivity)} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{9700}{9700 + 300} = 0.97000$$

$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \cdot \frac{0.97812 \cdot 0.97000}{0.97812 + 0.97000} = 0.97404$$

---

## 4. Confusion Matrix Breakdown

```
                       Confusion Matrix (CIFAKE Test Set)
                       
                         Predicted FAKE (0)    Predicted REAL (1)
    True FAKE (Class 0)       9,783 (TN)           217 (FP)
    True REAL (Class 1)         300 (FN)         9,700 (TP)
```

- **True Positives (TP)**: 9,700 (REAL photos correctly predicted as REAL)
- **True Negatives (TN)**: 9,783 (FAKE images correctly predicted as FAKE)
- **False Positives (FP)**: 217 (FAKE images misclassified as REAL)
- **False Negatives (FN)**: 300 (REAL photos misclassified as FAKE)

> [!NOTE]
> All metrics reported above are benchmark evaluations obtained from `evaluation/test_predictions.csv` on the CIFAKE test dataset and do not represent guaranteed accuracy on unrestricted real-world digital camera photographs.
