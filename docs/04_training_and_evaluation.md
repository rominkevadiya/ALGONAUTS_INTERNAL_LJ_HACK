# 04. Training and Evaluation — SignalScope

## 1. Training Setup & Hyperparameters

Forensic execution details extracted from `notebook/SignalScope_CIFAKE_ResNet50_Training.ipynb`:

- **Batch Size**: 32
- **Initial Learning Rate**: $1 \times 10^{-4}$
- **Optimizer**: `AdamW` (weight decay $1 \times 10^{-4}$)
- **Scheduler**: `ReduceLROnPlateau` (factor=0.1, patience=2 on validation loss)
- **Loss Function**: `CrossEntropyLoss`
- **Training Epochs**: 5 Epochs
- **Input Dimensions**: $224 \times 224 \times 3$
- **Training Hardware**: Google Colab Tesla T4 GPU (Execution duration: 78.00 minutes)
- **Best Model Selection**: Saved at Epoch 5 when validation accuracy peaked at **98.06%**.

---

## 2. Benchmark Test Metrics & Performance Table

Evaluated on the official untouched **CIFAKE test set** (20,000 images):

| Metric | Recalculated Score | Benchmark Value |
| :--- | :---: | :---: |
| **Test Accuracy** | `0.978650` | **97.87%** |
| **Precision (Macro)** | `0.978650` | **0.9786** |
| **Recall (Macro)** | `0.978650` | **97.87%** |
| **Macro F1-Score** | `0.978650` | **0.9786** |
| **ROC-AUC** | `0.997992` | **0.9980** |
| **PR-AUC** | `0.998072` | **0.9981** |
| **Sensitivity (Recall REAL)** | `0.978700` | **97.87%** |
| **Specificity (Recall FAKE)** | `0.978600` | **97.86%** |
| **Test Loss** | `0.057311` | **0.0573** |

---

## 3. Metric Formulas & Calculations

- **Positive Class**: Class 1 (`REAL`)
- **Negative Class**: Class 0 (`FAKE`)

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} = \frac{9787 + 9786}{20000} = 0.97865$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{9787}{9787 + 214} = 0.97860$$

$$\text{Recall (Sensitivity)} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{9787}{9787 + 213} = 0.97870$$

$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \cdot \frac{0.97860 \cdot 0.97870}{0.97860 + 0.97870} = 0.97865$$

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
