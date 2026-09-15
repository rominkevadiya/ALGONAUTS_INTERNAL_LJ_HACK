# 03. Dataset and Model Specifications — SignalScope

## 1. Dataset Documentation

### Dataset Identity & Provenance
- **Dataset Name**: CIFAKE: Real and AI-Generated Synthetic Images
- **Kaggle Source**: `birdy666/cifake-real-and-ai-generated-synthetic-images`
- **Citation**: Bird, J. J., & Lotfi, A. (2023). *CIFAKE: Real and AI-Generated Synthetic Images*. Kaggle.
- **Native Resolution**: $32 \times 32$ pixels (upscaled to $224 \times 224$ for ResNet-50 input)
- **Image Formats**: JPEG / PNG

---

### Dataset Image Counts & Split Partitioning

The dataset comprises **120,000 total images** evenly distributed across two binary classes:

| Subset | Total Images | FAKE Images (Class 0) | REAL Images (Class 1) | Selection / Split Method |
| :--- | :---: | :---: | :---: | :--- |
| **Training Set** | **80,000** | 40,000 | 40,000 | 80% random split of 100k pool (`SEED=42`) |
| **Validation Set** | **20,000** | 10,000 | 10,000 | 20% random split of 100k pool (`SEED=42`) |
| **Test Set** | **20,000** | 10,000 | 10,000 | Official untouched Kaggle test set |
| **Total** | **120,000** | **60,000** | **60,000** | **Complete Dataset Count** |

---

### Class Index Mapping & Verification
When PyTorch `ImageFolder` loads dataset directories alphabetically:
- `train/FAKE` $\rightarrow$ **Index 0 (`FAKE`)**
- `train/REAL` $\rightarrow$ **Index 1 (`REAL`)**

This mapping is verified by checkpoint metadata: `class_to_idx: {'FAKE': 0, 'REAL': 1}`.

---

### Image Transformations Pipeline
- **Training Transformations**: `Resize((224, 224))`, `RandomHorizontalFlip()`, `RandomRotation(10)`, `ColorJitter(0.2)`, `ToTensor()`, `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
- **Evaluation Transformations**: `Resize((224, 224))`, `ToTensor()`, `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.

---

## 2. Model Architecture & Checkpoint Metadata

### Network Architecture
- **Backbone**: ResNet-50 (50-layer Residual Neural Network)
- **Pretrained Weights**: Fine-tuned from ImageNet (`ResNet50_Weights.DEFAULT`)
- **CIFAR-Adapted Stem**: The standard ImageNet stem (`Conv2d(3, 64, kernel_size=7, stride=2, padding=3)` + `MaxPool2d(3,2,1)`) is **replaced** by:
  - `Conv2d(3, 64, kernel_size=3, stride=1, padding=1)` — small kernel for 32×32 native resolution
  - `nn.Identity()` — no spatial downsampling (replaces MaxPool2d)
  - This allows native 32×32 CIFAR images to traverse all 4 residual stages without collapsing to 1×1 spatial maps.
  - Detected and applied automatically by `model_loader.py` at load time.
- **Classifier Head**: Replaced final linear layer (`nn.Linear(in_features=2048, out_features=2)`)
- **Total Parameters**: **23,565,303 parameters** (23.57 million weights)
- **Output Classes**: 2 (`FAKE` vs `REAL`)

> [!NOTE]
> **Native 32×32 patch inference**: In Patch, Hybrid, and MultiScale strategies, each 32×32 crop is passed **directly** to the model without resizing to 224×224. The CIFAR-adapted stem is specifically designed for this — the model operates at native CIFAR resolution during inference. Only the Resize strategy performs bicubic downsampling of the full image.

---

### Checkpoint Metadata & Serialization
- **Checkpoint Location**: `model/best_resnet50_cifake_retrained.pth`
- **Exact File Size**: **~90 MB** (`94,343,171 bytes`)
- **Top-Level Checkpoint Keys**: `['model_state_dict', 'class_names', 'class_to_idx', 'img_size']`
- **Optimizer & Loss**: Trained using `AdamW` (learning rate $1 \times 10^{-4}$, weight decay $1 \times 10^{-4}$) and `CrossEntropyLoss`.
- **CIFAR-Adapted Stem**: Uses `Conv2d(3, 64, kernel_size=3, stride=1, padding=1)` and `nn.Identity()` maxpool instead of the standard ImageNet `7×7` stem, matched automatically by `model_loader.py`.
- **Checkpoint Security**: Loaded with `weights_only=True`. A comprehensive numpy dtype allowlist (`numpy.dtype`, `numpy._core.multiarray.scalar`, all `numpy.dtypes.*` concrete types) is registered at import time, satisfying PyTorch 2.6+ security requirements.
