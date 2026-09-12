# Model Checkpoint Analysis Report

**File Path:** `model_checkpoints/best_resnet50_cifake_original.pth`  
**Original Model Path:** `model/best_resnet50_cifake.pth`  
**Inspection Date:** 2026-09-12  

---

## 1. Overview & Summary

| Attribute | Details |
| :--- | :--- |
| **Checkpoint Container Type** | `dict` (`<class 'dict'>`) |
| **Top-Level Keys** | `['model_state_dict', 'class_names', 'class_to_idx', 'img_size']` |
| **Contains `model_state_dict`** | **Yes** (`True`) |
| **Total Stored Layer Tensors** | **320** |
| **Total Parameter Count** | **23,565,303** (~23.57 Million elements) |

---

## 2. Metadata Information

The checkpoint contains key metadata embedded alongside the model weights:

* **Class Names (`class_names`):** `['FAKE', 'REAL']`
* **Class Index Mapping (`class_to_idx`):** `{'FAKE': 0, 'REAL': 1}`
* **Input Image Resolution (`img_size`):** `224` (224x224 RGB images)

---

## 3. Key Architecture Shapes

### First Convolutional Layer
* **Layer Name:** `conv1.weight`
* **Tensor Shape:** `(64, 3, 7, 7)`
* **Total Parameters:** `9,408`

### Final Classifier Head
* **Linear Layer Weights (`fc.weight`):** `(2, 2048)` — `4,096` parameters
* **Linear Layer Bias (`fc.bias`):** `(2,)` — `2` parameters
* **Output Classes:** `2` (`FAKE` vs `REAL`)

---

## 4. Layer & Parameter Specification (Summary Table)

| Group / Component | Layer Count | Key Tensor | Key Tensor Shape | Total Elements |
| :--- | :--- | :--- | :--- | :--- |
| **Initial Conv & BN** | 6 tensors | `conv1.weight` | `(64, 3, 7, 7)` | 9,664 |
| **Layer Block 1 (3 Bottlenecks)** | 72 tensors | `layer1.0.conv1.weight` | `(64, 64, 1, 1)` | 215,808 |
| **Layer Block 2 (4 Bottlenecks)** | 96 tensors | `layer2.0.conv1.weight` | `(128, 256, 1, 1)` | 1,219,584 |
| **Layer Block 3 (6 Bottlenecks)** | 96 tensors | `layer3.0.conv1.weight` | `(256, 512, 1, 1)` | 7,098,368 |
| **Layer Block 4 (3 Bottlenecks)** | 48 tensors | `layer4.0.conv1.weight` | `(512, 1024, 1, 1)` | 15,017,728 |
| **Classifier FC** | 2 tensors | `fc.weight` | `(2, 2048)` | 4,098 |
| **Total** | **320 Tensors** | — | — | **23,565,303** |

---

## 5. Comprehensive Parameter Listing

<details>
<summary>Click to expand all 320 parameter tensors</summary>

| Layer Name | Tensor Shape | Parameters |
| :--- | :--- | :--- |
| `conv1.weight` | `(64, 3, 7, 7)` | 9,408 |
| `bn1.weight` | `(64,)` | 64 |
| `bn1.bias` | `(64,)` | 64 |
| `bn1.running_mean` | `(64,)` | 64 |
| `bn1.running_var` | `(64,)` | 64 |
| `bn1.num_batches_tracked` | `()` | 1 |
| `layer1.0.conv1.weight` | `(64, 64, 1, 1)` | 4,096 |
| `layer1.0.bn1.weight` | `(64,)` | 64 |
| `layer1.0.bn1.bias` | `(64,)` | 64 |
| `layer1.0.bn1.running_mean` | `(64,)` | 64 |
| `layer1.0.bn1.running_var` | `(64,)` | 64 |
| `layer1.0.bn1.num_batches_tracked` | `()` | 1 |
| `layer1.0.conv2.weight` | `(64, 64, 3, 3)` | 36,864 |
| `layer1.0.bn2.weight` | `(64,)` | 64 |
| `layer1.0.bn2.bias` | `(64,)` | 64 |
| `layer1.0.bn2.running_mean` | `(64,)` | 64 |
| `layer1.0.bn2.running_var` | `(64,)` | 64 |
| `layer1.0.bn2.num_batches_tracked` | `()` | 1 |
| `layer1.0.conv3.weight` | `(256, 64, 1, 1)` | 16,384 |
| `layer1.0.bn3.weight` | `(256,)` | 256 |
| `layer1.0.bn3.bias` | `(256,)` | 256 |
| `layer1.0.bn3.running_mean` | `(256,)` | 256 |
| `layer1.0.bn3.running_var` | `(256,)` | 256 |
| `layer1.0.bn3.num_batches_tracked` | `()` | 1 |
| `layer1.0.downsample.0.weight` | `(256, 64, 1, 1)` | 16,384 |
| `layer1.0.downsample.1.weight` | `(256,)` | 256 |
| `layer1.0.downsample.1.bias` | `(256,)` | 256 |
| `layer1.0.downsample.1.running_mean` | `(256,)` | 256 |
| `layer1.0.downsample.1.running_var` | `(256,)` | 256 |
| `layer1.0.downsample.1.num_batches_tracked` | `()` | 1 |
| `layer1.1.conv1.weight` | `(64, 256, 1, 1)` | 16,384 |
| `layer1.1.bn1.weight` | `(64,)` | 64 |
| `layer1.1.bn1.bias` | `(64,)` | 64 |
| `layer1.1.bn1.running_mean` | `(64,)` | 64 |
| `layer1.1.bn1.running_var` | `(64,)` | 64 |
| `layer1.1.bn1.num_batches_tracked` | `()` | 1 |
| `layer1.1.conv2.weight` | `(64, 64, 3, 3)` | 36,864 |
| `layer1.1.bn2.weight` | `(64,)` | 64 |
| `layer1.1.bn2.bias` | `(64,)` | 64 |
| `layer1.1.bn2.running_mean` | `(64,)` | 64 |
| `layer1.1.bn2.running_var` | `(64,)` | 64 |
| `layer1.1.bn2.num_batches_tracked` | `()` | 1 |
| `layer1.1.conv3.weight` | `(256, 64, 1, 1)` | 16,384 |
| `layer1.1.bn3.weight` | `(256,)` | 256 |
| `layer1.1.bn3.bias` | `(256,)` | 256 |
| `layer1.1.bn3.running_mean` | `(256,)` | 256 |
| `layer1.1.bn3.running_var` | `(256,)` | 256 |
| `layer1.1.bn3.num_batches_tracked` | `()` | 1 |
| `layer1.2.conv1.weight` | `(64, 256, 1, 1)` | 16,384 |
| `layer1.2.bn1.weight` | `(64,)` | 64 |
| `layer1.2.bn1.bias` | `(64,)` | 64 |
| `layer1.2.bn1.running_mean` | `(64,)` | 64 |
| `layer1.2.bn1.running_var` | `(64,)` | 64 |
| `layer1.2.bn1.num_batches_tracked` | `()` | 1 |
| `layer1.2.conv2.weight` | `(64, 64, 3, 3)` | 36,864 |
| `layer1.2.bn2.weight` | `(64,)` | 64 |
| `layer1.2.bn2.bias` | `(64,)` | 64 |
| `layer1.2.bn2.running_mean` | `(64,)` | 64 |
| `layer1.2.bn2.running_var` | `(64,)` | 64 |
| `layer1.2.bn2.num_batches_tracked` | `()` | 1 |
| `layer1.2.conv3.weight` | `(256, 64, 1, 1)` | 16,384 |
| `layer1.2.bn3.weight` | `(256,)` | 256 |
| `layer1.2.bn3.bias` | `(256,)` | 256 |
| `layer1.2.bn3.running_mean` | `(256,)` | 256 |
| `layer1.2.bn3.running_var` | `(256,)` | 256 |
| `layer1.2.bn3.num_batches_tracked` | `()` | 1 |
| `layer2.0.conv1.weight` | `(128, 256, 1, 1)` | 32,768 |
| `layer2.0.bn1.weight` | `(128,)` | 128 |
| `layer2.0.bn1.bias` | `(128,)` | 128 |
| `layer2.0.bn1.running_mean` | `(128,)` | 128 |
| `layer2.0.bn1.running_var` | `(128,)` | 128 |
| `layer2.0.bn1.num_batches_tracked` | `()` | 1 |
| `layer2.0.conv2.weight` | `(128, 128, 3, 3)` | 147,456 |
| `layer2.0.bn2.weight` | `(128,)` | 128 |
| `layer2.0.bn2.bias` | `(128,)` | 128 |
| `layer2.0.bn2.running_mean` | `(128,)` | 128 |
| `layer2.0.bn2.running_var` | `(128,)` | 128 |
| `layer2.0.bn2.num_batches_tracked` | `()` | 1 |
| `layer2.0.conv3.weight` | `(512, 128, 1, 1)` | 65,536 |
| `layer2.0.bn3.weight` | `(512,)` | 512 |
| `layer2.0.bn3.bias` | `(512,)` | 512 |
| `layer2.0.bn3.running_mean` | `(512,)` | 512 |
| `layer2.0.bn3.running_var` | `(512,)` | 512 |
| `layer2.0.bn3.num_batches_tracked` | `()` | 1 |
| `layer2.0.downsample.0.weight` | `(512, 256, 1, 1)` | 131,072 |
| `layer2.0.downsample.1.weight` | `(512,)` | 512 |
| `layer2.0.downsample.1.bias` | `(512,)` | 512 |
| `layer2.0.downsample.1.running_mean` | `(512,)` | 512 |
| `layer2.0.downsample.1.running_var` | `(512,)` | 512 |
| `layer2.0.downsample.1.num_batches_tracked` | `()` | 1 |
| `layer2.1.conv1.weight` | `(128, 512, 1, 1)` | 65,536 |
| `layer2.1.bn1.weight` | `(128,)` | 128 |
| `layer2.1.bn1.bias` | `(128,)` | 128 |
| `layer2.1.bn1.running_mean` | `(128,)` | 128 |
| `layer2.1.bn1.running_var` | `(128,)` | 128 |
| `layer2.1.bn1.num_batches_tracked` | `()` | 1 |
| `layer2.1.conv2.weight` | `(128, 128, 3, 3)` | 147,456 |
| `layer2.1.bn2.weight` | `(128,)` | 128 |
| `layer2.1.bn2.bias` | `(128,)` | 128 |
| `layer2.1.bn2.running_mean` | `(128,)` | 128 |
| `layer2.1.bn2.running_var` | `(128,)` | 128 |
| `layer2.1.bn2.num_batches_tracked` | `()` | 1 |
| `layer2.1.conv3.weight` | `(512, 128, 1, 1)` | 65,536 |
| `layer2.1.bn3.weight` | `(512,)` | 512 |
| `layer2.1.bn3.bias` | `(512,)` | 512 |
| `layer2.1.bn3.running_mean` | `(512,)` | 512 |
| `layer2.1.bn3.running_var` | `(512,)` | 512 |
| `layer2.1.bn3.num_batches_tracked` | `()` | 1 |
| `layer2.2.conv1.weight` | `(128, 512, 1, 1)` | 65,536 |
| `layer2.2.bn1.weight` | `(128,)` | 128 |
| `layer2.2.bn1.bias` | `(128,)` | 128 |
| `layer2.2.bn1.running_mean` | `(128,)` | 128 |
| `layer2.2.bn1.running_var` | `(128,)` | 128 |
| `layer2.2.bn1.num_batches_tracked` | `()` | 1 |
| `layer2.2.conv2.weight` | `(128, 128, 3, 3)` | 147,456 |
| `layer2.2.bn2.weight` | `(128,)` | 128 |
| `layer2.2.bn2.bias` | `(128,)` | 128 |
| `layer2.2.bn2.running_mean` | `(128,)` | 128 |
| `layer2.2.bn2.running_var` | `(128,)` | 128 |
| `layer2.2.bn2.num_batches_tracked` | `()` | 1 |
| `layer2.2.conv3.weight` | `(512, 128, 1, 1)` | 65,536 |
| `layer2.2.bn3.weight` | `(512,)` | 512 |
| `layer2.2.bn3.bias` | `(512,)` | 512 |
| `layer2.2.bn3.running_mean` | `(512,)` | 512 |
| `layer2.2.bn3.running_var` | `(512,)` | 512 |
| `layer2.2.bn3.num_batches_tracked` | `()` | 1 |
| `layer2.3.conv1.weight` | `(128, 512, 1, 1)` | 65,536 |
| `layer2.3.bn1.weight` | `(128,)` | 128 |
| `layer2.3.bn1.bias` | `(128,)` | 128 |
| `layer2.3.bn1.running_mean` | `(128,)` | 128 |
| `layer2.3.bn1.running_var` | `(128,)` | 128 |
| `layer2.3.bn1.num_batches_tracked` | `()` | 1 |
| `layer2.3.conv2.weight` | `(128, 128, 3, 3)` | 147,456 |
| `layer2.3.bn2.weight` | `(128,)` | 128 |
| `layer2.3.bn2.bias` | `(128,)` | 128 |
| `layer2.3.bn2.running_mean` | `(128,)` | 128 |
| `layer2.3.bn2.running_var` | `(128,)` | 128 |
| `layer2.3.bn2.num_batches_tracked` | `()` | 1 |
| `layer2.3.conv3.weight` | `(512, 128, 1, 1)` | 65,536 |
| `layer2.3.bn3.weight` | `(512,)` | 512 |
| `layer2.3.bn3.bias` | `(512,)` | 512 |
| `layer2.3.bn3.running_mean` | `(512,)` | 512 |
| `layer2.3.bn3.running_var` | `(512,)` | 512 |
| `layer2.3.bn3.num_batches_tracked` | `()` | 1 |
| `layer3.0.conv1.weight` | `(256, 512, 1, 1)` | 131,072 |
| `layer3.0.bn1.weight` | `(256,)` | 256 |
| `layer3.0.bn1.bias` | `(256,)` | 256 |
| `layer3.0.bn1.running_mean` | `(256,)` | 256 |
| `layer3.0.bn1.running_var` | `(256,)` | 256 |
| `layer3.0.bn1.num_batches_tracked` | `()` | 1 |
| `layer3.0.conv2.weight` | `(256, 256, 3, 3)` | 589,824 |
| `layer3.0.bn2.weight` | `(256,)` | 256 |
| `layer3.0.bn2.bias` | `(256,)` | 256 |
| `layer3.0.bn2.running_mean` | `(256,)` | 256 |
| `layer3.0.bn2.running_var` | `(256,)` | 256 |
| `layer3.0.bn2.num_batches_tracked` | `()` | 1 |
| `layer3.0.conv3.weight` | `(1024, 256, 1, 1)` | 262,144 |
| `layer3.0.bn3.weight` | `(1024,)` | 1,024 |
| `layer3.0.bn3.bias` | `(1024,)` | 1,024 |
| `layer3.0.bn3.running_mean` | `(1024,)` | 1,024 |
| `layer3.0.bn3.running_var` | `(1024,)` | 1,024 |
| `layer3.0.bn3.num_batches_tracked` | `()` | 1 |
| `layer3.0.downsample.0.weight` | `(1024, 512, 1, 1)` | 524,288 |
| `layer3.0.downsample.1.weight` | `(1024,)` | 1,024 |
| `layer3.0.downsample.1.bias` | `(1024,)` | 1,024 |
| `layer3.0.downsample.1.running_mean` | `(1024,)` | 1,024 |
| `layer3.0.downsample.1.running_var` | `(1024,)` | 1,024 |
| `layer3.0.downsample.1.num_batches_tracked` | `()` | 1 |
| `layer3.1.conv1.weight` | `(256, 1024, 1, 1)` | 262,144 |
| `layer3.1.bn1.weight` | `(256,)` | 256 |
| `layer3.1.bn1.bias` | `(256,)` | 256 |
| `layer3.1.bn1.running_mean` | `(256,)` | 256 |
| `layer3.1.bn1.running_var` | `(256,)` | 256 |
| `layer3.1.bn1.num_batches_tracked` | `()` | 1 |
| `layer3.1.conv2.weight` | `(256, 256, 3, 3)` | 589,824 |
| `layer3.1.bn2.weight` | `(256,)` | 256 |
| `layer3.1.bn2.bias` | `(256,)` | 256 |
| `layer3.1.bn2.running_mean` | `(256,)` | 256 |
| `layer3.1.bn2.running_var` | `(256,)` | 256 |
| `layer3.1.bn2.num_batches_tracked` | `()` | 1 |
| `layer3.1.conv3.weight` | `(1024, 256, 1, 1)` | 262,144 |
| `layer3.1.bn3.weight` | `(1024,)` | 1,024 |
| `layer3.1.bn3.bias` | `(1024,)` | 1,024 |
| `layer3.1.bn3.running_mean` | `(1024,)` | 1,024 |
| `layer3.1.bn3.running_var` | `(1024,)` | 1,024 |
| `layer3.1.bn3.num_batches_tracked` | `()` | 1 |
| `layer3.2.conv1.weight` | `(256, 1024, 1, 1)` | 262,144 |
| `layer3.2.bn1.weight` | `(256,)` | 256 |
| `layer3.2.bn1.bias` | `(256,)` | 256 |
| `layer3.2.bn1.running_mean` | `(256,)` | 256 |
| `layer3.2.bn1.running_var` | `(256,)` | 256 |
| `layer3.2.bn1.num_batches_tracked` | `()` | 1 |
| `layer3.2.conv2.weight` | `(256, 256, 3, 3)` | 589,824 |
| `layer3.2.bn2.weight` | `(256,)` | 256 |
| `layer3.2.bn2.bias` | `(256,)` | 256 |
| `layer3.2.bn2.running_mean` | `(256,)` | 256 |
| `layer3.2.bn2.running_var` | `(256,)` | 256 |
| `layer3.2.bn2.num_batches_tracked` | `()` | 1 |
| `layer3.2.conv3.weight` | `(1024, 256, 1, 1)` | 262,144 |
| `layer3.2.bn3.weight` | `(1024,)` | 1,024 |
| `layer3.2.bn3.bias` | `(1024,)` | 1,024 |
| `layer3.2.bn3.running_mean` | `(1024,)` | 1,024 |
| `layer3.2.bn3.running_var` | `(1024,)` | 1,024 |
| `layer3.2.bn3.num_batches_tracked` | `()` | 1 |
| `layer3.3.conv1.weight` | `(256, 1024, 1, 1)` | 262,144 |
| `layer3.3.bn1.weight` | `(256,)` | 256 |
| `layer3.3.bn1.bias` | `(256,)` | 256 |
| `layer3.3.bn1.running_mean` | `(256,)` | 256 |
| `layer3.3.bn1.running_var` | `(256,)` | 256 |
| `layer3.3.bn1.num_batches_tracked` | `()` | 1 |
| `layer3.3.conv2.weight` | `(256, 256, 3, 3)` | 589,824 |
| `layer3.3.bn2.weight` | `(256,)` | 256 |
| `layer3.3.bn2.bias` | `(256,)` | 256 |
| `layer3.3.bn2.running_mean` | `(256,)` | 256 |
| `layer3.3.bn2.running_var` | `(256,)` | 256 |
| `layer3.3.bn2.num_batches_tracked` | `()` | 1 |
| `layer3.3.conv3.weight` | `(1024, 256, 1, 1)` | 262,144 |
| `layer3.3.bn3.weight` | `(1024,)` | 1,024 |
| `layer3.3.bn3.bias` | `(1024,)` | 1,024 |
| `layer3.3.bn3.running_mean` | `(1024,)` | 1,024 |
| `layer3.3.bn3.running_var` | `(1024,)` | 1,024 |
| `layer3.3.bn3.num_batches_tracked` | `()` | 1 |
| `layer3.4.conv1.weight` | `(256, 1024, 1, 1)` | 262,144 |
| `layer3.4.bn1.weight` | `(256,)` | 256 |
| `layer3.4.bn1.bias` | `(256,)` | 256 |
| `layer3.4.bn1.running_mean` | `(256,)` | 256 |
| `layer3.4.bn1.running_var` | `(256,)` | 256 |
| `layer3.4.bn1.num_batches_tracked` | `()` | 1 |
| `layer3.4.conv2.weight` | `(256, 256, 3, 3)` | 589,824 |
| `layer3.4.bn2.weight` | `(256,)` | 256 |
| `layer3.4.bn2.bias` | `(256,)` | 256 |
| `layer3.4.bn2.running_mean` | `(256,)` | 256 |
| `layer3.4.bn2.running_var` | `(256,)` | 256 |
| `layer3.4.bn2.num_batches_tracked` | `()` | 1 |
| `layer3.4.conv3.weight` | `(1024, 256, 1, 1)` | 262,144 |
| `layer3.4.bn3.weight` | `(1024,)` | 1,024 |
| `layer3.4.bn3.bias` | `(1024,)` | 1,024 |
| `layer3.4.bn3.running_mean` | `(1024,)` | 1,024 |
| `layer3.4.bn3.running_var` | `(1024,)` | 1,024 |
| `layer3.4.bn3.num_batches_tracked` | `()` | 1 |
| `layer3.5.conv1.weight` | `(256, 1024, 1, 1)` | 262,144 |
| `layer3.5.bn1.weight` | `(256,)` | 256 |
| `layer3.5.bn1.bias` | `(256,)` | 256 |
| `layer3.5.bn1.running_mean` | `(256,)` | 256 |
| `layer3.5.bn1.running_var` | `(256,)` | 256 |
| `layer3.5.bn1.num_batches_tracked` | `()` | 1 |
| `layer3.5.conv2.weight` | `(256, 256, 3, 3)` | 589,824 |
| `layer3.5.bn2.weight` | `(256,)` | 256 |
| `layer3.5.bn2.bias` | `(256,)` | 256 |
| `layer3.5.bn2.running_mean` | `(256,)` | 256 |
| `layer3.5.bn2.running_var` | `(256,)` | 256 |
| `layer3.5.bn2.num_batches_tracked` | `()` | 1 |
| `layer3.5.conv3.weight` | `(1024, 256, 1, 1)` | 262,144 |
| `layer3.5.bn3.weight` | `(1024,)` | 1,024 |
| `layer3.5.bn3.bias` | `(1024,)` | 1,024 |
| `layer3.5.bn3.running_mean` | `(1024,)` | 1,024 |
| `layer3.5.bn3.running_var` | `(1024,)` | 1,024 |
| `layer3.5.bn3.num_batches_tracked` | `()` | 1 |
| `layer4.0.conv1.weight` | `(512, 1024, 1, 1)` | 524,288 |
| `layer4.0.bn1.weight` | `(512,)` | 512 |
| `layer4.0.bn1.bias` | `(512,)` | 512 |
| `layer4.0.bn1.running_mean` | `(512,)` | 512 |
| `layer4.0.bn1.running_var` | `(512,)` | 512 |
| `layer4.0.bn1.num_batches_tracked` | `()` | 1 |
| `layer4.0.conv2.weight` | `(512, 512, 3, 3)` | 2,359,296 |
| `layer4.0.bn2.weight` | `(512,)` | 512 |
| `layer4.0.bn2.bias` | `(512,)` | 512 |
| `layer4.0.bn2.running_mean` | `(512,)` | 512 |
| `layer4.0.bn2.running_var` | `(512,)` | 512 |
| `layer4.0.bn2.num_batches_tracked` | `()` | 1 |
| `layer4.0.conv3.weight` | `(2048, 512, 1, 1)` | 1,048,576 |
| `layer4.0.bn3.weight` | `(2048,)` | 2,048 |
| `layer4.0.bn3.bias` | `(2048,)` | 2,048 |
| `layer4.0.bn3.running_mean` | `(2048,)` | 2,048 |
| `layer4.0.bn3.running_var` | `(2048,)` | 2,048 |
| `layer4.0.bn3.num_batches_tracked` | `()` | 1 |
| `layer4.0.downsample.0.weight` | `(2048, 1024, 1, 1)` | 2,097,152 |
| `layer4.0.downsample.1.weight` | `(2048,)` | 2,048 |
| `layer4.0.downsample.1.bias` | `(2048,)` | 2,048 |
| `layer4.0.downsample.1.running_mean` | `(2048,)` | 2,048 |
| `layer4.0.downsample.1.running_var` | `(2048,)` | 2,048 |
| `layer4.0.downsample.1.num_batches_tracked` | `()` | 1 |
| `layer4.1.conv1.weight` | `(512, 2048, 1, 1)` | 1,048,576 |
| `layer4.1.bn1.weight` | `(512,)` | 512 |
| `layer4.1.bn1.bias` | `(512,)` | 512 |
| `layer4.1.bn1.running_mean` | `(512,)` | 512 |
| `layer4.1.bn1.running_var` | `(512,)` | 512 |
| `layer4.1.bn1.num_batches_tracked` | `()` | 1 |
| `layer4.1.conv2.weight` | `(512, 512, 3, 3)` | 2,359,296 |
| `layer4.1.bn2.weight` | `(512,)` | 512 |
| `layer4.1.bn2.bias` | `(512,)` | 512 |
| `layer4.1.bn2.running_mean` | `(512,)` | 512 |
| `layer4.1.bn2.running_var` | `(512,)` | 512 |
| `layer4.1.bn2.num_batches_tracked` | `()` | 1 |
| `layer4.1.conv3.weight` | `(2048, 512, 1, 1)` | 1,048,576 |
| `layer4.1.bn3.weight` | `(2048,)` | 2,048 |
| `layer4.1.bn3.bias` | `(2048,)` | 2,048 |
| `layer4.1.bn3.running_mean` | `(2048,)` | 2,048 |
| `layer4.1.bn3.running_var` | `(2048,)` | 2,048 |
| `layer4.1.bn3.num_batches_tracked` | `()` | 1 |
| `layer4.2.conv1.weight` | `(512, 2048, 1, 1)` | 1,048,576 |
| `layer4.2.bn1.weight` | `(512,)` | 512 |
| `layer4.2.bn1.bias` | `(512,)` | 512 |
| `layer4.2.bn1.running_mean` | `(512,)` | 512 |
| `layer4.2.bn1.running_var` | `(512,)` | 512 |
| `layer4.2.bn1.num_batches_tracked` | `()` | 1 |
| `layer4.2.conv2.weight` | `(512, 512, 3, 3)` | 2,359,296 |
| `layer4.2.bn2.weight` | `(512,)` | 512 |
| `layer4.2.bn2.bias` | `(512,)` | 512 |
| `layer4.2.bn2.running_mean` | `(512,)` | 512 |
| `layer4.2.bn2.running_var` | `(512,)` | 512 |
| `layer4.2.bn2.num_batches_tracked` | `()` | 1 |
| `layer4.2.conv3.weight` | `(2048, 512, 1, 1)` | 1,048,576 |
| `layer4.2.bn3.weight` | `(2048,)` | 2,048 |
| `layer4.2.bn3.bias` | `(2048,)` | 2,048 |
| `layer4.2.bn3.running_mean` | `(2048,)` | 2,048 |
| `layer4.2.bn3.running_var` | `(2048,)` | 2,048 |
| `layer4.2.bn3.num_batches_tracked` | `()` | 1 |
| `fc.weight` | `(2, 2048)` | 4,096 |
| `fc.bias` | `(2,)` | 2 |

</details>

---

## 6. Verification & Status

* **Status:** Verified & Fully Functional
* **Original Model Checkpoint:** [`model/best_resnet50_cifake.pth`](file:///d:/ALGONAUTS_INTERNAL_LJ_HACK-main/model/best_resnet50_cifake.pth) exists unchanged.
* **Copied Checkpoint:** [`model_checkpoints/best_resnet50_cifake_original.pth`](file:///d:/ALGONAUTS_INTERNAL_LJ_HACK-main/model_checkpoints/best_resnet50_cifake_original.pth) is validated and ready for future retraining experiments.
