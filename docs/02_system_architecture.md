# 02. System Architecture — SignalScope

## 1. System Overview & Component Responsibilities

SignalScope is organized into decoupled Python modules:

- **`app/config.py`**: Central configuration storing model path (`model/best_resnet50_cifake_retrained.pth`), ImageNet normalization mean/std, spatial input dimensions (`32, 32`), class mapping (`0: FAKE`, `1: REAL`), and benchmark metrics constants.
- **`app/model_loader.py`**: Handles device selection (`cuda` vs `cpu`), instantiates `torchvision.models.resnet50(weights=None)`, auto-detects the checkpoint stem variant, replacing the final layer with `Linear(2048, 2)`, restores weight tensors, and caches the model using Streamlit's `@st.cache_resource`.
- **`app/predictor.py`**: Facade module that delegates inference to specialized strategies (`auto`, `resize`, `patch`, `hybrid`, `tta`). Processes both single and batch predictions.
- **`app/strategies/`**: Contains modular inference strategies (Auto-Dispatcher, Baseline Resize, Native Patch Voting, Hybrid Consensus, 8-View TTA).
- **`app/api/gemini_gateway.py`**: Centralized, resilient gateway for Google GenAI SDK calls. Implements default `gemini-3.1-flash-lite` model, automated multi-model failover (`gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.5-flash` $\rightarrow$ `gemini-3.5-flash-lite`), zero-thinking latency optimization (`thinking_budget=0`), thread-safe in-memory response caching, rate limiting (12 req/min), bounded 429 retries, and regex markdown fence stripping.
- **`app/api/prompt_registry.py`**: Versioned prompt definitions for Generator Attribution (Module B), Faithful Explanation (Module A), and Multimodal Consistency (Module E).
- **`app/diagnostics/`**: Contains analytics modules (Shannon Entropy, Statistical Disagreement, 2D FFT Spectral scoring). Also contains **`explainer.py`** (Gemini explanation), **`metadata_inspector.py`** (C2PA/EXIF screening), and **`grad_cam.py`** (Heatmap generation).
- **`model/`**: Contains the frozen weights (`best_resnet50_cifake_retrained.pth`) and **`generator_attribution.py`** (Gemini-based Generator identification).
- **`app/app.py`**: Streamlit web dashboard managing UI rendering, single/batch upload tabs, `st.session_state` inference/API caching, on-demand retry buttons, live diagnostic logits expanders, and CSV downloads.

---

## 2. Training-Time vs. Inference-Time Architecture

| Feature | Training Pipeline (`notebook/`) | Local Inference App (`app/`) |
| :--- | :--- | :--- |
| **Execution Host** | Google Colab (Tesla T4 GPU) | Local Workstation (CPU / CUDA) |
| **Model Mode** | `model.train()` (Backpropagation) | `model.eval()` (`torch.no_grad()`) |
| **Transformations** | Random Horizontal Flip, Rotation, ColorJitter | Deterministic Resize $224 \times 224$ & Normalization |
| **Weight Modification**| Weights updated via AdamW ($1 \times 10^{-4}$) | Read-only frozen checkpoint weights |

---

## 3. CPU / CUDA Execution & Error Handling

- **Device Auto-Detection**: `get_device()` checks `torch.cuda.is_available()`. Loads weights using `map_location=device` to ensure seamless CPU execution when GPU acceleration is unavailable.
- **Error Shielding**: `model_loader.py` catches `FileNotFoundError` if the checkpoint is missing, and `app.py` catches PIL decode errors to display friendly user alerts without exposing raw tracebacks.

---

## 4. Architectural Mermaid Diagrams

### Diagram 1: System Architecture
```mermaid
graph TD
    User["User Upload (JPG / PNG)"] --> UI["Streamlit Interface (app/app.py)"]
    UI --> Loader["Model Loader (app/model_loader.py)"]
    Loader --> Checkpoint["Checkpoint File (model/best_resnet50_cifake_retrained.pth)"]
    UI --> Predictor["Prediction Engine (app/predictor.py)"]
    Predictor --> Preproc["Preprocessing (RGB, Resize 224x224, Normalize)"]
    Preproc --> Forward["ResNet-50 Forward Pass (eval mode)"]
    Forward --> Softmax["Softmax Probability Computation"]
    Softmax --> Output["UI Display (Result Badges & Live Logits)"]
```

### Diagram 2: Training Pipeline
```mermaid
graph TD
    Dataset["CIFAKE Dataset (120,000 Images)"] --> Pool["100,000 Training Pool"]
    Dataset --> TestSet["20,000 Untouched Test Set"]
    Pool --> TrainSplit["80,000 Train Set (80%)"]
    Pool --> ValSplit["20,000 Validation Set (20%)"]
    TrainSplit --> Augment["Augmentation Transforms"]
    Augment --> ResNet["ResNet-50 Model (ImageNet Weights)"]
    ResNet --> AdamW["AdamW Optimizer (lr=1e-4)"]
    AdamW --> Epochs["5 Epoch Training Loop"]
    ValSplit --> Monitor["Validation Monitoring"]
    Epochs --> Monitor
    Monitor --> BestSave["Save Best Model (Epoch 5, Val Acc 98.06%)"]
    BestSave --> CheckpointFile["best_resnet50_cifake_retrained.pth"]
```

### Diagram 3: Inference Pipeline
```mermaid
graph LR
    User["User Image"] --> Preproc["Preprocessing (RGB, EXIF)"]
    Preproc --> Predictor["Predictor Facade"]
    Predictor --> Dispatch["Auto-Dispatcher"]
    Dispatch -- "<64px" --> Resize["Resize Strategy"]
    Dispatch -- "64px - 256px" --> Patch["Patch Strategy (Native crops)"]
    Dispatch -- ">256px" --> Hybrid["Hybrid Strategy + FFT"]
    Resize --> EvalPass["ResNet-50 eval() (torch.no_grad)"]
    Patch --> EvalPass
    Hybrid --> EvalPass
    EvalPass --> Logits["Raw Logits (z0, z1)"]
    Logits --> Diagnostics["Diagnostics (Entropy, Disagreement)"]
    Diagnostics --> DictOut["Result Dict (Label, Conf, Probs)"]
```

### Diagram 4: Gemini Multimodal Vision Gateway & Session Caching
```mermaid
graph TD
    UI["Streamlit UI (app/app.py)"] --> Check{"Fresh Image or Settings Changed?"}
    Check -- "No (Tab Switch / Re-run)" --> Cache["st.session_state Cache (0 API Calls, 0ms Latency)"]
    Cache --> Render["Instant Dashboard Render"]
    Check -- "Yes (New Analysis)" --> PyTorch["PyTorch ResNet-50 Pipeline"]
    PyTorch --> Gateway["Central Gemini Gateway (app/api/gemini_gateway.py)"]
    Gateway --> ModA["Module A: Faithful Explanation"]
    Gateway --> ModB["Module B: Generator Attribution"]
    Gateway --> ModE["Module E: Multimodal Caption Consistency"]
    Gateway --> Pool["Multi-Model Failover Pool"]
    Pool --> M1["gemini-3.1-flash-lite (Primary, 2.5s, 1.5k RPD)"]
    Pool -- "On 429/404" --> M2["gemini-3.5-flash (Backup 1)"]
    Pool -- "On 429/404" --> M3["gemini-3.5-flash-lite (Backup 2)"]
    M1 --> Clean["Markdown Code Fence Cleaning (re.sub)"]
    M2 --> Clean
    M3 --> Clean
    Clean --> Store["Store in st.session_state"]
    Store --> Render
```

