# 02. System Architecture — SignalScope

## 1. System Overview & Component Responsibilities

SignalScope is organized into decoupled Python modules:

- **`app/config.py`**: Central configuration storing model path (`model/best_resnet50_cifake_retrained.pth`), ImageNet normalization mean/std, spatial input dimensions (`32, 32`), class mapping (`0: FAKE`, `1: REAL`), strategy thresholds (`MULTISCALE_FAKE_THRESHOLD=0.608`, `MULTISCALE_REAL_THRESHOLD=0.40`, `MULTISCALE_TOP_K_RATIO=0.20`), and benchmark metrics constants.
- **`app/model_loader.py`**: Handles device selection (`cuda` vs `cpu`), instantiates `torchvision.models.resnet50(weights=None)`, auto-detects the checkpoint stem variant (standard 7×7 ImageNet stem vs CIFAR-adapted 3×3 stem), replacing the final layer with `Linear(2048, 2)`, restores weight tensors, and caches the model using Streamlit's `@st.cache_resource`. Uses `weights_only=True` with a comprehensive numpy dtype allowlist for checkpoint security.
- **`app/predictor.py`**: Facade module that re-exports all inference functions from the strategy packages for backward compatibility. Delegates directly to `auto_strategy.predict_image_auto`.
- **`app/strategies/`**: Contains six modular inference strategies: Auto-Dispatcher, Baseline Resize, Native Patch Voting, Hybrid Consensus, 3-Branch MultiScale, and 8-View TTA.
- **`app/api/gemini_gateway.py`**: Centralized, resilient gateway for Google GenAI SDK calls. Implements `gemini-3.1-flash-lite` (primary), automated multi-model failover, zero-thinking latency optimization (`thinking_budget=0`), thread-safe LRU in-memory response cache (500-entry max, OrderedDict-based eviction), rate limiting (12 req/min, 1s spacing), request coalescing (in-flight deduplication via `threading.Event`), bounded 429 retries, and regex markdown fence stripping.
- **`app/api/prompt_registry.py`**: Versioned prompt definitions for Generator Attribution (Module B — neutral framing with 30+ generator families), Faithful Explanation (Module A — richer diagnostic context, 300 token budget), and Multimodal Consistency (Module E).
- **`app/diagnostics/`**: Contains analytics modules:
  - `entropy.py` — Shannon entropy + confidence label bands
  - `disagreement.py` — patch-level statistical disagreement metrics
  - `fft_spectral.py` — 2D FFT radial spectral anomaly scoring (used exclusively by hybrid strategy)
  - `explainer.py` — Gemini faithful explanation caller (Module A)
  - `metadata_inspector.py` — C2PA/EXIF/PNG AI provenance inspector with 44 known AI signatures (Module D)
  - `multimodal_consistency.py` — Image-caption consistency scorer (Module E)
- **`model/`**: Contains the frozen weights (`best_resnet50_cifake_retrained.pth`) and `generator_attribution.py` (Module B Gemini fallback — uses `app.api` imports).
- **`app/app.py`**: Streamlit web dashboard managing UI rendering, single/batch upload tabs, `st.session_state` inference/API caching, on-demand retry buttons, live diagnostic logits expanders, and CSV downloads.

---

## 2. Training-Time vs. Inference-Time Architecture

| Feature | Training Pipeline (`notebook/`) | Local Inference App (`app/`) |
| :--- | :--- | :--- |
| **Execution Host** | Google Colab (Tesla T4 GPU) | Local Workstation (CPU / CUDA) |
| **Model Mode** | `model.train()` (Backpropagation) | `model.eval()` (`torch.inference_mode()`) |
| **Transformations** | Random Horizontal Flip, Rotation, ColorJitter | Strategy-dependent (resize, native 32×32 crop, or TTA augmentations) |
| **Weight Modification**| Weights updated via AdamW (1×10⁻⁴) | Read-only frozen checkpoint weights |

---

## 3. CPU / CUDA Execution & Error Handling

- **Device Auto-Detection**: `get_device()` checks `torch.cuda.is_available()`. Loads weights using `map_location=device` to ensure seamless CPU execution when GPU acceleration is unavailable.
- **Error Shielding**: `model_loader.py` catches `FileNotFoundError` if the checkpoint is missing, and `app.py` catches PIL decode errors to display friendly user alerts without exposing raw tracebacks.
- **Security**: Checkpoint loading uses `weights_only=True` with a comprehensive numpy dtype allowlist (`numpy.dtype`, `numpy._core.multiarray.scalar`, and all `numpy.dtypes.*` concrete types).

---

## 4. Architectural Mermaid Diagrams

### Diagram 1: Full System Data Flow
```mermaid
graph TD
    User["User Upload (JPG / PNG)"] --> UI["Streamlit Interface (app/app.py)"]
    UI --> Meta["Metadata Inspector (C2PA / EXIF)"]
    Meta --> Auto["Auto Dispatcher (auto_strategy.py)"]
    Auto --> Loader["Model Loader (ResNet-50 @cache_resource)"]
    Loader --> Checkpoint["Checkpoint (best_resnet50_cifake_retrained.pth)"]
    Auto --> Strategy["Selected Strategy"]
    Strategy --> Resize["Resize Strategy"]
    Strategy --> Patch["Patch Strategy"]
    Strategy --> Hybrid["Hybrid Strategy + FFT"]
    Strategy --> Multiscale["MultiScale Strategy (3-branch)"]
    Strategy --> TTA["TTA Strategy (8-view)"]
    Resize & Patch & Hybrid & Multiscale & TTA --> Diag["Diagnostics (Entropy, Disagreement)"]
    Diag --> Gemini["Gemini Gateway (A/B/E)"]
    Gemini --> Output["UI Display (Result + Diagnostics)"]
```

### Diagram 2: Auto-Dispatcher Routing (4-Tier)
```mermaid
graph LR
    User["User Image"] --> Preproc["prepare_image (RGB, EXIF)"]
    Preproc --> Dispatch["Auto-Dispatcher (min_dim)"]
    Dispatch -- "< 64px" --> Resize["Resize Strategy"]
    Dispatch -- "64–255px" --> Patch["Patch Strategy (native crops)"]
    Dispatch -- "256–511px" --> Hybrid["Hybrid Strategy + FFT"]
    Dispatch -- "≥ 512px" --> MS["MultiScale Strategy (3-branch)"]
    Resize & Patch & Hybrid & MS --> Forward["ResNet-50 inference_mode()"]
    Forward --> Logits["Raw Logits (z0=FAKE, z1=REAL)"]
    Logits --> Diag["Diagnostics (Entropy, Disagreement)"]
    Diag --> Out["Result Dict (Label, Conf, Probs, Diagnostics)"]
```

### Diagram 3: Training Pipeline
```mermaid
graph TD
    Dataset["CIFAKE Dataset (120,000 Images)"] --> Pool["100,000 Training Pool"]
    Dataset --> TestSet["20,000 Untouched Test Set"]
    Pool --> TrainSplit["80,000 Train Set (80%)"]
    Pool --> ValSplit["20,000 Validation Set (20%)"]
    TrainSplit --> Augment["Augmentation Transforms"]
    Augment --> ResNet["ResNet-50 CIFAR-Adapted Stem (3×3 conv, Identity maxpool)"]
    ResNet --> AdamW["AdamW Optimizer (lr=1e-4, wd=1e-4)"]
    AdamW --> Epochs["Training Loop (8 Epochs)"]
    ValSplit --> Monitor["Validation Monitoring (MCC-based best model save)"]
    Epochs --> Monitor
    Monitor --> BestSave["Save Best Model (Epoch 8, Val MCC 0.9334)"]
    BestSave --> CheckpointFile["best_resnet50_cifake_retrained.pth"]
```

### Diagram 4: Gemini Gateway Architecture
```mermaid
graph TD
    UI["Streamlit UI (app/app.py)"] --> Check{"Cached in st.session_state?"}
    Check -- "Yes (Tab Switch / Re-run)" --> Cache["st.session_state Cache (0 API Calls, 0ms)"]
    Cache --> Render["Instant Dashboard Render"]
    Check -- "No (New Image)" --> PyTorch["PyTorch ResNet-50 Pipeline"]
    PyTorch --> Gateway["Central Gemini Gateway (app/api/gemini_gateway.py)"]
    Gateway --> InFlight{"In-flight deduplication (threading.Event)"}
    InFlight --> LRUCache["LRU Cache check (max 500 entries, SHA-256 key)"]
    LRUCache --> RateLimit["Rate Limiter (12 RPM, 1s min interval)"]
    RateLimit --> Pool["Multi-Model Failover Pool"]
    Pool --> M1["gemini-3.1-flash-lite (Primary)"]
    Pool -- "On 429/404" --> M2["gemini-3.5-flash (Backup 1)"]
    Pool -- "On 429/404" --> M3["gemini-3.5-flash-lite (Backup 2)"]
    M1 & M2 & M3 --> Clean["Markdown Fence Strip + JSON Parse"]
    Clean --> Store["Store in LRU Cache + st.session_state"]
    Store --> Render
```
