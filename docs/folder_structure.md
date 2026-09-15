# 07. Project Folder Structure — SignalScope

Below is the complete directory structure for the SignalScope project, detailing the purpose of each directory and file.

```text
ALGONAUTS_INTERNAL_LJ_HACK-main/
│
├── README.md                               # Primary project documentation (entry point)
├── AUDIT_REPORT.md                         # Forensic technical audit report (verified metrics)
├── requirements.txt                        # Pinned Python dependency versions
├── rules.md                                # Team collaboration & project guidelines
│
├── model/
│   ├── best_resnet50_cifake_retrained.pth  # Trained PyTorch ResNet-50 checkpoint (~90MB)
│   └── generator_attribution.py           # Bonus B — Gemini generator family classifier (uses app.api)
│
├── evaluation/                             # Scripts and results for model benchmarking
│   ├── evaluate_strategies.py              # Comprehensive strategy benchmark suite
│   ├── evaluate_unseen.py                  # Unseen-generator evaluation script
│   ├── adversarial_analysis.py             # Bonus G — FGSM adversarial robustness analysis
│   ├── degradation_analysis.py             # Bonus C — JPEG compression robustness analysis
│   ├── evaluation_summary.json             # Strategy benchmark results (robustness + determinism populated)
│   ├── final_evaluation_metrics.csv        # Numerical test metrics export
│   ├── inference_strategy_benchmark.csv    # Per-strategy comparison table
│   └── test_predictions.csv               # 20,000 raw test predictions (source of verified metrics)
│
├── app/                                    # Main application source code
│   ├── __init__.py                         # Package marker
│   ├── app.py                              # Streamlit web application interface (~43KB)
│   ├── config.py                           # Global paths, constants, and strategy thresholds
│   │                                       #   MULTISCALE_FAKE_THRESHOLD=0.608
│   │                                       #   MULTISCALE_REAL_THRESHOLD=0.40
│   │                                       #   MULTISCALE_TOP_K_RATIO=0.20
│   │                                       #   PATCH_N=32
│   ├── model_loader.py                     # ResNet-50 loader with weights_only=True security
│   │                                       # + numpy dtype allowlist + @st.cache_resource
│   ├── predictor.py                        # Unified re-export facade delegating to auto_strategy
│   │
│   ├── api/                                # Gemini API gateway & prompt registry
│   │   ├── __init__.py
│   │   ├── gemini_gateway.py               # Gateway: LRU cache (500 entries), multi-model failover,
│   │   │                                   # rate limiting (12 RPM), request coalescing, retry
│   │   └── prompt_registry.py              # Versioned prompts: neutral framing, 30+ generators,
│   │                                       # 300-token faithful explanation budget
│   │
│   ├── diagnostics/                        # Analytics and bonus modules
│   │   ├── __init__.py
│   │   ├── entropy.py                      # Shannon entropy (H/ln2) + confidence label bands
│   │   ├── disagreement.py                 # Patch-level statistical disagreement metrics
│   │   ├── fft_spectral.py                 # 2D FFT radial spectral scoring (hybrid strategy only)
│   │   ├── grad_cam.py                     # Gradient-weighted Class Activation Mapping
│   │   ├── bounding_box.py                 # Top-N suspect region RGBA overlay renderer
│   │   ├── explainer.py                    # Bonus A — Gemini faithful explanation caller
│   │   ├── metadata_inspector.py           # Bonus D — C2PA/EXIF/PNG AI provenance inspector
│   │   │                                   # 44 known AI signatures; CAMERA_REAL is informational only
│   │   └── multimodal_consistency.py       # Bonus E — image-caption consistency scorer
│   │
│   └── strategies/                         # Modular inference strategy implementations
│       ├── __init__.py
│       ├── base_strategy.py                # ABC + validate_strategy_output() schema validator
│       ├── strategy_registry.py            # Singleton strategy registry (dict-based, import-time registration)
│       ├── auto/
│       │   ├── __init__.py
│       │   └── auto_strategy.py            # 4-tier dispatcher: <64px→resize, 64-255px→patch,
│       │                                   # 256-511px→hybrid, ≥512px→multiscale
│       ├── hybrid/
│       │   ├── __init__.py
│       │   └── hybrid_strategy.py          # Resize+Patch+FFT decision tree (calibrated Branch 1b)
│       ├── multiscale/
│       │   ├── __init__.py
│       │   └── multiscale_strategy.py      # 3-branch: Global + Context (16 patches) + Native Texture
│       │                                   # Adaptive weight fusion; tri-state: REAL/FAKE/UNCERTAIN
│       ├── patch/
│       │   ├── __init__.py
│       │   ├── patch_extractor.py          # Variance-guided 32×32 patch sampler (center/corner/grid/variance)
│       │   └── patch_strategy.py           # Patch voting with center-weighted aggregation
│       │                                   # Top-K ratio: MULTISCALE_TOP_K_RATIO=20% (config-unified)
│       ├── resize/
│       │   ├── __init__.py
│       │   └── resize_strategy.py          # Baseline 32×32 bicubic resize + single forward pass
│       └── tta/
│           ├── __init__.py
│           └── tta_strategy.py             # 8-view TTA: Original + H-Flip + Center Crop + Bright±15%
│                                           # + Contrast+20% + Rotate 90°/45°; inverse-entropy weighting
│
├── tests/                                  # Pytest unit testing suite (39 tests total)
│   ├── __init__.py
│   ├── test_predictor.py                   # Core model + preprocessing tests (8 tests)
│   ├── test_patch_inference.py             # Patch extraction + aggregation + diagnostics (13 tests)
│   ├── test_multiscale_inference.py        # MultiScale branch + pipeline tests (6 tests)
│   ├── test_real_world_stability.py        # Stability across seeds/patch counts (3 tests)
│   └── test_strategy_registry.py          # Registry listing + strategy run tests (5 tests)
│
├── docs/                                   # Project documentation
│   ├── 01_project_overview.md             # High-level summary, tech stack, features
│   ├── 02_system_architecture.md          # Mermaid diagrams and component roles
│   ├── 03_dataset_and_model.md            # CIFAKE dataset & ResNet-50 specs
│   ├── 04_training_and_evaluation.md      # Verified benchmark metrics (97.87% / 0.9980 ROC-AUC)
│   ├── 05_application_and_setup.md        # Local installation & full API reference
│   ├── 06_testing_limitations_and_future_scope.md  # Test results, limitations, roadmap
│   ├── README.md                          # Master README for the docs folder
│   └── folder_structure.md               # Project structure reference (this file)
│
├── notebook/
│   └── SignalScope_CIFAKE_ResNet50_Native32.ipynb  # Google Colab training notebook
│
├── report/
│   └── report.md                          # One-page model report
│
└── .venv/                                 # Local Python virtual environment (git-ignored)
```

---

## Key Architectural Notes

- **`app.predictor`** acts as a unified re-export facade for the Streamlit app. The web UI does not need to know which strategy is executing.
- **`app/strategies/`** encapsulates the exact logic for predicting the label. All strategies are stateless — no mutable state between calls.
- **`app/diagnostics/`** provides mathematical tools (entropy, FFT, Grad-CAM) that strategies and the UI call independently.
- **`app/api/`** is the single integration point for all Gemini calls. No other module should instantiate Gemini clients directly.
- **Strategy routing** (auto mode) uses minimum image dimension as the sole routing criterion. FFT computation is deferred until hybrid is selected to avoid wasted compute.
- **Checkpoint security**: `weights_only=True` with numpy dtype allowlist. The checkpoint is from the project's own trusted training pipeline — adding numpy types to the allowlist is safe and does not compromise security.
