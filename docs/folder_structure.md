# 07. Project Folder Structure — SignalScope

Below is the complete directory structure for the SignalScope project, detailing the purpose of each directory and file.

```text
ALGONAUTS_INTERNAL_LJ_HACK-main/
│
├── model/
│   └── best_resnet50_cifake_retrained.pth # Trained PyTorch ResNet-50 checkpoint (~94.3 MB)
│
├── evaluation/                             # Scripts and results for model benchmarking
│   ├── final_evaluation_metrics.csv        # Numerical test metrics export
│   ├── inference_strategy_benchmark.csv    # Strategy benchmark comparison
│   ├── run_experimental_evaluation.py      # Non-retraining benchmark script
│   └── test_predictions.csv                # Detailed 20k test predictions
│
├── app/                                    # Main application source code
│   ├── __init__.py                         # Package marker
│   ├── app.py                              # Streamlit web application interface
│   ├── config.py                           # Global paths, constants, and thresholds
│   ├── model_loader.py                     # PyTorch checkpoint loader with caching
│   ├── predictor.py                        # Facade API delegating to inference strategies
│   │
│   ├── diagnostics/                        # Analytics modules
│   │   ├── __init__.py
│   │   ├── entropy.py                      # Shannon entropy calculations
│   │   ├── disagreement.py                 # Statistical variance and stability
│   │   └── fft_spectral.py                 # 2D Fast Fourier Transform structural scoring
│   │
│   └── strategies/                         # Modular inference strategy implementations
│       ├── __init__.py
│       ├── auto/                           # Graduated resolution dispatcher
│       │   ├── __init__.py
│       │   └── auto_strategy.py
│       ├── hybrid/                         # Multi-strategy consensus decision tree
│       │   ├── __init__.py
│       │   └── hybrid_strategy.py
│       ├── patch/                          # Native crop extraction logic
│       │   ├── __init__.py
│       │   ├── patch_extractor.py          # Variance-guided patch sampler
│       │   └── patch_strategy.py           # Patch voting with center-weighted aggregation
│       ├── resize/                         # Baseline 32x32 scaling
│       │   ├── __init__.py
│       │   └── resize_strategy.py
│       └── tta/                            # Test-time augmentation (8-views)
│           ├── __init__.py
│           └── tta_strategy.py             # Inverse-entropy weighted TTA evaluation
│
├── tests/                                  # Pytest unit testing suite
│   ├── __init__.py                         # Test package marker
│   ├── test_patch_inference.py             # Unit tests for all inference modes (24 tests)
│   └── test_predictor.py                   # Baseline structural & initialization tests
│
├── docs/                                   # Project documentation
│   ├── 01_project_overview.md              # High-level summary & workflow
│   ├── 02_system_architecture.md           # Mermaid diagrams and component roles
│   ├── 03_dataset_and_model.md             # CIFAKE dataset & ResNet-50 specs
│   ├── 04_training_and_evaluation.md       # Benchmarks & confusion matrix
│   ├── 05_application_and_setup.md         # Local installation & API reference
│   ├── 06_testing_limitations_and_future_scope.md # Unit test results & roadmaps
│   ├── README.md                           # Master README for the docs folder
│   └── folder_structure.md                 # Project structure reference (this file)
│
├── .venv/                                  # Local Python virtual environment (ignored in git)
├── requirements.txt                        # Python dependencies required for the project
├── README.md                               # Primary project repository documentation
└── rules.md                                # Team collaboration & project guidelines
```

---

## Key Architectural Notes
- **`app.predictor`** acts as a unified facade for the Streamlit app. The web UI does not need to know which strategy is executing.
- **`app/strategies/`** encapsulates the exact logic for predicting the label. The strategies are completely stateless.
- **`app/diagnostics/`** provides mathematical tools that strategies can use to measure certainty (like Entropy or FFT anomalies).
