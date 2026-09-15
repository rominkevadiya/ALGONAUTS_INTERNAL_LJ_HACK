# SignalScope Technical Documentation Index

Welcome to the technical documentation for **SignalScope**, an AI-generated image detection application utilizing a fine-tuned ResNet-50 deep learning model.

---

## 📌 Documentation Index & Table of Contents

1. [`01_project_overview.md`](01_project_overview.md) — Problem statement, motivation, scope, core features, tech stack, and input/output definitions.
2. [`02_system_architecture.md`](02_system_architecture.md) — System components, module relationships, training vs. inference architecture, error handling, and Mermaid diagrams.
3. [`03_dataset_and_model.md`](03_dataset_and_model.md) — Combined CIFAKE dataset specifications (120,000 images, 80k/20k/20k splits) and ResNet-50 architecture/checkpoint metadata.
4. [`04_training_and_evaluation.md`](04_training_and_evaluation.md) — Training hyperparameters (5 epochs, AdamW + CosineAnnealingLR), verified benchmark test metrics (97.87% accuracy, 0.9786 F1, 0.9980 ROC-AUC), corrected confusion matrix (9,786 TN / 9,787 TP), and metric formulas.
5. [`05_application_and_setup.md`](05_application_and_setup.md) — Streamlit web app interface, live PyTorch diagnostic panel, 14-step inference pipeline, PowerShell setup, and complete API reference.
6. [`06_testing_limitations_and_future_scope.md`](06_testing_limitations_and_future_scope.md) — Pytest test suite, manual verification cases, CIFAKE dataset domain shift risks, security guidelines, and future roadmap.

---

## 🧭 Recommended Reading Order

- **For Evaluators & Judges**: [`01_project_overview.md`](01_project_overview.md) $\rightarrow$ [`04_training_and_evaluation.md`](04_training_and_evaluation.md) $\rightarrow$ [`05_application_and_setup.md`](05_application_and_setup.md)
- **For Developers & Contributors**: [`01_project_overview.md`](01_project_overview.md) $\rightarrow$ [`02_system_architecture.md`](02_system_architecture.md) $\rightarrow$ [`05_application_and_setup.md`](05_application_and_setup.md) $\rightarrow$ [`06_testing_limitations_and_future_scope.md`](06_testing_limitations_and_future_scope.md)

---

> [!NOTE]
> *Benchmark Note: All performance numbers reported in this documentation are benchmark results from the active **native32 checkpoint** (`model/best_resnet50_cifake_retrained.pth`) evaluated on the official CIFAKE test dataset (20,000 images): **97.87% accuracy**, **0.9786 Macro F1**, **0.9980 ROC-AUC**. These numbers reflect the CIFAR-10 derived image distribution ($32 \times 32$ upscaled to $224 \times 224$). High-resolution digital photographs or modern webcam photos may exhibit domain shift.*
