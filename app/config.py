"""
SignalScope Configuration Settings
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_resnet50_cifake_native32.pth"
if not MODEL_PATH.exists():
    _fallback = BASE_DIR / "model" / "best_resnet50_cifake.pth"
    if _fallback.exists():
        MODEL_PATH = _fallback
EVALUATION_DIR = BASE_DIR / "evaluation"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Classification Settings
CLASS_MAPPING = {
    0: "FAKE",
    1: "REAL"
}
NUM_CLASSES = 2

# Preprocessing Constants
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Confidence Interpretation Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.90
MODERATE_CONFIDENCE_THRESHOLD = 0.70

# Disclaimer Messages
DISCLAIMER_TEXT = (
    "SignalScope is an AI-based screening tool. It should not be treated as "
    "definitive proof that an image is real or AI-generated. Performance may vary on "
    "images from generators or datasets not represented in CIFAKE."
)

CONFIDENCE_DISCLAIMER = (
    "Confidence represents model probability, not a absolute guarantee of authenticity."
)

# Benchmark Performance (CIFAKE Test Set)
BENCHMARK_METRICS = {
    "Model Architecture": "ResNet-50",
    "Dataset": "CIFAKE",
    "Input Size": "224 x 224",
    "Classes": "FAKE, REAL",
    "Test Accuracy": "97.42%",
    "Macro F1 Score": "0.9741",
    "ROC-AUC": "0.9964",
    "PR-AUC": "0.9966",
    "Inference Speed": "~1092 img/sec (GPU Batch)"
}
