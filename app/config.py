"""
SignalScope Configuration Settings
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_resnet50_cifake_native32_2.pth"
EVALUATION_DIR = BASE_DIR / "evaluation"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Classification Settings
CLASS_MAPPING = {
    0: "FAKE",
    1: "REAL"
}
NUM_CLASSES = 2

# Preprocessing Constants
IMAGE_SIZE = (32, 32)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Confidence Interpretation Thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.90
MODERATE_CONFIDENCE_THRESHOLD = 0.70

# Inference Strategy & Patch Settings
DEFAULT_INFERENCE_MODE = "auto"
PATCH_N = 32
PATCH_THRESHOLD_PX = 128
PATCH_AGGREGATION_DEFAULT = "mean"
PATCH_AGGREGATION_METHODS = ["mean", "median", "majority", "logit_mean", "max", "top_k"]
INFERENCE_MODES = ["auto", "resize", "patch", "hybrid", "tta"]

# Uncertainty & Agreement Thresholds
ENTROPY_LOW_THRESHOLD = 0.2
ENTROPY_HIGH_THRESHOLD = 0.5
HYBRID_STRONG_DIFF = 0.15
HYBRID_PARTIAL_DIFF = 0.30

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
    "Model Architecture": "ResNet-50 (Native 32x32 Stem)",
    "Dataset": "CIFAKE",
    "Input Size": "32 x 32",
    "Classes": "FAKE, REAL",
    "Test Accuracy": "98.33%",
    "Macro F1 Score": "0.9832",
    "ROC-AUC": "0.9987",
    "PR-AUC": "0.9988",
    "Inference Speed": "~916 img/sec (GPU Batch)"
}

