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
INFERENCE_MODES = ["auto", "resize", "patch", "hybrid", "multiscale", "tta"]

# Multi-Scale Inference Configuration
MULTISCALE_CONTEXT_SIZE = (128, 128)
MULTISCALE_PATCH_SIZE = (32, 32)
MULTISCALE_GLOBAL_WEIGHT = 0.50
MULTISCALE_CONTEXT_WEIGHT = 0.30
MULTISCALE_TEXTURE_WEIGHT = 0.20
MULTISCALE_TOP_K_RATIO = 0.20
MULTISCALE_FAKE_THRESHOLD = 0.60
MULTISCALE_REAL_THRESHOLD = 0.40
MAX_NATIVE_PATCHES = 256
INFERENCE_BATCH_SIZE = 32

class InferenceConfig:
    patch_size = 32
    context_size = 128
    patch_stride = 32
    top_k_ratio = 0.20

    global_weight = 0.50
    object_context_weight = 0.30
    native_texture_weight = 0.20

    fake_threshold = 0.60
    real_threshold = 0.40

    min_supporting_patch_ratio = 0.25
    max_native_patches = 256
    inference_batch_size = 32

    enable_fft_diagnostics = True
    enable_ycbcr_diagnostics = True
    enable_debug_patch_output = False

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

