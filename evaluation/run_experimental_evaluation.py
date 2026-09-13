"""
Experimental Inference Strategy Evaluation (Non-Retraining)
Evaluates and benchmarks multiple inference strategies (Resize Baseline, Patch Voting 8/16/32/64, Hybrid, TTA)
on available test evaluation data using the trained ResNet-50 model.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score
)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.model_loader import load_model
from app.predictor import (
    predict_image,
    predict_image_patch_vote,
    predict_image_hybrid,
    predict_image_tta,
    prepare_image
)
from PIL import Image


def generate_synthetic_benchmark_dataset(n_samples: int = 40):
    """
    Generates a deterministic synthetic evaluation set of varied resolutions and patterns
    (natural photo textures vs high-frequency synthetic patterns) for inference strategy benchmarking.
    """
    images = []
    labels = []  # 0: FAKE, 1: REAL
    rng = np.random.RandomState(42)

    sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]

    for i in range(n_samples):
        size = sizes[i % len(sizes)]
        label = i % 2
        labels.append(label)

        if label == 1:
            # REAL texture: smooth gradient with low-frequency variations
            arr = np.zeros((*size, 3), dtype=np.uint8)
            for r in range(size[0]):
                for c in range(size[1]):
                    arr[r, c, 0] = int(50 + (r / size[0]) * 150)
                    arr[r, c, 1] = int(100 + (c / size[1]) * 100)
                    arr[r, c, 2] = int(150 - (r / size[0]) * 50)
            img = Image.fromarray(arr)
        else:
            # FAKE pattern: high-frequency noise / grid patterns
            arr = rng.randint(0, 256, (*size, 3), dtype=np.uint8)
            img = Image.fromarray(arr)

        images.append(img)

    return images, labels


def evaluate_strategy(name: str, eval_fn, images, ground_truth):
    """
    Evaluates a specific inference strategy function across images.
    """
    preds = []
    fake_probs = []
    times = []

    for img in images:
        start_t = time.time()
        res = eval_fn(img)
        elapsed = (time.time() - start_t) * 1000
        times.append(elapsed)

        pred_label_idx = 0 if res["label"] == "FAKE" else 1
        preds.append(pred_label_idx)
        fake_probs.append(res["fake_probability"])

    acc = accuracy_score(ground_truth, preds)
    prec = precision_score(ground_truth, preds, zero_division=0)
    rec = recall_score(ground_truth, preds, zero_division=0)
    f1 = f1_score(ground_truth, preds, zero_division=0)
    bal_acc = balanced_accuracy_score(ground_truth, preds)
    
    try:
        # Note: ROC-AUC against fake_probability (where FAKE is class 0)
        # Using 1 - fake_prob as real_prob score for class 1
        auc = roc_auc_score(ground_truth, [1.0 - p for p in fake_probs])
    except Exception:
        auc = 0.5

    avg_time = float(np.mean(times))

    return {
        "Strategy": name,
        "Accuracy": f"{acc * 100:.2f}%",
        "Precision": f"{prec:.4f}",
        "Recall": f"{rec:.4f}",
        "F1 Score": f"{f1:.4f}",
        "Balanced Acc": f"{bal_acc * 100:.2f}%",
        "ROC-AUC": f"{auc:.4f}",
        "Avg Time (ms)": f"{avg_time:.2f} ms"
    }


def main():
    print("=" * 70)
    print("SIGNALSSCOPE NON-RETRAINING INFERENCE EVALUATION")
    print("=" * 70)

    model, device = load_model()
    print(f"Loaded trained ResNet-50 model on {device.type.upper()}")

    images, ground_truth = generate_synthetic_benchmark_dataset(n_samples=40)
    print(f"Benchmark Dataset: {len(images)} samples across multi-resolution sizes (64x64 to 512x512)\n")

    strategies = {
        "1. Resize Baseline (32x32)": lambda img: predict_image(img, model=model, device=device),
        "2. Patch Voting (N=8)": lambda img: predict_image_patch_vote(img, model=model, device=device, n_patches=8),
        "3. Patch Voting (N=16)": lambda img: predict_image_patch_vote(img, model=model, device=device, n_patches=16),
        "4. Patch Voting (N=32)": lambda img: predict_image_patch_vote(img, model=model, device=device, n_patches=32),
        "5. Patch Voting (N=64)": lambda img: predict_image_patch_vote(img, model=model, device=device, n_patches=64),
        "6. Hybrid (Resize + Patch32)": lambda img: predict_image_hybrid(img, model=model, device=device, n_patches=32),
        "7. Test-Time Augmentation (TTA)": lambda img: predict_image_tta(img, model=model, device=device),
    }

    results = []
    for name, fn in strategies.items():
        print(f"Evaluating: {name}...")
        res = evaluate_strategy(name, fn, images, ground_truth)
        results.append(res)

    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("INFERENCE STRATEGY BENCHMARK RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))

    output_csv = ROOT_DIR / "evaluation" / "inference_strategy_benchmark.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved benchmark results to: {output_csv}")


if __name__ == "__main__":
    main()
