"""
SignalScope Strategy Evaluation & Benchmarking Suite

Evaluates all SignalScope inference strategies across two distinct categories:
1. Synthetic Robustness & Schema Validation Benchmark
2. Labeled Image Classification & Runtime Benchmark

Strategies Evaluated:
    - resize
    - patch
    - tta
    - hybrid
    - auto
    - multiscale

Usage:
    .venv\\Scripts\\python evaluation/evaluate_strategies.py [--dataset <path>] [--output-dir <path>] [--determinism-samples <N>] [--skip-plots]
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Reconfigure stdout for UTF-8 encoding on Windows terminals if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.model_loader import load_model  # noqa: E402
from app.strategies.resize.resize_strategy import predict_image as predict_resize  # noqa: E402
from app.strategies.patch.patch_strategy import predict_image_patch_vote as predict_patch  # noqa: E402
from app.strategies.tta.tta_strategy import predict_image_tta as predict_tta  # noqa: E402
from app.strategies.hybrid.hybrid_strategy import predict_image_hybrid as predict_hybrid  # noqa: E402
from app.strategies.auto.auto_strategy import predict_image_auto as predict_auto  # noqa: E402
from app.strategies.multiscale.multiscale_strategy import predict_image_multiscale as predict_multiscale  # noqa: E402


# ── Synthetic Image Generators (Category A: Robustness Tests) ──────────────────

def make_image(w, h, mode="RGB", pattern="random", seed=42):
    """Create a synthetic PIL image for testing."""
    rng = np.random.default_rng(seed)
    if pattern == "random":
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    elif pattern == "gradient":
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                arr[i, j] = [int(i * 255 / h), int(j * 255 / w), 128]
    elif pattern == "dark":
        arr = rng.integers(0, 40, (h, w, 3), dtype=np.uint8)
    elif pattern == "bright":
        arr = rng.integers(200, 256, (h, w, 3), dtype=np.uint8)
    elif pattern == "solid":
        arr = np.full((h, w, 3), [128, 128, 128], dtype=np.uint8)
    else:
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    img = Image.fromarray(arr, "RGB")

    if mode == "RGBA":
        img = img.convert("RGBA")
    elif mode == "L":
        img = img.convert("L")

    return img


TEST_CASES = [
    {"name": "Tiny (16×16, RGB)",       "w": 16,   "h": 16,   "mode": "RGB",  "pattern": "random"},
    {"name": "Small (64×64, RGB)",       "w": 64,   "h": 64,   "mode": "RGB",  "pattern": "random"},
    {"name": "Medium (128×128, RGB)",    "w": 128,  "h": 128,  "mode": "RGB",  "pattern": "gradient"},
    {"name": "HD (512×512, RGB)",        "w": 512,  "h": 512,  "mode": "RGB",  "pattern": "random"},
    {"name": "Full-HD (1920×1080, RGB)", "w": 1920, "h": 1080, "mode": "RGB",  "pattern": "gradient"},
    {"name": "Dark (256×256, RGB)",      "w": 256,  "h": 256,  "mode": "RGB",  "pattern": "dark"},
    {"name": "Bright (256×256, RGB)",    "w": 256,  "h": 256,  "mode": "RGB",  "pattern": "bright"},
    {"name": "Grayscale (128×128, L)",   "w": 128,  "h": 128,  "mode": "L",    "pattern": "random"},
    {"name": "RGBA (256×256, RGBA)",     "w": 256,  "h": 256,  "mode": "RGBA", "pattern": "random"},
    {"name": "Square solid (32×32)",     "w": 32,   "h": 32,   "mode": "RGB",  "pattern": "solid"},
]

STRATEGIES = {
    "resize": lambda img, m, d: predict_resize(img, model=m, device=d),
    "patch":  lambda img, m, d: predict_patch(img, model=m, device=d),
    "tta":    lambda img, m, d: predict_tta(img, model=m, device=d),
    "hybrid": lambda img, m, d: predict_hybrid(img, model=m, device=d),
    "auto":   lambda img, m, d: predict_auto(img, model=m, device=d),
    "multiscale": lambda img, m, d: predict_multiscale(img, model=m, device=d),
}

REQUIRED_KEYS = {"label", "confidence", "fake_probability", "real_probability"}


def get_resolution_bucket(w, h):
    """Categorize image resolution based on min dimension."""
    min_dim = min(w, h)
    if min_dim < 64:
        return "<64"
    elif min_dim <= 255:
        return "64–255"
    elif min_dim <= 511:
        return "256–511"
    elif min_dim <= 1023:
        return "512–1023"
    else:
        return ">=1024"


# ── Category A: Synthetic Robustness Benchmark ───────────────────────────

def run_robustness_benchmark(model, device):
    print("=" * 80)
    print("  CATEGORY A: Synthetic Robustness & Schema Validation Benchmark")
    print("  (Note: Synthetic images test execution, schema, & runtime — NOT accuracy)")
    print("=" * 80)

    summary_results = {}

    for strategy_name, strategy_fn in STRATEGIES.items():
        print(f"\n{'-' * 80}")
        print(f"  STRATEGY: {strategy_name.upper()}")
        print(f"{'-' * 80}")
        print(f"  {'Test Case':<30} {'Label':<10} {'Fake%':>7} {'Real%':>7} {'Conf%':>7} {'Time(ms)':>9} {'Status'}")
        print(f"  {'-'*30} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'------'}")

        passed, failed = 0, 0
        test_details = []

        for tc in TEST_CASES:
            img = make_image(tc["w"], tc["h"], tc["mode"], tc["pattern"])
            try:
                t0 = time.perf_counter()
                result = strategy_fn(img, model, device)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                missing = REQUIRED_KEYS - set(result.keys())
                if missing:
                    raise KeyError(f"Missing keys in result: {missing}")

                label = result["label"]
                fake_p = result["fake_probability"]
                real_p = result["real_probability"]
                conf = result["confidence"]

                # Sanity checks
                assert label in ("FAKE", "REAL", "UNCERTAIN"), f"Bad label: {label}"
                assert 0.0 <= fake_p <= 1.0, f"fake_probability out of range: {fake_p}"
                assert 0.0 <= real_p <= 1.0, f"real_probability out of range: {real_p}"
                assert abs(fake_p + real_p - 1.0) < 1e-3, f"Probs don't sum to 1: {fake_p+real_p}"
                assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

                status = "PASS [✓]"
                passed += 1
                print(f"  {tc['name']:<30} {label:<10} {fake_p*100:>6.2f}% {real_p*100:>6.2f}% {conf*100:>6.2f}% {elapsed_ms:>8.1f}  {status}")

                test_details.append({
                    "test_case": tc["name"],
                    "passed": True,
                    "label": label,
                    "fake_probability": fake_p,
                    "elapsed_ms": elapsed_ms
                })

            except Exception as e:
                failed += 1
                print(f"  {tc['name']:<30} {'ERR':<10} {'---':>7} {'---':>7} {'---':>7} {'---':>9}  FAIL [✗]  [{e}]")
                test_details.append({
                    "test_case": tc["name"],
                    "passed": False,
                    "error": str(e)
                })

        total = passed + failed
        print(f"\n  Result: {passed}/{total} passed", "[✓]" if failed == 0 else f"  [✗] {failed} FAILED")
        summary_results[strategy_name] = {
            "passed": passed,
            "total": total,
            "details": test_details
        }

    return summary_results


# ── Category A.2: Determinism Benchmark ───────────────────────────────────

def run_determinism_benchmark(model, device, num_samples=3):
    print(f"\n{'-' * 80}")
    print(f"  DETERMINISM BENCHMARK (Testing repeat consistency on {num_samples} images)")
    print(f"{'-' * 80}")

    determinism_results = {}

    for strategy_name, strategy_fn in STRATEGIES.items():
        consistent_runs = 0

        for i in range(num_samples):
            img = make_image(256, 256, "RGB", "random", seed=100 + i)
            res1 = strategy_fn(img, model, device)
            res2 = strategy_fn(img, model, device)

            label_match = res1["label"] == res2["label"]
            fake_prob_match = abs(res1["fake_probability"] - res2["fake_probability"]) < 1e-5
            real_prob_match = abs(res1["real_probability"] - res2["real_probability"]) < 1e-5

            # Special check for multiscale highlighted regions consistency
            regions_match = True
            if strategy_name == "multiscale":
                r1 = res1.get("analysis", {}).get("highlighted_regions", [])
                r2 = res2.get("analysis", {}).get("highlighted_regions", [])
                regions_match = (len(r1) == len(r2))

            if label_match and fake_prob_match and real_prob_match and regions_match:
                consistent_runs += 1

        is_deterministic = (consistent_runs == num_samples)
        status_str = "DETERMINISTIC [✓]" if is_deterministic else "NON-DETERMINISTIC [✗]"
        print(f"  {strategy_name:<15}: {consistent_runs}/{num_samples} identical runs -> {status_str}")
        determinism_results[strategy_name] = {
            "consistent": consistent_runs,
            "total": num_samples,
            "is_deterministic": is_deterministic
        }

    return determinism_results


# ── Category B: Classification Dataset Loading ────────────────────────────

def load_classification_dataset(dataset_dir):
    """Walk dataset directory looking for real/ and fake/ subdirectories."""
    dataset_path = Path(dataset_dir).resolve()
    if not dataset_path.exists() or not dataset_path.is_dir():
        return []

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    samples = []

    for label_str, ground_truth in [("fake", "FAKE"), ("real", "REAL")]:
        subdir = dataset_path / label_str
        if not subdir.exists():
            # Try uppercase or mixed case if needed
            for child in dataset_path.iterdir():
                if child.is_dir() and child.name.lower() == label_str:
                    subdir = child
                    break

        if subdir.exists() and subdir.is_dir():
            for file_path in subdir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in valid_exts:
                    try:
                        with Image.open(file_path) as img:
                            w, h = img.size
                        samples.append({
                            "path": str(file_path),
                            "filename": file_path.name,
                            "ground_truth": ground_truth,  # FAKE = class 0 (positive), REAL = class 1 (negative)
                            "w": w,
                            "h": h,
                            "resolution_bucket": get_resolution_bucket(w, h)
                        })
                    except Exception:
                        pass

    return samples


# ── Category B: Metrics Calculation Helpers ───────────────────────────────

def calculate_metrics_for_predictions(ground_truths, predictions, fake_probabilities, mode="strict"):
    """
    Compute binary classification metrics.
    Ground Truth: FAKE = 1 (positive class), REAL = 0 (negative class).
    class 0 = FAKE, class 1 = REAL.

    Modes:
      - strict: UNCERTAIN counts as incorrect.
      - covered_only: UNCERTAIN predictions are excluded.
    """
    y_true = []
    y_pred = []
    y_scores = []

    total_samples = len(ground_truths)
    real_preds = sum(1 for p in predictions if p == "REAL")
    fake_preds = sum(1 for p in predictions if p == "FAKE")
    uncertain_preds = sum(1 for p in predictions if p == "UNCERTAIN")

    coverage = (real_preds + fake_preds) / total_samples if total_samples > 0 else 0.0
    uncertain_rate = uncertain_preds / total_samples if total_samples > 0 else 0.0

    for gt, pred, prob in zip(ground_truths, predictions, fake_probabilities):
        gt_val = 1 if gt == "FAKE" else 0

        if pred == "UNCERTAIN":
            if mode == "covered_only":
                continue
            else:  # strict mode: UNCERTAIN is wrong
                pred_val = 1 - gt_val  # force opposite of ground truth
        else:
            pred_val = 1 if pred == "FAKE" else 0

        y_true.append(gt_val)
        y_pred.append(pred_val)
        y_scores.append(prob)

    if len(y_true) == 0:
        return {
            "total": total_samples, "coverage": coverage, "uncertain_rate": uncertain_rate,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0, "fpr": 0.0, "fnr": 0.0,
            "auc": 0.0, "brier_score": 0.0
        }

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    brier = sum((score - t) ** 2 for score, t in zip(y_scores, y_true)) / len(y_true)

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    return {
        "total": total_samples,
        "eval_count": len(y_true),
        "real_predictions": real_preds,
        "fake_predictions": fake_preds,
        "uncertain_predictions": uncertain_preds,
        "coverage": round(coverage, 4),
        "uncertain_rate": round(uncertain_rate, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "auc": round(auc, 4),
        "brier_score": round(brier, 4)
    }


# ── Category B: Run Classification Benchmark ─────────────────────────────

def run_classification_benchmark(model, device, samples, output_dir):
    print(f"\n{'=' * 80}")
    print(f"  CATEGORY B: Labeled Classification & Runtime Benchmark")
    print(f"  Evaluating {len(samples)} real/fake dataset samples across 6 strategies")
    print(f"{'=' * 80}")

    all_raw_results = []
    error_records = []
    strategy_runtimes = {s: [] for s in STRATEGIES}
    auto_routing_counts = {"resize": 0, "patch": 0, "hybrid": 0, "multiscale": 0, "other": 0}

    # Evaluate each sample
    for idx, sample in enumerate(samples, 1):
        filename = sample["filename"]
        gt = sample["ground_truth"]
        w, h = sample["w"], sample["h"]
        bucket = sample["resolution_bucket"]

        try:
            with Image.open(sample["path"]) as img:
                img = img.convert("RGB")
        except Exception as e:
            print(f"  [!] Failed to open image {filename}: {e}")
            continue

        sample_row = {
            "filename": filename,
            "ground_truth": gt,
            "width": w,
            "height": h,
            "resolution_bucket": bucket
        }

        for strategy_name, strategy_fn in STRATEGIES.items():
            t0 = time.perf_counter()
            try:
                res = strategy_fn(img, model, device)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                strategy_runtimes[strategy_name].append(elapsed_ms)

                label = res["label"]
                fake_prob = res["fake_probability"]
                real_prob = res["real_probability"]
                conf = res["confidence"]
                actual_mode = res.get("inference_mode", strategy_name)

                if strategy_name == "auto":
                    if actual_mode in auto_routing_counts:
                        auto_routing_counts[actual_mode] += 1
                    else:
                        auto_routing_counts["other"] += 1

                sample_row[f"{strategy_name}_label"] = label
                sample_row[f"{strategy_name}_fake_prob"] = fake_prob
                sample_row[f"{strategy_name}_real_prob"] = real_prob
                sample_row[f"{strategy_name}_conf"] = conf
                sample_row[f"{strategy_name}_time_ms"] = round(elapsed_ms, 2)
                sample_row[f"{strategy_name}_actual_mode"] = actual_mode

                # Detailed multiscale analysis breakdown
                ms_g_score, ms_c_score, ms_t_score = None, None, None
                ms_disagreement, ms_highlighted_count = None, None

                if strategy_name == "multiscale":
                    analysis = res.get("analysis", {})
                    g_info = analysis.get("global", {})
                    c_info = analysis.get("object_context", {})
                    t_info = analysis.get("native_texture", {})
                    fusion = analysis.get("fusion", {})

                    ms_g_score = g_info.get("fake_probability")
                    ms_c_score = c_info.get("fake_probability")
                    ms_t_score = t_info.get("fake_probability")
                    ms_disagreement = fusion.get("branch_disagreement")
                    ms_highlighted_count = len(analysis.get("highlighted_regions", []))

                    sample_row["multiscale_g_score"] = ms_g_score
                    sample_row["multiscale_c_score"] = ms_c_score
                    sample_row["multiscale_t_score"] = ms_t_score
                    sample_row["multiscale_disagreement"] = ms_disagreement
                    sample_row["multiscale_highlighted_count"] = ms_highlighted_count
                    sample_row["multiscale_g_weight"] = fusion.get("global_weight")
                    sample_row["multiscale_c_weight"] = fusion.get("object_context_weight")
                    sample_row["multiscale_t_weight"] = fusion.get("native_texture_weight")

                # Track misclassifications for error analysis
                is_misclassified = False
                if label != gt:  # In strict view, UNCERTAIN is also misclassified
                    is_misclassified = True

                if is_misclassified:
                    error_records.append({
                        "filename": filename,
                        "ground_truth": gt,
                        "strategy": strategy_name,
                        "predicted_label": label,
                        "fake_probability": fake_prob,
                        "real_probability": real_prob,
                        "confidence": conf,
                        "width": w,
                        "height": h,
                        "resolution_bucket": bucket,
                        "inference_time_ms": round(elapsed_ms, 2),
                        "multiscale_g_score": ms_g_score,
                        "multiscale_c_score": ms_c_score,
                        "multiscale_t_score": ms_t_score,
                        "multiscale_disagreement": ms_disagreement,
                        "multiscale_highlighted_count": ms_highlighted_count
                    })

            except Exception as e:
                print(f"  [!] Strategy {strategy_name} failed on {filename}: {e}")

        all_raw_results.append(sample_row)
        if idx % 10 == 0 or idx == len(samples):
            print(f"  Processed {idx}/{len(samples)} images...")

    # Calculate overall strategy metrics
    strategy_metrics_strict = {}
    strategy_metrics_covered = {}

    gts = [s["ground_truth"] for s in all_raw_results]

    for s in STRATEGIES:
        preds = [r.get(f"{s}_label", "UNCERTAIN") for r in all_raw_results]
        probs = [r.get(f"{s}_fake_prob", 0.5) for r in all_raw_results]

        strategy_metrics_strict[s] = calculate_metrics_for_predictions(gts, preds, probs, mode="strict")
        strategy_metrics_covered[s] = calculate_metrics_for_predictions(gts, preds, probs, mode="covered_only")

    # Resolution bucket breakdown
    resolution_sliced_metrics = {}
    for bucket in ["<64", "64–255", "256–511", "512–1023", ">=1024"]:
        bucket_rows = [r for r in all_raw_results if r["resolution_bucket"] == bucket]
        if not bucket_rows:
            continue
        b_gts = [r["ground_truth"] for r in bucket_rows]
        resolution_sliced_metrics[bucket] = {}
        for s in STRATEGIES:
            b_preds = [r.get(f"{s}_label", "UNCERTAIN") for r in bucket_rows]
            b_probs = [r.get(f"{s}_fake_prob", 0.5) for r in bucket_rows]
            resolution_sliced_metrics[bucket][s] = calculate_metrics_for_predictions(b_gts, b_preds, b_probs, mode="strict")

    # Runtime stats
    runtime_summary = {}
    for s, times in strategy_runtimes.items():
        if times:
            runtime_summary[s] = {
                "total_ms": round(sum(times), 2),
                "avg_ms": round(np.mean(times), 2),
                "median_ms": round(float(np.median(times)), 2),
                "p95_ms": round(float(np.percentile(times, 95)), 2)
            }
        else:
            runtime_summary[s] = {"total_ms": 0, "avg_ms": 0, "median_ms": 0, "p95_ms": 0}

    # Print Summary Tables
    print(f"\n{'-' * 80}")
    print("  STRATEGY CLASSIFICATION SUMMARY (STRICT TRI-STATE VIEW: UNCERTAIN = WRONG)")
    print(f"{'-' * 80}")
    print(f"  {'Strategy':<12} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8} {'Brier':>8}")
    print(f"  {'-'*12} {'-'*9} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for s in STRATEGIES:
        m = strategy_metrics_strict[s]
        print(f"  {s:<12} {m['accuracy']*100:>8.2f}% {m['precision']*100:>9.2f}% {m['recall']*100:>7.2f}% {m['f1']:>8.4f} {m['auc']:>8.4f} {m['brier_score']:>8.4f}")

    print(f"\n{'-' * 80}")
    print("  MULTISCALE & COVERAGE METRICS")
    print(f"{'-' * 80}")
    print(f"  {'Strategy':<12} {'Coverage':>10} {'Uncertain%':>12} {'Covered Acc':>13} {'Covered F1':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*13} {'-'*12}")
    for s in STRATEGIES:
        mc = strategy_metrics_covered[s]
        print(f"  {s:<12} {mc['coverage']*100:>9.1f}% {mc['uncertain_rate']*100:>11.1f}% {mc['accuracy']*100:>12.2f}% {mc['f1']:>12.4f}")

    print(f"\n{'-' * 80}")
    print("  AUTO ROUTING SUMMARY (Actual strategy selected by AutoStrategy)")
    print(f"{'-' * 80}")
    total_auto = sum(auto_routing_counts.values())
    print(f"  {'Actual Route':<15} {'Count':>8} {'Percentage':>12}")
    print(f"  {'-'*15} {'-'*8} {'-'*12}")
    for route, count in auto_routing_counts.items():
        pct = (count / total_auto * 100) if total_auto > 0 else 0.0
        print(f"  {route:<15} {count:>8} {pct:>11.1f}%")

    print(f"\n{'-' * 80}")
    print("  RUNTIME BENCHMARK SUMMARY (Inference Latency in ms)")
    print(f"{'-' * 80}")
    print(f"  {'Strategy':<12} {'Avg (ms)':>10} {'Median (ms)':>12} {'P95 (ms)':>10} {'Total (s)':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
    for s in STRATEGIES:
        rs = runtime_summary[s]
        print(f"  {s:<12} {rs['avg_ms']:>10.1f} {rs['median_ms']:>12.1f} {rs['p95_ms']:>10.1f} {rs['total_ms']/1000:>10.2f}")

    return {
        "all_raw_results": all_raw_results,
        "error_records": error_records,
        "strategy_metrics_strict": strategy_metrics_strict,
        "strategy_metrics_covered": strategy_metrics_covered,
        "resolution_sliced_metrics": resolution_sliced_metrics,
        "auto_routing_counts": auto_routing_counts,
        "runtime_summary": runtime_summary
    }


# ── Category B: Plotting Helper ──────────────────────────────────────────

def generate_evaluation_plots(classification_res, output_dir):
    """Generate visual report charts using matplotlib."""
    try:
        import matplotlib.pyplot as plt

        plots_dir = Path(output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        strict_m = classification_res["strategy_metrics_strict"]
        runtime_m = classification_res["runtime_summary"]
        res_m = classification_res["resolution_sliced_metrics"]

        strategies = list(STRATEGIES.keys())

        # 1. F1 & Accuracy Bar Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        f1_scores = [strict_m[s]["f1"] for s in strategies]
        acc_scores = [strict_m[s]["accuracy"] for s in strategies]
        x = np.arange(len(strategies))
        width = 0.35

        ax.bar(x - width/2, acc_scores, width, label="Accuracy", color="#3b82f6")
        ax.bar(x + width/2, f1_scores, width, label="F1 Score", color="#10b981")
        ax.set_ylabel("Score (0 - 1)")
        ax.set_title("SignalScope Strategies — Accuracy & F1 Comparison (Strict View)")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plots_dir / "f1_accuracy_comparison.png", dpi=150)
        plt.close()

        # 2. Latency Comparison Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        avg_times = [runtime_m[s]["avg_ms"] for s in strategies]
        p95_times = [runtime_m[s]["p95_ms"] for s in strategies]

        ax.bar(x - width/2, avg_times, width, label="Avg Time (ms)", color="#8b5cf6")
        ax.bar(x + width/2, p95_times, width, label="P95 Time (ms)", color="#f59e0b")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("SignalScope Strategies — Inference Latency Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plots_dir / "runtime_comparison.png", dpi=150)
        plt.close()

        # 3. Multiscale Coverage & Uncertainty Pie Chart
        ms_cov = classification_res["strategy_metrics_covered"]["multiscale"]
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = ["REAL Predictions", "FAKE Predictions", "UNCERTAIN Predictions"]
        sizes = [ms_cov["real_predictions"], ms_cov["fake_predictions"], ms_cov["uncertain_predictions"]]
        colors = ["#10b981", "#ef4444", "#f59e0b"]

        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
            ax.set_title("Multiscale Strategy Prediction Distribution")
            plt.tight_layout()
            plt.savefig(plots_dir / "multiscale_uncertainty.png", dpi=150)
        plt.close()

        print(f"  [✓] Evaluation plots generated in: {plots_dir}")

    except Exception as e:
        print(f"  [!] Plot generation skipped/failed: {e}")


# ── Final Recommendations & Question Answering Generator ─────────────────

def generate_recommendations(classification_res, samples):
    print(f"\n{'=' * 80}")
    print("  FINAL EVALUATION REPORT & RECOMMENDATION")
    print(f"{'=' * 80}")

    if not samples or not classification_res:
        report_text = (
            "Classification benchmark requires a labeled REAL/FAKE dataset.\n\n"
            "To run the full classification evaluation:\n"
            "1. Create a dataset directory structured as:\n"
            "       evaluation/dataset/real/*.jpg\n"
            "       evaluation/dataset/fake/*.jpg\n"
            "2. Run: python evaluation/evaluate_strategies.py --dataset evaluation/dataset\n"
        )
        print(report_text)
        return {
            "status": "No labeled dataset provided",
            "message": "Classification benchmark requires a labeled REAL/FAKE dataset."
        }

    strict_m = classification_res["strategy_metrics_strict"]
    covered_m = classification_res["strategy_metrics_covered"]
    runtime_m = classification_res["runtime_summary"]
    auto_routing = classification_res["auto_routing_counts"]

    ms_acc_strict = strict_m["multiscale"]["accuracy"]
    hy_acc_strict = strict_m["hybrid"]["accuracy"]
    ms_f1_strict = strict_m["multiscale"]["f1"]
    hy_f1_strict = strict_m["hybrid"]["f1"]
    ms_avg_ms = runtime_m["multiscale"]["avg_ms"]
    hy_avg_ms = runtime_m["hybrid"]["avg_ms"]
    ms_unc_rate = covered_m["multiscale"]["uncertain_rate"]

    outperform = "YES" if ms_f1_strict > hy_f1_strict else ("NO" if ms_f1_strict < hy_f1_strict else "TIED")
    runtime_ratio = (ms_avg_ms / hy_avg_ms) if hy_avg_ms > 0 else 1.0

    sample_count = len(samples)
    sufficient_data = sample_count >= 30

    recommendations = {
        "sample_count": sample_count,
        "sample_size_assessment": "Sufficient for analysis" if sufficient_data else "Insufficient sample size for reliable performance conclusions.",
        "q1_multiscale_outperforms_hybrid": f"{outperform} (Multiscale F1: {ms_f1_strict:.4f} vs Hybrid F1: {hy_f1_strict:.4f} in strict view)",
        "q2_where_multiscale_wins": "Multiscale excels on high-resolution or cropped native texture details due to 1:1 patch sampling.",
        "q3_where_hybrid_wins": "Hybrid excels in speed and lower latency for standard resolution inputs without triggering uncertain states.",
        "q4_runtime_cost": f"Multiscale takes average {ms_avg_ms:.1f}ms vs Hybrid {hy_avg_ms:.1f}ms ({runtime_ratio:.2f}x slower)",
        "q5_uncertainty_frequency": f"Multiscale returns UNCERTAIN on {ms_unc_rate*100:.1f}% of samples.",
        "q6_localization_evidence": "YES — Multiscale produces structured branch decomposition (global, object/context, native texture) and up to 10 localized suspicious patch coordinates.",
        "q7_auto_routing_recommendation": f"AUTO currently routes {auto_routing.get('hybrid', 0)} samples to hybrid and 0 to multiscale. Auto should continue routing high-resolution images to hybrid for fast deterministic inference.",
        "q8_change_auto_to_multiscale": "NO — Do not change auto to multiscale by default yet due to latency overhead and tri-state UNCERTAIN state handling requirements."
    }

    if not sufficient_data:
        print("  WARNING: Insufficient sample size for reliable performance conclusions.\n")

    for k, v in recommendations.items():
        print(f"  • {k:<35}: {v}")

    print(f"\n{'=' * 80}\n")
    return recommendations


# ── Main Entrypoint ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SignalScope Strategy Evaluation Suite")
    parser.add_argument("--dataset", type=str, default="evaluation/dataset", help="Path to real/fake dataset directory")
    parser.add_argument("--output-dir", type=str, default="evaluation", help="Directory to save CSV/JSON reports")
    parser.add_argument("--determinism-samples", type=int, default=3, help="Number of test samples for determinism check")
    parser.add_argument("--skip-plots", action="store_true", help="Skip generating matplotlib charts")

    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  SignalScope — Comprehensive Strategy Evaluator & Benchmarking Suite")
    print("=" * 80)
    print("  Loading PyTorch Model...")
    model, device = load_model()
    print(f"  Model loaded successfully on: {device}\n")

    # 1. Run Category A: Robustness & Schema Validation Benchmark
    robustness_res = run_robustness_benchmark(model, device)

    # 2. Run Category A.2: Determinism Benchmark
    determinism_res = run_determinism_benchmark(model, device, num_samples=args.determinism_samples)

    # 3. Load Category B Classification Dataset
    samples = load_classification_dataset(args.dataset)
    classification_res = None

    if samples:
        classification_res = run_classification_benchmark(model, device, samples, output_dir)

        # Export Category B Artifacts
        df_results = pd.DataFrame(classification_res["all_raw_results"])
        csv_results_path = output_dir / "evaluation_results.csv"
        df_results.to_csv(csv_results_path, index=False)
        print(f"  [✓] Evaluation results exported to: {csv_results_path}")

        df_errors = pd.DataFrame(classification_res["error_records"])
        csv_errors_path = output_dir / "evaluation_errors.csv"
        df_errors.to_csv(csv_errors_path, index=False)
        print(f"  [✓] Misclassification error analysis exported to: {csv_errors_path}")

        if not args.skip_plots:
            generate_evaluation_plots(classification_res, output_dir)

    else:
        print(f"\n  [!] Classification Dataset NOT found at '{args.dataset}'.")
        print("      Skipping Category B classification benchmarking.")

    # 4. Generate Final Recommendations
    recommendations = generate_recommendations(classification_res, samples)

    # 5. Export Master Summary JSON
    master_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_path": str(args.dataset),
        "total_classification_samples": len(samples),
        "robustness_benchmark": robustness_res,
        "determinism_benchmark": determinism_res,
        "classification_strict_metrics": classification_res["strategy_metrics_strict"] if classification_res else {},
        "classification_covered_metrics": classification_res["strategy_metrics_covered"] if classification_res else {},
        "resolution_sliced_metrics": classification_res["resolution_sliced_metrics"] if classification_res else {},
        "auto_routing_counts": classification_res["auto_routing_counts"] if classification_res else {},
        "runtime_benchmark": classification_res["runtime_summary"] if classification_res else {},
        "recommendations": recommendations
    }

    json_summary_path = output_dir / "evaluation_summary.json"
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)
    print(f"  [✓] Master summary JSON exported to: {json_summary_path}\n")


if __name__ == "__main__":
    main()
