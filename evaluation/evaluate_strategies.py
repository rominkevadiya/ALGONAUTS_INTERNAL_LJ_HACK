"""
SignalScope Strategy Evaluation Script
Evaluates each inference strategy individually across multiple synthetic test cases.
Tests include: tiny images, small images, HD resolution, bright/dark lighting, RGBA inputs.

Usage:
    .venv\Scripts\python evaluation/evaluate_strategies.py
"""

import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.model_loader import load_model  # noqa: E402
from app.strategies.resize.resize_strategy import predict_image as predict_resize  # noqa: E402
from app.strategies.patch.patch_strategy import predict_image_patch_vote as predict_patch  # noqa: E402
from app.strategies.tta.tta_strategy import predict_image_tta as predict_tta  # noqa: E402
from app.strategies.hybrid.hybrid_strategy import predict_image_hybrid as predict_hybrid  # noqa: E402
from app.strategies.auto.auto_strategy import predict_image_auto as predict_auto  # noqa: E402


# ── Synthetic Image Generators ────────────────────────────────────────────────

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
}

REQUIRED_KEYS = {"label", "confidence", "fake_probability", "real_probability"}

# ── Run Evaluation ─────────────────────────────────────────────────────────────

def fmt(val):
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def run_evaluation():
    print("=" * 70)
    print("  SignalScope — Individual Strategy Evaluation")
    print("=" * 70)
    print("Loading model...")
    model, device = load_model()
    print(f"  Model loaded on: {device}\n")

    summary_rows = []

    for strategy_name, strategy_fn in STRATEGIES.items():
        print(f"\n{'-' * 70}")
        print(f"  STRATEGY: {strategy_name.upper()}")
        print(f"{'-' * 70}")
        print(f"  {'Test Case':<30} {'Label':<6} {'Fake%':>7} {'Real%':>7} {'Conf%':>7} {'Time(ms)':>9} {'Status'}")
        print(f"  {'-'*30} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'------'}")

        passed, failed = 0, 0
        for tc in TEST_CASES:
            img = make_image(tc["w"], tc["h"], tc["mode"], tc["pattern"])
            try:
                t0 = time.perf_counter()
                result = strategy_fn(img, model, device)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                # Validate output schema
                missing = REQUIRED_KEYS - set(result.keys())
                if missing:
                    raise KeyError(f"Missing keys in result: {missing}")

                label = result["label"]
                fake_p = result["fake_probability"]
                real_p = result["real_probability"]
                conf = result["confidence"]

                # Sanity checks
                assert label in ("FAKE", "REAL"), f"Bad label: {label}"
                assert 0.0 <= fake_p <= 1.0, f"fake_probability out of range: {fake_p}"
                assert 0.0 <= real_p <= 1.0, f"real_probability out of range: {real_p}"
                assert abs(fake_p + real_p - 1.0) < 1e-3, f"Probs don't sum to 1: {fake_p+real_p}"
                assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

                status = "PASS ✓"
                passed += 1
                print(f"  {tc['name']:<30} {label:<6} {fake_p*100:>6.2f}% {real_p*100:>6.2f}% {conf*100:>6.2f}% {elapsed_ms:>8.1f}  {status}")

            except Exception as e:
                failed += 1
                print(f"  {tc['name']:<30} {'ERR':<6} {'---':>7} {'---':>7} {'---':>7} {'---':>9}  FAIL ✗  [{e}]")

            summary_rows.append({
                "strategy": strategy_name,
                "test_case": tc["name"],
                "passed": failed == 0
            })

        total = passed + failed
        print(f"\n  Result: {passed}/{total} passed", "✓" if failed == 0 else f"  ✗ {failed} FAILED")

    print(f"\n{'=' * 70}")
    print("  Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
