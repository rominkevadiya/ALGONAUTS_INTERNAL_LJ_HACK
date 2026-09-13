import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
from sklearn.metrics import accuracy_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.model_loader import load_model
from app.predictor import predict_image_auto
from evaluation.run_experimental_evaluation import generate_synthetic_benchmark_dataset


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Applies JPEG compression to an image."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def apply_resizing(image: Image.Image, scale_factor: float) -> Image.Image:
    """Downscales then upscales an image to simulate resolution loss."""
    w, h = image.size
    new_w = max(16, int(w * scale_factor))
    new_h = max(16, int(h * scale_factor))
    downscaled = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    upscaled = downscaled.resize((w, h), Image.Resampling.BICUBIC)
    return upscaled


def evaluate_degradation(images, labels, model, device, degradation_fn, levels, param_name):
    results = []
    
    for level in levels:
        preds = []
        for img in images:
            deg_img = degradation_fn(img, level)
            res = predict_image_auto(deg_img, model=model, device=device)
            pred_label = 0 if res["label"] == "FAKE" else 1
            preds.append(pred_label)
        
        acc = accuracy_score(labels, preds)
        results.append({param_name: level, "Accuracy": acc})
        print(f"Evaluated {param_name}={level} -> Accuracy: {acc*100:.2f}%")
        
    return pd.DataFrame(results)


def main():
    print("Loading model for degradation analysis...")
    model, device = load_model()
    
    print("Generating synthetic benchmark dataset (N=40)...")
    images, labels = generate_synthetic_benchmark_dataset(n_samples=40)
    
    # 1. JPEG Compression Degradation
    print("\n--- Running JPEG Compression Analysis ---")
    jpeg_levels = [100, 90, 70, 50, 30, 10]
    df_jpeg = evaluate_degradation(images, labels, model, device, apply_jpeg_compression, jpeg_levels, "JPEG_Quality")
    
    # 2. Resize Degradation
    print("\n--- Running Resize Degradation Analysis ---")
    scale_levels = [1.0, 0.75, 0.5, 0.25, 0.1]
    df_resize = evaluate_degradation(images, labels, model, device, apply_resizing, scale_levels, "Scale_Factor")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(df_jpeg["JPEG_Quality"], df_jpeg["Accuracy"], marker='o', color='b')
    ax1.set_title("Robustness to JPEG Compression")
    ax1.set_xlabel("JPEG Quality (Higher is better)")
    ax1.set_ylabel("Accuracy")
    ax1.invert_xaxis()
    ax1.grid(True)
    
    ax2.plot(df_resize["Scale_Factor"], df_resize["Accuracy"], marker='s', color='r')
    ax2.set_title("Robustness to Resolution Loss")
    ax2.set_xlabel("Scale Factor (1.0 = Original)")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)
    
    plt.tight_layout()
    output_path = ROOT_DIR / "evaluation" / "degradation_vs_accuracy.png"
    plt.savefig(output_path)
    print(f"\nSaved plot to {output_path}")

    # Save CSV
    df_jpeg.to_csv(ROOT_DIR / "evaluation" / "degradation_jpeg_results.csv", index=False)
    df_resize.to_csv(ROOT_DIR / "evaluation" / "degradation_resize_results.csv", index=False)


if __name__ == "__main__":
    main()
