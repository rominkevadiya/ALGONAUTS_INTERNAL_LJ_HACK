import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.run_experimental_evaluation import evaluate_strategy
from app.predictor import predict_image_hybrid, predict_image_patch_vote, predict_image
from PIL import Image

def generate_mock_unseen_dataset(n_samples=50):
    """
    Mock dataset generator to simulate 'Unseen Generators' split 
    (e.g., Midjourney v6, DALL-E 3, whereas CIFAKE used Stable Diffusion v1.4).
    In a real scenario, this would load from a directory.
    """
    images = []
    labels = []
    rng = np.random.RandomState(99) # Different seed for different dataset

    for i in range(n_samples):
        # We generate images simulating unseen generator textures
        size = (512, 512)
        label = i % 2
        labels.append(label)

        if label == 1:
            # REAL texture
            arr = np.zeros((*size, 3), dtype=np.uint8)
            for r in range(size[0]):
                for c in range(size[1]):
                    arr[r, c, 0] = int(20 + (r / size[0]) * 100)
                    arr[r, c, 1] = int(80 + (c / size[1]) * 100)
                    arr[r, c, 2] = int(120 - (r / size[0]) * 50)
            img = Image.fromarray(arr)
        else:
            # UNSEEN FAKE pattern (e.g. smoother, less noisy than standard CIFAKE, maybe structured grids)
            arr = rng.randint(50, 200, (*size, 3), dtype=np.uint8)
            # Add grid artifacts
            arr[::32, :] = 255
            arr[:, ::32] = 255
            img = Image.fromarray(arr)

        images.append(img)
    return images, labels

def main():
    print("Generating Mock Unseen Generator Dataset...")
    images, labels = generate_mock_unseen_dataset(n_samples=50)

    print("\nEvaluating on Unseen Generator Split using Hybrid Strategy...")
    # Using predict_image_hybrid as it is the most robust
    acc, prec, rec, f1, bal_acc, auc, mean_t = evaluate_strategy(
        "Hybrid Inference", predict_image_hybrid, images, labels
    )
    
    results = {
        "Metric": ["Unseen-Split Accuracy", "Unseen-Split Macro-F1", "Unseen-Split ROC-AUC", "Mean Inference Time (ms)"],
        "Value": [f"{acc*100:.2f}%", f"{f1:.4f}", f"{auc:.4f}", f"{mean_t:.2f} ms"]
    }

    df = pd.DataFrame(results)
    print("\n--- Unseen Generator Benchmark Results ---")
    print(df.to_string(index=False))

    df.to_csv(ROOT_DIR / "evaluation" / "unseen_generator_metrics.csv", index=False)
    print("\nSaved metrics to evaluation/unseen_generator_metrics.csv")


if __name__ == "__main__":
    main()
