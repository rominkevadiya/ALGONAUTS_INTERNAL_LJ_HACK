import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.model_loader import load_model
from app.config import IMAGENET_MEAN, IMAGENET_STD
from evaluation.run_experimental_evaluation import generate_synthetic_benchmark_dataset

def fgsm_attack(image_tensor, epsilon, data_grad):
    """
    FGSM attack code.
    image_tensor: original image tensor
    epsilon: attack magnitude
    data_grad: gradient of the data w.r.t the loss
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image_tensor + epsilon * sign_data_grad
    # We should ideally clamp to valid image range, but since input is normalized with ImageNet mean/std,
    # exact clamping is complex. We'll do a basic attack for demonstration.
    return perturbed_image

def main():
    print("Loading model for adversarial analysis...")
    model, device = load_model()
    model.eval()

    print("Generating synthetic benchmark dataset (N=40)...")
    images, labels = generate_synthetic_benchmark_dataset(n_samples=40)

    # Transform matching ResNet-50 input
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    results = []

    criterion = nn.CrossEntropyLoss()

    for eps in epsilons:
        correct = 0
        total = 0
        for img, label in zip(images, labels):
            img_tensor = transform(img).unsqueeze(0).to(device)
            img_tensor.requires_grad = True
            
            target_label = torch.tensor([label], dtype=torch.long).to(device)

            output = model(img_tensor)
            init_pred = output.max(1, keepdim=True)[1] 
            
            # If initially incorrect, skip for robustness calculation
            if init_pred.item() != label:
                continue

            loss = criterion(output, target_label)
            model.zero_grad()
            loss.backward()
            
            data_grad = img_tensor.grad.data
            perturbed_data = fgsm_attack(img_tensor, eps, data_grad)
            
            output_perturbed = model(perturbed_data)
            final_pred = output_perturbed.max(1, keepdim=True)[1]
            
            if final_pred.item() == label:
                correct += 1
            total += 1
        
        acc = correct / float(total) if total > 0 else 0
        print(f"Epsilon: {eps}\tTest Accuracy = {correct} / {total} = {acc:.4f}")
        results.append({'Epsilon': eps, 'Accuracy': acc, 'Total_Attacked': total})

    df = pd.DataFrame(results)
    df.to_csv(ROOT_DIR / "evaluation" / "adversarial_results.csv", index=False)
    print(f"\nAdversarial robustness results saved to evaluation/adversarial_results.csv")

if __name__ == '__main__':
    main()
