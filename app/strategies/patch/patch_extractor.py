import random
import math
from typing import List, Tuple
from PIL import Image, ImageOps
import torch
from torchvision import transforms

from app.config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PATCH_N,
)


def prepare_image(image: Image.Image) -> Image.Image:
    """
    Validates and standardizes input PIL image:
    1. Corrects EXIF orientation tags.
    2. Converts RGBA, LA, P, or Grayscale images to 3-channel RGB.
    """
    if not isinstance(image, Image.Image):
        raise ValueError(f"Expected PIL.Image.Image, got {type(image)}")

    # Apply EXIF transpose (handles orientation metadata from phone/camera photos)
    image = ImageOps.exif_transpose(image)

    # Convert to RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image.copy()


def get_inference_transform() -> transforms.Compose:
    """
    Returns torchvision transforms matching training preprocessing pipeline:
    - Resize to IMAGE_SIZE (32, 32) using BICUBIC interpolation
    - ToTensor (scale [0, 255] -> [0.0, 1.0])
    - Normalize with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_patch_transform() -> transforms.Compose:
    """
    Returns torchvision transforms for native resolution patches (no resize):
    - ToTensor
    - Normalize with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Safely preprocesses a PIL Image for standard 32x32 resize inference:
    1. Validates and converts image to 3-channel RGB via prepare_image.
    2. Applies evaluation transforms with bicubic interpolation.
    3. Adds a batch dimension (shape: 1 x 3 x 32 x 32).
    """
    clean_image = prepare_image(image)
    transform = get_inference_transform()
    tensor_image = transform(clean_image)
    return tensor_image.unsqueeze(0)


def extract_native_patches(
    image: Image.Image,
    patch_size: Tuple[int, int] = IMAGE_SIZE,
    n_patches: int = PATCH_N,
    seed: int = 42
) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]]]:
    """
    Extracts native-resolution patch crops from the original image.
    Uses a local deterministic random generator (rng = random.Random(seed)).
    Combines center crop, 4 corner crops, grid crops, and random crops.
    """
    clean_image = prepare_image(image)
    w, h = clean_image.size
    pw, ph = patch_size

    if w < pw or h < ph:
        return [clean_image], [(0, 0, w, h)]

    rng = random.Random(seed)
    crops: List[Image.Image] = []
    coords: List[Tuple[int, int, int, int]] = []
    seen_coords = set()

    def add_crop(x1: int, y1: int):
        x1 = max(0, min(x1, w - pw))
        y1 = max(0, min(y1, h - ph))
        box = (x1, y1, x1 + pw, y1 + ph)
        if box not in seen_coords:
            seen_coords.add(box)
            crops.append(clean_image.crop(box))
            coords.append(box)

    # 1. Center crop
    add_crop((w - pw) // 2, (h - ph) // 2)

    # 2. Four corners
    add_crop(0, 0)
    add_crop(w - pw, 0)
    add_crop(0, h - ph)
    add_crop(w - pw, h - ph)

    # 3. Grid crops
    grid_cols = max(2, int(math.sqrt(n_patches)))
    grid_rows = max(2, int(math.sqrt(n_patches)))
    x_steps = [int(x) for x in range(0, max(1, w - pw + 1), max(1, (w - pw) // max(1, grid_cols - 1)))]
    y_steps = [int(y) for y in range(0, max(1, h - ph + 1), max(1, (h - ph) // max(1, grid_rows - 1)))]

    for gx in x_steps:
        for gy in y_steps:
            add_crop(gx, gy)

    # 4. Fill remaining with random crops
    attempts = 0
    max_attempts = n_patches * 10
    while len(crops) < n_patches and attempts < max_attempts:
        rx = rng.randint(0, w - pw)
        ry = rng.randint(0, h - ph)
        add_crop(rx, ry)
        attempts += 1

    # If still fewer, duplicate with random crops
    while len(crops) < n_patches:
        rx = rng.randint(0, max(0, w - pw))
        ry = rng.randint(0, max(0, h - ph))
        box = (rx, ry, rx + pw, ry + ph)
        crops.append(clean_image.crop(box))
        coords.append(box)

    return crops[:n_patches], coords[:n_patches]
