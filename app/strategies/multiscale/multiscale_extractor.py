"""
SignalScope Multi-Scale Patch Extractor & Coordinate Remapper
Extracts 128x128 Object/Context patches (16 total) and Native 32x32 Texture patches.
Maps patch coordinates accurately back to the original image coordinate system.
"""

from typing import List, Tuple, Dict, Any
from PIL import Image
import math

from app.config import (
    IMAGE_SIZE,
    MULTISCALE_CONTEXT_SIZE,
    MAX_NATIVE_PATCHES,
)
from app.strategies.patch.patch_extractor import prepare_image


def extract_object_context_patches(
    image: Image.Image
) -> Tuple[List[Image.Image], List[Dict[str, int]], List[Dict[str, int]]]:
    """
    Object/Context Analysis Branch:
    1. Converts image to RGB.
    2. Resizes image to 128x128.
    3. Divides 128x128 image into 16 non-overlapping 32x32 patches (4x4 grid).
    4. Maps patch coordinates from 128x128 space back to original image space.

    Returns:
        - List of 16 PIL Image patches (each 32x32 px)
        - List of 16 128x128 space coordinate dicts
        - List of 16 mapped original-image space coordinate dicts
    """
    clean_img = prepare_image(image)
    orig_w, orig_h = clean_img.size

    # Resize to 128x128 context view using BICUBIC interpolation
    img_128 = clean_img.resize(MULTISCALE_CONTEXT_SIZE, Image.Resampling.BICUBIC)

    patches: List[Image.Image] = []
    coords_128: List[Dict[str, int]] = []
    coords_orig: List[Dict[str, int]] = []

    scale_x = orig_w / 128.0
    scale_y = orig_h / 128.0

    # 4x4 grid of 32x32 patches
    for row in range(4):
        for col in range(4):
            x1_128 = col * 32
            y1_128 = row * 32
            x2_128 = x1_128 + 32
            y2_128 = y1_128 + 32

            patch = img_128.crop((x1_128, y1_128, x2_128, y2_128))
            patches.append(patch)

            coords_128.append({
                "x": x1_128,
                "y": y1_128,
                "width": 32,
                "height": 32
            })

            # Remap coordinates back to original image resolution
            x_orig = int(round(x1_128 * scale_x))
            y_orig = int(round(y1_128 * scale_y))
            w_orig = int(round(32 * scale_x))
            h_orig = int(round(32 * scale_y))

            coords_orig.append({
                "x": x_orig,
                "y": y_orig,
                "width": w_orig,
                "height": h_orig
            })

    return patches, coords_128, coords_orig


def extract_native_texture_patches(
    image: Image.Image,
    patch_size: Tuple[int, int] = IMAGE_SIZE,
    stride: int = 32,
    max_patches: int = MAX_NATIVE_PATCHES
) -> Tuple[List[Image.Image], List[Dict[str, int]], int]:
    """
    Native Texture Analysis Branch:
    Extracts native-resolution 32x32 patches directly from the original image.
    Uses non-overlapping stride (default: 32).
    Ignores incomplete border patches if image dimensions are not divisible by 32.

    Returns:
        - List of native 32x32 PIL Image patches
        - List of coordinate dicts in original image space
        - Count of skipped border patches
    """
    clean_img = prepare_image(image)
    orig_w, orig_h = clean_img.size
    pw, ph = patch_size

    # Handle image smaller than patch size (32x32)
    if orig_w < pw or orig_h < ph:
        return [], [], 0

    x_steps = list(range(0, orig_w - pw + 1, stride))
    y_steps = list(range(0, orig_h - ph + 1, stride))

    # Calculate skipped pixels/incomplete border patches
    skipped_x = orig_w - (x_steps[-1] + pw) if x_steps else 0
    skipped_y = orig_h - (y_steps[-1] + ph) if y_steps else 0
    skipped_count = (1 if skipped_x > 0 else 0) + (1 if skipped_y > 0 else 0)

    patches: List[Image.Image] = []
    coords: List[Dict[str, int]] = []

    for y in y_steps:
        for x in x_steps:
            box = (x, y, x + pw, y + ph)
            crop = clean_img.crop(box)
            patches.append(crop)
            coords.append({
                "x": x,
                "y": y,
                "width": pw,
                "height": ph
            })

    # If patch count exceeds max_patches limit, sample deterministically with uniform grid step
    if len(patches) > max_patches:
        step = len(patches) / max_patches
        selected_indices = [int(i * step) for i in range(max_patches)]
        patches = [patches[i] for i in selected_indices]
        coords = [coords[i] for i in selected_indices]

    return patches, coords, skipped_count
