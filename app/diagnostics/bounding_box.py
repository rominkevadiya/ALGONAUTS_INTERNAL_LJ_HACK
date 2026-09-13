"""
SignalScope Bounding Box Visualization Utility
Annotates uploaded images with bounding boxes over high-scoring inference regions.
"""

from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont


def render_highlighted_regions(
    image: Image.Image,
    regions: List[Dict[str, Any]],
    box_color: str = "#ef4444",
    max_boxes: int = 8
) -> Image.Image:
    """
    Draws bounding boxes around high-scoring inference regions on a copy of the original PIL image.

    Args:
        image: Original PIL Image
        regions: List of region dicts containing x, y, width, height, fake_probability, source
        box_color: Hex color string for bounding box borders (default: red #ef4444)
        max_boxes: Maximum number of boxes to draw (default: 8)

    Returns:
        Annotated copy of PIL Image with bounding boxes and confidence badges.
    """
    if not regions or not isinstance(image, Image.Image):
        return image

    annotated = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Sort regions by fake probability descending and limit to max_boxes
    top_regions = sorted(regions, key=lambda r: r.get("fake_probability", 0.0), reverse=True)[:max_boxes]

    img_w, img_h = annotated.size
    line_width = max(2, min(6, int(min(img_w, img_h) / 200)))

    for idx, region in enumerate(top_regions):
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("width", 32))
        h = int(region.get("height", 32))
        prob = float(region.get("fake_probability", 0.0))
        source = region.get("source", "inference")

        # Clamp box coordinates to image dimensions
        x1 = max(0, min(x, img_w - 1))
        y1 = max(0, min(y, img_h - 1))
        x2 = max(x1 + 1, min(x + w, img_w))
        y2 = max(y1 + 1, min(y + h, img_h))

        # Semi-transparent fill box
        draw.rectangle([x1, y1, x2, y2], fill=(239, 68, 68, 35), outline=(239, 68, 68, 220), width=line_width)

        # Label tag background
        label_text = f"#{idx+1} {prob*100:.0f}% FAKE ({source})"
        text_bbox = draw.textbbox((x1, y1), label_text)
        tb_w = text_bbox[2] - text_bbox[0] + 8
        tb_h = text_bbox[3] - text_bbox[1] + 4

        tag_y1 = max(0, y1 - tb_h)
        draw.rectangle([x1, tag_y1, x1 + tb_w, tag_y1 + tb_h], fill=(239, 68, 68, 230))
        draw.text((x1 + 4, tag_y1 + 2), label_text, fill=(255, 255, 255, 255))

    # Composite overlay onto annotated image
    final_img = Image.alpha_composite(annotated, overlay).convert("RGB")
    return final_img
