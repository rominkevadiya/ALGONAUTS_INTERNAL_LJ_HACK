"""
SignalScope Grad-CAM Diagnostic
Generates a class-activation heatmap overlay explaining which regions of the
32x32 model input most influenced the network's decision.
"""

import cv2
import numpy as np
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initializes GradCAM.
        Args:
            model: PyTorch model
            target_layer: The layer to compute the gradients from (e.g. model.layer2)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.fwd_handle = self.target_layer.register_forward_hook(self.save_activation)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self.save_gradient)

    def remove_hooks(self):
        if hasattr(self, 'fwd_handle') and self.fwd_handle:
            self.fwd_handle.remove()
        if hasattr(self, 'bwd_handle') and self.bwd_handle:
            self.bwd_handle.remove()

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generates the Grad-CAM heatmap.
        Args:
            input_tensor: Normalized input image tensor (1, C, H, W)
            target_class: Class index to explain. None = the class the model
                          actually predicted for this input (keeps the heatmap
                          consistent with the verdict shown next to it, instead
                          of always explaining a fixed class).
        """
        self.model.eval()

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()

        # Target for backprop
        target = output[0, target_class]
        target.backward(retain_graph=True)

        # Get pooled gradients
        gradients = self.gradients.detach().cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.detach().cpu().numpy()[0]  # (C, H, W)

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))  # (C,)

        # Weight the activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU to only keep features that have a positive influence
        cam = np.maximum(cam, 0)

        # Normalize between 0 and 1
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max

        return cam, target_class


def apply_colormap_on_image(
    org_im: Image.Image,
    activation: np.ndarray,
    colormap_name: int = cv2.COLORMAP_TURBO,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Applies a colormap heatmap onto an original image.

    Args:
        org_im: Original PIL Image
        activation: 2D numpy array of activations (0 to 1), native resolution
                    of the target conv layer (coarse — a handful of pixels).
        colormap_name: OpenCV colormap. Defaults to Turbo rather than the
                        classic Jet colormap: Jet's non-uniform luminance ramp
                        creates false banding and perceived "hotspots" that do
                        not correspond to the underlying activation magnitude,
                        which is exactly the "looks wrong even though the math
                        is right" complaint Grad-CAM overlays are known for.
                        Turbo is a drop-in OpenCV colormap with far better
                        perceptual uniformity.
        alpha: Heatmap opacity in the blend (0-1).
    """
    org_im_cv = np.array(org_im.convert('RGB'))
    org_im_cv = org_im_cv[:, :, ::-1]  # RGB to BGR for OpenCV
    h, w = org_im_cv.shape[:2]

    # The activation map comes straight off a conv layer fed a 32x32 input, so
    # it is only a handful of pixels across (e.g. 16x16 for layer2, 8x8 for
    # layer3). Upscaling that directly to the original image's resolution with
    # the default nearest/linear resize produces hard, blocky squares instead
    # of a smooth heatmap. Two changes fix the visual blockiness without
    # changing what the heatmap mathematically represents:
    #   1. INTER_CUBIC for a smooth upsample instead of blocky linear/nearest.
    #   2. A Gaussian blur sized relative to one "cell" of the coarse grid, to
    #      soften the residual grid pattern cubic interpolation alone leaves
    #      behind at large upscale factors.
    activation_resized = cv2.resize(activation, (w, h), interpolation=cv2.INTER_CUBIC)
    activation_resized = np.clip(activation_resized, 0.0, 1.0)

    cell_w = max(1, w // max(1, activation.shape[1]))
    cell_h = max(1, h // max(1, activation.shape[0]))
    blur_k = max(3, (min(cell_w, cell_h) // 2) | 1)  # odd kernel size
    activation_resized = cv2.GaussianBlur(activation_resized, (blur_k, blur_k), 0)

    # Convert to 8-bit heatmap
    heatmap = np.uint8(255 * activation_resized)
    heatmap = cv2.applyColorMap(heatmap, colormap_name)

    # Saturating blend (cv2.addWeighted clips to [0, 255] instead of relying
    # on the blend weights summing to 1 to keep values in range by luck).
    superimposed_img = cv2.addWeighted(heatmap, alpha, org_im_cv, 1.0 - alpha, 0)

    # Convert back to PIL
    superimposed_img = superimposed_img[:, :, ::-1]  # BGR to RGB
    return Image.fromarray(superimposed_img)


def _select_target_layer(model):
    """
    Picks the finest-resolution conv block that still carries useful semantic
    features. For the CIFAR-adapted 32x32 stem (conv1 stride 1, maxpool =
    Identity), spatial size halves at layer2/3/4 only, so:
        layer1 = 32x32 (too shallow — mostly edges/color, weak class signal)
        layer2 = 16x16 (good balance — the fix used here)
        layer3 =  8x8  (previous default — noticeably blockier once upscaled)
        layer4 =  4x4  (most semantic, extremely coarse)
    layer2 is used as the primary target: it roughly doubles the heatmap's
    linear resolution versus the previous layer3 default, which is the single
    biggest lever on visual blockiness, while still sitting deep enough in the
    network to carry meaningful class-discriminative activation.
    """
    for attr in ("layer2", "layer3", "layer4", "layer1"):
        if hasattr(model, attr):
            return getattr(model, attr)[-1]
    return list(model.modules())[-2]


def run_grad_cam(model, image: Image.Image, target_class: int | None = None) -> Image.Image:
    """
    Helper function to generate a Grad-CAM heatmap overlay for the given image.

    Args:
        target_class: Class to explain (0 = FAKE, 1 = REAL). Defaults to None,
                       which explains whichever class this forward pass itself
                       predicts — so the heatmap always matches the verdict
                       shown beside it instead of always rendering a "why is
                       this FAKE" explanation even when the model said REAL.
    """
    from torchvision import transforms
    from app.config import IMAGENET_MEAN, IMAGENET_STD

    target_layer = _select_target_layer(model)
    grad_cam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Must match model's training size (CIFAKE is 32x32)
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # We must ensure the model parameters require_grad for backward to work
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad = True

    try:
        input_tensor = transform(image.convert("RGB")).unsqueeze(0)
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        heatmap, _resolved_class = grad_cam.generate_heatmap(input_tensor, target_class)

    finally:
        # Clean up hooks
        grad_cam.remove_hooks()

        # Restore requires_grad
        for name, param in model.named_parameters():
            param.requires_grad = original_requires_grad[name]

    return apply_colormap_on_image(image, heatmap)
