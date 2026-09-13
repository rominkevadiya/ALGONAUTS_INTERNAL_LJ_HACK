import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initializes GradCAM.
        Args:
            model: PyTorch model
            target_layer: The layer to compute the gradients from (e.g. model.layer4)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class):
        """
        Generates the Grad-CAM heatmap.
        Args:
            input_tensor: Normalized input image tensor (1, C, H, W)
            target_class: The target class index to generate the heatmap for.
        """
        # Ensure model is in eval mode, but requires_grad must be enabled for the input if needed.
        # Actually, for Grad-CAM we just need to backprop from the output, so the input doesn't strictly need requires_grad,
        # but the network parameters do. If the model was loaded strictly for eval, we might need to enable gradients.
        self.model.eval()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        
        # Target for backprop
        target = output[0, target_class]
        target.backward(retain_graph=True)
        
        # Get pooled gradients
        gradients = self.gradients.detach().cpu().numpy()[0] # (C, H, W)
        activations = self.activations.detach().cpu().numpy()[0] # (C, H, W)
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2)) # (C,)
        
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
            
        return cam

def apply_colormap_on_image(org_im: Image.Image, activation: np.ndarray, colormap_name: int = cv2.COLORMAP_JET) -> Image.Image:
    """
    Applies a colormap heatmap onto an original image.
    Args:
        org_im: Original PIL Image
        activation: 2D numpy array of activations (0 to 1)
    """
    org_im_cv = np.array(org_im.convert('RGB'))
    org_im_cv = org_im_cv[:, :, ::-1] # RGB to BGR for OpenCV
    
    # Resize activation to match image size
    activation_resized = cv2.resize(activation, (org_im_cv.shape[1], org_im_cv.shape[0]))
    
    # Convert to 8-bit heatmap
    heatmap = np.uint8(255 * activation_resized)
    heatmap = cv2.applyColorMap(heatmap, colormap_name)
    
    # Superimpose
    superimposed_img = heatmap * 0.4 + org_im_cv * 0.6
    superimposed_img = np.uint8(superimposed_img)
    
    # Convert back to PIL
    superimposed_img = superimposed_img[:, :, ::-1] # BGR to RGB
    return Image.fromarray(superimposed_img)

def run_grad_cam(model, image: Image.Image, target_class: int = 0) -> Image.Image:
    """
    Helper function to generate a Grad-CAM heatmap overlay for the given image.
    By default targets the FAKE class (0).
    """
    from torchvision import transforms
    from app.config import IMAGENET_MEAN, IMAGENET_STD
    
    # Target the last bottleneck layer of ResNet-50
    # The actual attribute name depends on the model architecture, for standard ResNet it's layer4[-1]
    target_layer = None
    if hasattr(model, 'layer4'):
        target_layer = model.layer4[-1]
    else:
        # Fallback if it's wrapped
        target_layer = list(model.modules())[-2]

    grad_cam = GradCAM(model, target_layer)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # standard size for visualization
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    # We must ensure the model parameters require_grad for backward to work
    for param in model.parameters():
        param.requires_grad = True
        
    input_tensor = transform(image.convert("RGB")).unsqueeze(0)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    heatmap = grad_cam.generate_heatmap(input_tensor, target_class)
    
    # Clean up hooks
    if grad_cam.target_layer:
        # We can't easily unregister without saving the hook handle, but for Streamlit it's okay for one-offs.
        pass
        
    return apply_colormap_on_image(image, heatmap)
