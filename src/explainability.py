"""
Explainability Module using Grad-CAM (Gradient-weighted Class Activation Mapping).
Visualizes which visual regions (grille, headlamps, badging) triggered the car classification.
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM engine for PyTorch Convolutional Networks.
    Hooks into target layer forward activations and backward gradients.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int = None
    ) -> np.ndarray:
        """
        Computes Grad-CAM activation heatmap normalized to [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Target score for class
        target_score = output[0, target_class_idx]
        target_score.backward(retain_graph=True)

        # Global average pooling on gradients
        gradients = self.gradients[0].detach().cpu().numpy()  # (C, H, W)
        activations = self.activations[0].detach().cpu().numpy()  # (C, H, W)

        weights = np.mean(gradients, axis=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU on combined map
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_on_image(
        self,
        pil_image: Image.Image,
        heatmap: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5,
        cyber_monochrome: bool = False
    ) -> Image.Image:
        """
        Overlays the Grad-CAM heatmap onto the original PIL image.
        Supports standard JET colormap or Cyber Monochrome high-contrast style.
        """
        img_np = np.array(pil_image.convert("RGB"))
        h, w = img_np.shape[:2]

        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap, (w, h))

        if cyber_monochrome:
            # High-contrast glowing white activation heatmap on grayscale image
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            # Mask intensity
            glow_mask = (heatmap_resized * 255).astype(np.uint8)
            glow_colored = np.zeros_like(img_np)
            glow_colored[:, :, 0] = glow_mask  # R
            glow_colored[:, :, 1] = glow_mask  # G
            glow_colored[:, :, 2] = glow_mask  # B

            blended = cv2.addWeighted(gray_3ch, 0.4, glow_colored, 0.6, 0)
            return Image.fromarray(blended)

        # Standard thermal overlay
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)
        return Image.fromarray(blended)
