"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for EfficientNet-B0 Weather Classifier.
Visualizes which regions of the image (clouds, sky, rain streaks, sun disc)
drove the model's prediction.
"""

import base64
import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class WeatherGradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.model.eval()
        if target_layer is None:
            # Final convolutional stage in EfficientNet-B0
            self.target_layer = self.model.backbone.features[-1]
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._hook_forward()
        self._hook_backward()

    def _hook_forward(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        self.target_layer.register_forward_hook(forward_hook)

    def _hook_backward(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate raw Grad-CAM heatmap normalized to [0.0, 1.0].
        input_tensor: shape (1, 3, H, W)
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)  # (1, num_classes)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward()

        # Global average pooling of gradients
        # gradients: (1, C, H', W'), activations: (1, C, H', W')
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Upsample to input tensor resolution
        _, _, h, w = input_tensor.shape
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class

    def overlay_on_image(
        self,
        original_pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> Image.Image:
        """
        Blend raw heatmap onto original RGB PIL image.
        """
        orig_w, orig_h = original_pil_image.size
        # Resize heatmap to match original image dimensions
        resized_cam = cv2.resize(heatmap, (orig_w, orig_h))
        cam_uint8 = np.uint8(255 * resized_cam)
        heatmap_colored = cv2.applyColorMap(cam_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        orig_np = np.array(original_pil_image)
        blended = (alpha * heatmap_colored + (1.0 - alpha) * orig_np).astype(np.uint8)
        return Image.fromarray(blended)


def pil_to_base64_data_uri(pil_img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"
