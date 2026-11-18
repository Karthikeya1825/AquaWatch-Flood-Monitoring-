"""
AquaWatch - Semantic Segmentation Architecture (U-Net with ResNet34 Backbone)

Architecture:
- U-Net with pretrained ResNet34 encoder from segmentation_models_pytorch (smp)
- Adapts the first convolutional layer (conv1) from 3 channels (RGB) to 4 channels (RGB + NIR)
  Pretrained ImageNet weights are preserved for RGB channels, and the 4th channel (NIR) is initialized
  with the average of RGB weights for seamless transfer learning.
- Output: Single-channel raw logit map (512x512) for binary water vs land classification.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional


class AquaWatchUNet(nn.Module):
    """
    U-Net segmentation network with a pretrained ResNet34 encoder.
    Optimized for multi-spectral Earth Observation (Sentinel-2).
    """
    def __init__(
        self,
        in_channels: int = 4,
        classes: int = 1,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        activation: Optional[str] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.encoder_name = encoder_name

        # Initialize U-Net via segmentation_models_pytorch
        # When in_channels != 3, smp automatically modifies the first conv layer (conv1)
        # and transfers pretrained weights for the existing channels
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, in_channels, H, W)
        Returns:
            Raw logits tensor of shape (B, classes, H, W)
        """
        return self.model(x)

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary segmentation mask.
        Returns tensor of shape (B, 1, H, W) with {0.0, 1.0} values.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
        return preds


class SimpleUNet(nn.Module):
    """
    Lightweight Plain U-Net baseline without pretrained backbone (Optional Baseline 2).
    Useful for studying the benefit of transfer learning.
    """
    def __init__(self, in_channels: int = 4, classes: int = 1, base_filters: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 4, base_filters * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base_filters * 8, base_filters * 4)
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_filters * 2, base_filters)

        self.final_conv = nn.Conv2d(base_filters, classes, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final_conv(d1)


def build_model(
    model_type: str = "unet_resnet34",
    in_channels: int = 4,
    classes: int = 1,
    encoder_weights: Optional[str] = "imagenet"
) -> nn.Module:
    """Factory function to build the requested segmentation model."""
    if model_type == "unet_resnet34":
        return AquaWatchUNet(
            in_channels=in_channels,
            classes=classes,
            encoder_name="resnet34",
            encoder_weights=encoder_weights
        )
    elif model_type == "simple_unet":
        return SimpleUNet(in_channels=in_channels, classes=classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    # Test model instantiation and forward pass
    print("Testing AquaWatchUNet with 4-band input (RGB+NIR)...")
    model = build_model("unet_resnet34", in_channels=4, classes=1, encoder_weights=None)
    dummy_input = torch.randn(2, 4, 512, 512)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape} -> Output logit shape: {output.shape}")
    assert output.shape == (2, 1, 512, 512), f"Unexpected output shape: {output.shape}"
    print("AquaWatchUNet test passed successfully!")
