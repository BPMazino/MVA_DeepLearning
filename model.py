"""
Implementation of the YOLOv1 architecture with Batch Normalization.
"""

import torch
import torch.nn as nn
from typing import List, Union, Tuple

# Define a type alias for an item in the architecture configuration.
ArchitectureItem = Union[Tuple[int, int, int, int], str, list]

# -----------------------------------------------------------------------------
# Architecture Configuration
# -----------------------------------------------------------------------------
# Each tuple represents a convolutional layer configuration:
#   (kernel_size, number_of_filters, stride, padding)
#
# The string "M" indicates a max-pooling layer with kernel size 2x2 and stride 2x2.
#
# A list represents a sequence of layers to be repeated. It contains two tuples
# (each with the convolution parameters) followed by the number of repeats.
architecture_config: List[ArchitectureItem] = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

# -----------------------------------------------------------------------------
# CNN Block Definition
# -----------------------------------------------------------------------------
class CNNBlock(nn.Module):
    """
    A convolutional block that performs convolution followed by batch normalization
    and LeakyReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, **conv_kwargs) -> None:
        """
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            **conv_kwargs: Additional keyword arguments for nn.Conv2d (e.g., kernel_size, stride, padding).
        """
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **conv_kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.batchnorm(self.conv(x)))

# -----------------------------------------------------------------------------
# YOLOv1 Model Definition
# -----------------------------------------------------------------------------
class Yolov1(nn.Module):
    """
    Implementation of the YOLOv1 model.
    """
    def __init__(self, in_channels: int = 3, split_size: int = 7, num_boxes: int = 2, num_classes: int = 20) -> None:
        """
        Parameters:
            in_channels (int): Number of input channels (default is 3 for RGB images).
            split_size (int): Grid size (S) to divide the image.
            num_boxes (int): Number of bounding boxes per grid cell.
            num_classes (int): Number of classes for detection.
        """
        super(Yolov1, self).__init__()
        self.in_channels = in_channels
        self.architecture = architecture_config

        # Feature extractor (Darknet)
        self.darknet = self._create_conv_layers(self.architecture)
        # Fully connected layers (head)
        self.fcs = self._create_fcs(split_size, num_boxes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Parameters:
            x (torch.Tensor): Input image tensor.
            
        Returns:
            torch.Tensor: Output predictions.
        """
        x = self.darknet(x)
        x = self.fcs(x)
        return x

    def _create_conv_layers(self, architecture: List[ArchitectureItem]) -> nn.Sequential:
        """
        Creates the convolutional layers based on the architecture configuration.
        
        Parameters:
            architecture (List[ArchitectureItem]): List defining the layer configuration.
            
        Returns:
            nn.Sequential: A sequential container of convolutional layers.
        """
        layers = []
        in_channels = self.in_channels

        for layer in architecture:
            if isinstance(layer, tuple):
                kernel_size, filters, stride, padding = layer
                layers.append(
                    CNNBlock(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding)
                )
                in_channels = filters

            elif isinstance(layer, str) and layer == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            elif isinstance(layer, list):
                conv1, conv2, num_repeats = layer
                for _ in range(num_repeats):
                    # First convolutional block in the repeated sequence.
                    k1, f1, s1, p1 = conv1
                    layers.append(
                        CNNBlock(in_channels, f1, kernel_size=k1, stride=s1, padding=p1)
                    )
                    # Second convolutional block in the repeated sequence.
                    k2, f2, s2, p2 = conv2
                    layers.append(
                        CNNBlock(f1, f2, kernel_size=k2, stride=s2, padding=p2)
                    )
                    in_channels = f2

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size: int, num_boxes: int, num_classes: int) -> nn.Sequential:
        """
        Creates the fully connected layers.
        
        Parameters:
            split_size (int): Grid size (S).
            num_boxes (int): Number of bounding boxes per grid cell.
            num_classes (int): Number of classes.
            
        Returns:
            nn.Sequential: A sequential container of fully connected layers.
        """
        S, B, C = split_size, num_boxes, num_classes
        # In the original YOLO paper, the first FC layer maps from 1024*S*S to 4096, 
        # followed by a LeakyReLU, and then to S*S*(B*5+C).
        # Here, we use an intermediate dimension of 496 as a slight modification.
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5))
        )

# ---------------------------------- Testing ----------------------------------

def test():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
    x = torch.randn((2, 3, 448, 448))
    out = model(x)
    print("Output shape:", out.shape)  
    # Should be [2, 1470] when S=7, B=2, C=20

if __name__ == "__main__":
    test()
