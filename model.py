import torch 
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A modular CNN  block:
    Conv2d -> LeakyReLU -> MaxPool2d
    (No BatchNorm2d is used because it is not used in the original paper of YOLOv1 but it is used in YOLOv2)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, use_pool = False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if use_pool else None

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if self.pool:
            x = self.pool(x)
        return x
    

class YOLOv1(nn.Module):
    """
    YOLOv1 model
    
    Parameters:
    S: int, number of grid cells in each dimension
    B: int, number of bounding boxes predicted by each grid cell
    C: int, number of classes
    """
    
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.conv_layers = self._build_conv_layers()
        self.fc_layers = self._build_fc_layers()
        
    def _build_conv_layers(self):
        return nn.Sequential(
            # First layer
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3, use_pool=True),
            
            # Second layer
            ConvBlock(64, 192, kernel_size=3, padding=1, use_pool=True),
            
            # Third layer
            ConvBlock(192, 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1, use_pool=True),
            
            # Fourth layer (4 repetitions)
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1, use_pool=True),
            
            # Fifth layer (2 repetitions)
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            
            # Final layers
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1),  # Downsample to 7x7
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1)
        )

        
    def _build_fc_layers(self):
        input_size = 1024 * self.S * self.S
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 4096), # 7*7*1024 is the shape of the output of the last ConvBlock
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5)), # 7*7*(20+2*5) = 1470
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        print(f"Shape after conv layers: {x.shape}")
        x = self.fc_layers(x)
        x =x.view(-1, self.S, self.S, self.C + self.B*5)
        return x
        

        
