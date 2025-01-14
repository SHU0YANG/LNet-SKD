import torch
import torch.nn as nn
import torch.nn.functional as F


class LNet(nn.Module):
    def __init__(self, num_features=41, num_classes=5, in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, type=1):
        super(LNet, self).__init__()
        self.num_features = num_features
        self.conv1 = DeepMax(in_channels, out_channels, kernel_size, stride, padding, type) # 1_32
        self.conv2 = DeepMax(out_channels, out_channels*2, kernel_size, stride, padding, type)# 32_64
        # self.conv3 = DeepMax(out_channels*2, out_channels*4, kernel_size, stride, padding, type)
        self.bn = nn.BatchNorm1d(out_channels*2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flt = nn.Flatten()
        self.classifier = nn.Linear(out_channels*2, num_classes)
    def forward(self,x):
        x = x.reshape(x.size(0),1, self.num_features)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.flt(x)
        x = self.classifier(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class MFM(nn.Module):
    """Max-Feature-Map (MFM) layer"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, choise: int = 0):
        super(MFM, self).__init__()
        self.out_channels = out_channels
        if choise == 0:
            self.filter = nn.Conv1d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        elif choise == 1: 
            self.filter = DSC(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MFM layer."""
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class DSC(nn.Module):
    """Depthwise Separable Convolution (DSConv) Layer"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(DSC, self).__init__()
        self.depth_conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels) 
        self.point_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DSC layer."""
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class DeepMax(nn.Module):
    """DeepMax block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, choise: int = 1):
        super(DeepMax, self).__init__()
        self.conv = MFM(in_channels, out_channels, kernel_size, stride, padding, choise)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DeepMax layer."""
        x = self.conv(x)
        x = self.pool(x)
        return x

class LNet(nn.Module):
    """LNet model architecture."""
    
    def __init__(self, num_features: int, num_classes: int, in_channels: int = 1, out_channels: int = 32, kernel_size: int = 3, stride: int = 1, padding: int = 1, choise: int = 1):
        super(LNet, self).__init__()
        self.num_features = num_features
        self.conv1 = DeepMax(in_channels, out_channels, kernel_size, stride, padding, choise)  # 1_32
        self.conv2 = DeepMax(out_channels, out_channels * 2, kernel_size, stride, padding, choise)  # 32_64
        self.bn = nn.BatchNorm1d(out_channels * 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flt = nn.Flatten()
        self.classifier = nn.Linear(out_channels * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LNet model."""
        x = x.reshape(x.size(0), 1, self.num_features)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.flt(x)
        x = self.classifier(x)
        return x