"""
Skin Cancer Classification CNN Models

This module implements various CNN architectures for binary classification of skin cancer images.
Models include baseline CNN, normalization variants, ResNet with skip connections, and VGG-16 transfer learning.
Based on Assignment 5 - Skin Cancer Classification using Deep Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights


class BaselineCNN(nn.Module):
    """
    Baseline CNN model with 2 convolutional blocks and 2 fully connected layers.
    Architecture: Conv2d(32) -> ReLU -> MaxPool -> Conv2d(64) -> ReLU -> MaxPool -> FC(128) -> FC(1) -> Sigmoid
    """
    
    def __init__(self, n_channels=3):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BatchNormCNN(nn.Module):
    """
    CNN with Batch Normalization after each convolutional layer.
    Normalizes across the batch dimension to reduce internal covariate shift.
    """
    
    def __init__(self, n_channels=3):
        super(BatchNormCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class GroupNormCNN(nn.Module):
    """
    CNN with Group Normalization after each convolutional layer.
    Groups channels and normalizes within groups, independent of batch size.
    """
    
    def __init__(self, n_channels=3, num_groups=8):
        super(GroupNormCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LayerNormCNN(nn.Module):
    """
    CNN with Layer Normalization after each convolutional layer.
    Normalizes across spatial and channel dimensions for each sample independently.
    """
    
    def __init__(self, n_channels=3):
        super(LayerNormCNN, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([32, 128, 128])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([64, 64, 64])
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        
        x = self.conv2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        
        x = self.classifier(x)
        return x


class ResBlock(nn.Module):
    """
    Residual Block with skip connections.
    Enables training of deeper networks by allowing gradients to flow through skip connections.
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection for dimension matching
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = F.relu(out)
        
        return out


class ResNetCNN(nn.Module):
    """
    ResNet-style CNN with residual connections.
    Shallow ResNet implementation with 2 residual blocks.
    """
    
    def __init__(self, n_channels=3):
        super(ResNetCNN, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.res_block1 = ResBlock(16, 32, stride=2)
        self.res_block2 = ResBlock(32, 64, stride=2)
        
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class VGGMLPHead(nn.Module):
    """
    Custom MLP head for VGG-16 transfer learning.
    4-layer fully connected network for binary classification.
    """
    
    def __init__(self):
        super(VGGMLPHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG16CNN(nn.Module):
    """
    VGG-16 based transfer learning model.
    Uses pretrained VGG-16 feature extractor with custom MLP head.
    """
    
    def __init__(self, pretrained=True):
        super(VGG16CNN, self).__init__()
        self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Replace classifier with custom MLP head
        self.vgg.classifier = VGGMLPHead()
        
        # Adaptive pooling to ensure correct input size for MLP
        self.vgg.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    
    def forward(self, x):
        return self.vgg(x)


def get_model(model_name, **kwargs):
    """
    Factory function to create CNN models.
    
    Args:
        model_name (str): Name of the model architecture
        **kwargs: Additional arguments for model initialization
    
    Returns:
        torch.nn.Module: Instantiated model
    """
    models_dict = {
        'baseline': BaselineCNN,
        'batch_norm': BatchNormCNN,
        'group_norm': GroupNormCNN,
        'layer_norm': LayerNormCNN,
        'resnet': ResNetCNN,
        'vgg16': VGG16CNN,
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models_dict.keys())}")
    
    return models_dict[model_name](**kwargs)


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size=(3, 128, 128)):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    """
    try:
        from torchinfo import summary
        summary(model, input_size=(1,) + input_size)
    except ImportError:
        print(f"Model: {model.__class__.__name__}")
        print(f"Total parameters: {count_parameters(model):,}")
