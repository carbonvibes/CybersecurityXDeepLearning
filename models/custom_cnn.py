"""
Custom CNN Model with Residual Blocks for Malware Classification

This module implements a custom CNN architecture designed specifically for
malware visualization classification. Unlike transfer learning models,
this network is trained from scratch.

Architecture Design:
1. 5 Residual Blocks - Inspired by ResNet but lighter weight
2. Batch Normalization - Stabilizes training, enables higher learning rates
3. Dropout - Prevents overfitting on the relatively small Malimg dataset
4. Global Average Pooling - Reduces parameters, provides spatial invariance

Design Rationale:
- Smaller than ResNet-50 (5-10M vs 25M parameters)
- Faster training from scratch
- More interpretable - we control every layer
- Can be optimized specifically for malware patterns

Block Structure:
    Input
      |
    Conv3x3 → BN → ReLU
      |
    Conv3x3 → BN
      |
   (+) ← Identity/Projection shortcut
      |
    ReLU
      |
    Output

Author: ML/Security Research Team
Date: November 26, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class ResidualBlock(nn.Module):
    """
    Basic Residual Block with two 3x3 convolutions.
    
    Features:
    - Skip connection for gradient flow
    - Batch normalization for training stability
    - Optional downsampling (stride=2)
    - Projection shortcut when dimensions change
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        """
        Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first convolution (2 for downsampling)
            downsample: Module to match dimensions in skip connection
        """
        super(ResidualBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.downsample = downsample
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual
        out += identity
        out = self.relu(out)
        
        return out


class AttentionBlock(nn.Module):
    """
    Channel Attention Block (Squeeze-and-Excitation style).
    
    Learns to weight channel importance dynamically,
    helping the model focus on relevant features for malware detection.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize attention block.
        
        Args:
            channels: Number of input/output channels
            reduction: Channel reduction ratio for bottleneck
        """
        super(AttentionBlock, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention."""
        b, c, _, _ = x.size()
        
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        
        # Excite
        y = self.fc(y).view(b, c, 1, 1)
        
        # Scale
        return x * y.expand_as(x)


class CustomMalwareCNN(nn.Module):
    """
    Custom CNN for Malware Classification.
    
    Architecture Overview:
    - Stem: Initial convolution to extract low-level features
    - Stage 1-4: Residual blocks with increasing channels
    - Attention: Channel attention for feature refinement
    - Head: Global pooling + FC for classification
    
    Layer Details:
    | Stage   | Output Size | Channels | Blocks |
    |---------|-------------|----------|--------|
    | Stem    | 112x112     | 64       | -      |
    | Stage 1 | 56x56       | 64       | 2      |
    | Stage 2 | 28x28       | 128      | 2      |
    | Stage 3 | 14x14       | 256      | 2      |
    | Stage 4 | 7x7         | 512      | 2      |
    | Pool    | 1x1         | 512      | -      |
    | FC      | -           | 25       | -      |
    """
    
    def __init__(self,
                 num_classes: int = 25,
                 input_channels: int = 1,
                 base_channels: int = 64,
                 blocks_per_stage: List[int] = [2, 2, 2, 2],
                 dropout_rate: float = 0.5,
                 use_attention: bool = True):
        """
        Initialize the custom CNN.
        
        Args:
            num_classes: Number of malware families
            input_channels: Input channels (1 for grayscale)
            base_channels: Base channel count (doubled each stage)
            blocks_per_stage: Number of residual blocks in each stage
            dropout_rate: Dropout rate before final FC
            use_attention: Whether to use attention blocks
        """
        super(CustomMalwareCNN, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Channel progression: 64 → 128 → 256 → 512
        channels = [base_channels * (2 ** i) for i in range(4)]
        
        # ==================== STEM ====================
        # Initial convolution to extract low-level features
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=7, 
                     stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # ==================== STAGES ====================
        # Stage 1: 64 channels
        self.stage1 = self._make_stage(
            channels[0], channels[0], blocks_per_stage[0], stride=1
        )
        
        # Stage 2: 128 channels
        self.stage2 = self._make_stage(
            channels[0], channels[1], blocks_per_stage[1], stride=2
        )
        
        # Stage 3: 256 channels
        self.stage3 = self._make_stage(
            channels[1], channels[2], blocks_per_stage[2], stride=2
        )
        
        # Stage 4: 512 channels
        self.stage4 = self._make_stage(
            channels[2], channels[3], blocks_per_stage[3], stride=2
        )
        
        # ==================== ATTENTION ====================
        if use_attention:
            self.attention = AttentionBlock(channels[3], reduction=16)
        
        # ==================== HEAD ====================
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(channels[3], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.6),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _make_stage(self,
                    in_channels: int,
                    out_channels: int,
                    num_blocks: int,
                    stride: int) -> nn.Sequential:
        """
        Create a stage with multiple residual blocks.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_blocks: Number of residual blocks
            stride: Stride for first block (for downsampling)
            
        Returns:
            Sequential container of residual blocks
        """
        # Downsample if channels change or stride > 1
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        
        # First block (may downsample)
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                       nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Stem
        x = self.stem(x)
        
        # Stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Attention
        if self.use_attention:
            x = self.attention(x)
        
        # Head
        x = self.avgpool(x)
        x = self.classifier(x)
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings before classification.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor of shape (batch, 512)
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def get_intermediate_features(self, x: torch.Tensor) -> dict:
        """
        Get features from all stages for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of feature maps from each stage
        """
        features = {}
        
        x = self.stem(x)
        features['stem'] = x.clone()
        
        x = self.stage1(x)
        features['stage1'] = x.clone()
        
        x = self.stage2(x)
        features['stage2'] = x.clone()
        
        x = self.stage3(x)
        features['stage3'] = x.clone()
        
        x = self.stage4(x)
        features['stage4'] = x.clone()
        
        if self.use_attention:
            x = self.attention(x)
            features['attention'] = x.clone()
        
        return features
    
    def get_model_size(self) -> dict:
        """Get model size statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_mb': total_params * 4 / (1024 * 1024)
        }


class LightMalwareCNN(nn.Module):
    """
    Lightweight CNN for fast inference.
    
    Designed for edge deployment with:
    - Fewer parameters (~1M)
    - Depthwise separable convolutions
    - Fast inference (<50ms on mobile CPU)
    """
    
    def __init__(self,
                 num_classes: int = 25,
                 input_channels: int = 1,
                 width_multiplier: float = 1.0):
        """
        Initialize lightweight CNN.
        
        Args:
            num_classes: Number of output classes
            input_channels: Input channels
            width_multiplier: Width scaling factor (0.5 for smaller, 2.0 for larger)
        """
        super(LightMalwareCNN, self).__init__()
        
        def _make_channels(c):
            return max(8, int(c * width_multiplier))
        
        # Simple VGG-style architecture with depthwise separable convs
        self.features = nn.Sequential(
            # Block 1: 224 → 112
            nn.Conv2d(input_channels, _make_channels(32), 3, stride=2, padding=1),
            nn.BatchNorm2d(_make_channels(32)),
            nn.ReLU(inplace=True),
            
            # Block 2: 112 → 56
            self._depthwise_separable(_make_channels(32), _make_channels(64), stride=2),
            
            # Block 3: 56 → 28
            self._depthwise_separable(_make_channels(64), _make_channels(128), stride=2),
            self._depthwise_separable(_make_channels(128), _make_channels(128), stride=1),
            
            # Block 4: 28 → 14
            self._depthwise_separable(_make_channels(128), _make_channels(256), stride=2),
            self._depthwise_separable(_make_channels(256), _make_channels(256), stride=1),
            
            # Block 5: 14 → 7
            self._depthwise_separable(_make_channels(256), _make_channels(512), stride=2),
            
            # Global pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(_make_channels(512), num_classes)
        )
    
    def _depthwise_separable(self, in_c, out_c, stride):
        """Create a depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_custom_cnn(num_classes: int = 25,
                      model_size: str = 'medium',
                      use_attention: bool = True) -> nn.Module:
    """
    Factory function to create a custom CNN model.
    
    Args:
        num_classes: Number of malware families
        model_size: 'small', 'medium', or 'large'
        use_attention: Whether to use attention blocks
        
    Returns:
        Custom CNN model
    """
    if model_size == 'small':
        return LightMalwareCNN(num_classes=num_classes)
    
    elif model_size == 'medium':
        return CustomMalwareCNN(
            num_classes=num_classes,
            base_channels=64,
            blocks_per_stage=[2, 2, 2, 2],
            use_attention=use_attention
        )
    
    elif model_size == 'large':
        return CustomMalwareCNN(
            num_classes=num_classes,
            base_channels=64,
            blocks_per_stage=[3, 4, 6, 3],  # Similar to ResNet-50
            use_attention=use_attention
        )
    
    else:
        raise ValueError(f"Unknown model size: {model_size}")


if __name__ == "__main__":
    # Test model creation
    print("Testing Custom CNN for Malware Classification")
    print("=" * 50)
    
    # Test different model sizes
    for size in ['small', 'medium', 'large']:
        print(f"\n{size.upper()} Model:")
        model = create_custom_cnn(num_classes=25, model_size=size)
        
        # Get size info
        if hasattr(model, 'get_model_size'):
            info = model.get_model_size()
        else:
            total = sum(p.numel() for p in model.parameters())
            info = {'total_params': total, 'size_mb': total * 4 / (1024 * 1024)}
        
        print(f"  Parameters: {info['total_params']:,}")
        print(f"  Size: {info['size_mb']:.2f} MB")
        
        # Test forward pass
        x = torch.randn(4, 1, 224, 224)
        y = model(x)
        print(f"  Input: {x.shape} → Output: {y.shape}")
    
    # Test feature extraction
    print("\n\nTesting feature extraction:")
    model = create_custom_cnn(num_classes=25, model_size='medium')
    x = torch.randn(2, 1, 224, 224)
    
    features = model.extract_features(x)
    print(f"Feature shape: {features.shape}")
    
    # Test intermediate features
    intermediate = model.get_intermediate_features(x)
    print("\nIntermediate feature shapes:")
    for name, feat in intermediate.items():
        print(f"  {name}: {feat.shape}")
