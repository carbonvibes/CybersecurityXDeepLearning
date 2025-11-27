"""
ResNet-50 Model for Malware Classification

This module implements a modified ResNet-50 architecture for malware image
classification using transfer learning from ImageNet pre-trained weights.

Key Modifications:
1. Input layer: Changed from 3 channels (RGB) to 1 channel (grayscale)
2. Output layer: Changed from 1000 classes (ImageNet) to 25 (malware families)
3. Custom classification head with dropout for regularization

Transfer Learning Strategy:
- Phase 1: Freeze backbone, train only classifier head
- Phase 2: Unfreeze deeper layers, fine-tune with lower learning rate
- Phase 3: Full network fine-tuning with very low learning rate

Design Decisions:
1. ResNet-50 chosen over ResNet-101/152 for faster training with similar accuracy
2. Average pooling used to reduce spatial dimensions before FC layers
3. Dropout (0.5) added to prevent overfitting on small dataset
4. Option to use different ResNet variants (18, 34, 50, 101, 152)

Author: ML/Security Research Team
Date: November 26, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, List


class MalwareResNet(nn.Module):
    """
    ResNet-based model for malware classification.
    
    Supports multiple ResNet variants with customizable input channels
    and number of output classes.
    
    Attributes:
        backbone: ResNet feature extractor
        classifier: Custom classification head
        num_classes: Number of output classes
    """
    
    def __init__(self,
                 num_classes: int = 25,
                 resnet_variant: str = 'resnet50',
                 pretrained: bool = True,
                 input_channels: int = 1,
                 dropout_rate: float = 0.5,
                 freeze_backbone: bool = False):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of malware families to classify
            resnet_variant: Which ResNet to use ('resnet18', 'resnet34', 
                           'resnet50', 'resnet101', 'resnet152')
            pretrained: Whether to use ImageNet pre-trained weights
            input_channels: Number of input channels (1 for grayscale)
            dropout_rate: Dropout rate in classifier head
            freeze_backbone: If True, freeze all backbone layers initially
        """
        super(MalwareResNet, self).__init__()
        
        self.num_classes = num_classes
        self.resnet_variant = resnet_variant
        
        # Load pre-trained ResNet
        self.backbone = self._create_backbone(resnet_variant, pretrained)
        
        # Modify first conv layer for grayscale input
        self._modify_input_layer(input_channels)
        
        # Get the feature dimension from backbone
        self.feature_dim = self._get_feature_dim(resnet_variant)
        
        # Replace the classifier head
        self.backbone.fc = nn.Identity()  # Remove original FC layer
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.6),  # Slightly less dropout
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
        
        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone()
    
    def _create_backbone(self, variant: str, pretrained: bool) -> nn.Module:
        """Create the ResNet backbone."""
        weights = 'IMAGENET1K_V1' if pretrained else None
        
        if variant == 'resnet18':
            return models.resnet18(weights=weights)
        elif variant == 'resnet34':
            return models.resnet34(weights=weights)
        elif variant == 'resnet50':
            return models.resnet50(weights=weights)
        elif variant == 'resnet101':
            return models.resnet101(weights=weights)
        elif variant == 'resnet152':
            return models.resnet152(weights=weights)
        else:
            raise ValueError(f"Unknown ResNet variant: {variant}")
    
    def _get_feature_dim(self, variant: str) -> int:
        """Get feature dimension for the given ResNet variant."""
        if variant in ['resnet18', 'resnet34']:
            return 512
        else:  # resnet50, resnet101, resnet152
            return 2048
    
    def _modify_input_layer(self, input_channels: int) -> None:
        """
        Modify the first convolutional layer to accept different input channels.
        
        Strategy: Average the pre-trained weights across RGB channels,
        then replicate for the new number of input channels.
        """
        original_conv = self.backbone.conv1
        
        if input_channels == 3:
            return  # No modification needed
        
        # Create new conv layer with same parameters but different input channels
        new_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize weights from pre-trained model
        with torch.no_grad():
            # Average across RGB channels and replicate
            rgb_weights = original_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight = nn.Parameter(rgb_weights.repeat(1, input_channels, 1, 1))
            
            if original_conv.bias is not None:
                new_conv.bias = nn.Parameter(original_conv.bias.clone())
        
        self.backbone.conv1 = new_conv
    
    def _init_classifier_weights(self) -> None:
        """Initialize classifier weights using Xavier initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings without classification.
        
        Useful for:
        - Visualization (t-SNE, UMAP)
        - Similarity analysis
        - Ensemble methods
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        return self.backbone(x)
    
    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen - only classifier head will be trained")
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None) -> None:
        """
        Unfreeze backbone parameters.
        
        Args:
            num_layers: If specified, only unfreeze the last N layers.
                       If None, unfreeze all layers.
        """
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Full backbone unfrozen")
        else:
            # ResNet layer structure: conv1, bn1, layer1, layer2, layer3, layer4
            layers = ['layer4', 'layer3', 'layer2', 'layer1', 'bn1', 'conv1']
            layers_to_unfreeze = layers[:min(num_layers, len(layers))]
            
            for name, param in self.backbone.named_parameters():
                if any(layer in name for layer in layers_to_unfreeze):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            print(f"Unfrozen layers: {layers_to_unfreeze}")
    
    def get_trainable_params(self) -> List[dict]:
        """
        Get parameter groups for differential learning rates.
        
        Returns:
            List of parameter groups suitable for optimizer
        """
        backbone_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'classifier' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
        
        return [
            {'params': backbone_params, 'lr_scale': 0.1},  # Lower LR for backbone
            {'params': classifier_params, 'lr_scale': 1.0}  # Full LR for classifier
        ]
    
    def get_model_size(self) -> dict:
        """
        Get model size statistics.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params,
            'size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


def create_resnet_model(num_classes: int = 25,
                        variant: str = 'resnet50',
                        pretrained: bool = True,
                        freeze_backbone: bool = True) -> MalwareResNet:
    """
    Factory function to create a ResNet model for malware classification.
    
    Args:
        num_classes: Number of malware families
        variant: ResNet variant to use
        pretrained: Use ImageNet pre-trained weights
        freeze_backbone: Initially freeze backbone layers
        
    Returns:
        Configured MalwareResNet model
    """
    model = MalwareResNet(
        num_classes=num_classes,
        resnet_variant=variant,
        pretrained=pretrained,
        input_channels=1,  # Grayscale
        dropout_rate=0.5,
        freeze_backbone=freeze_backbone
    )
    
    return model


class EfficientNetMalware(nn.Module):
    """
    EfficientNet-based model for malware classification.
    
    EfficientNet offers better accuracy-to-parameter ratio than ResNet,
    making it ideal for production deployment.
    """
    
    def __init__(self,
                 num_classes: int = 25,
                 variant: str = 'efficientnet_b0',
                 pretrained: bool = True,
                 input_channels: int = 1,
                 dropout_rate: float = 0.3):
        """
        Initialize EfficientNet model.
        
        Args:
            num_classes: Number of output classes
            variant: EfficientNet variant ('efficientnet_b0' to 'efficientnet_b7')
            pretrained: Use ImageNet pre-trained weights
            input_channels: Number of input channels
            dropout_rate: Dropout rate
        """
        super(EfficientNetMalware, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pre-trained EfficientNet
        weights = 'IMAGENET1K_V1' if pretrained else None
        
        if variant == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=weights)
            feature_dim = 1280
        elif variant == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(weights=weights)
            feature_dim = 1280
        elif variant == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(weights=weights)
            feature_dim = 1408
        elif variant == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(weights=weights)
            feature_dim = 1536
        else:
            raise ValueError(f"Unknown EfficientNet variant: {variant}")
        
        # Modify first conv for grayscale
        if input_channels != 3:
            original_conv = self.backbone.features[0][0]
            new_conv = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            with torch.no_grad():
                rgb_weights = original_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight = nn.Parameter(rgb_weights)
            
            self.backbone.features[0][0] = new_conv
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


if __name__ == "__main__":
    # Test model creation
    print("Testing ResNet Model for Malware Classification")
    print("=" * 50)
    
    # Create model
    model = create_resnet_model(num_classes=25, variant='resnet50', pretrained=True)
    
    # Print model size
    size_info = model.get_model_size()
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {size_info['total_params']:,}")
    print(f"  Trainable parameters: {size_info['trainable_params']:,}")
    print(f"  Frozen parameters: {size_info['frozen_params']:,}")
    print(f"  Model size: {size_info['size_mb']:.2f} MB")
    
    # Test forward pass
    dummy_input = torch.randn(4, 1, 224, 224)  # Batch of 4 grayscale images
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test feature extraction
    features = model.extract_features(dummy_input)
    print(f"Feature shape: {features.shape}")
    
    # Test unfreezing
    print("\nTesting layer unfreezing...")
    model.unfreeze_backbone(num_layers=2)
    
    size_info = model.get_model_size()
    print(f"Trainable parameters after unfreezing: {size_info['trainable_params']:,}")
