"""
Models Package for Malware Detection Project

This package provides neural network architectures for malware classification:
- resnet: Transfer learning with pretrained ResNet variants
- custom_cnn: Custom CNN designed for malware visualization images
"""

# Note: Imports are commented out to avoid circular imports
# Each model should be imported directly when needed:
#   from models.resnet import MalwareResNet
#   from models.custom_cnn import MalwareCNN

__all__ = [
    'resnet',
    'custom_cnn',
]
