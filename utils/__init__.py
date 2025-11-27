"""
Utils Package for Malware Detection Project

This package provides utility modules for:
- binary_to_image: Convert PE/ELF files to grayscale images
- data_loader: PyTorch Dataset and DataLoader utilities
- metrics: Evaluation metrics and reporting
- visualization: Grad-CAM, plots, and visual analysis
"""

# Note: Imports are commented out to avoid circular imports and missing dependencies
# Each module should be imported directly when needed:
#   from utils.binary_to_image import bytes_to_image, calculate_image_dimensions
#   from utils.data_loader import MalwareDataset
#   from utils.metrics import MetricsCalculator
#   from utils.visualization import GradCAM

__all__ = [
    'binary_to_image',
    'data_loader', 
    'metrics',
    'visualization',
]
