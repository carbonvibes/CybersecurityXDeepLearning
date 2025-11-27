"""
Visualization Utilities for Malware Classification

This module provides visualization tools for:
1. Grad-CAM - Gradient-weighted Class Activation Mapping
2. Feature map visualization
3. t-SNE/UMAP embeddings
4. Sample visualizations

Explainability is crucial for security applications:
- Analysts need to understand WHY a sample is classified
- Validates that model focuses on meaningful regions
- Builds trust in automated detection

Author: ML/Security Research Team
Date: November 26, 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Generates visual explanations for CNN predictions by:
    1. Computing gradients of target class score w.r.t. feature maps
    2. Global average pooling gradients to get importance weights
    3. Weighted combination of feature maps
    4. ReLU to keep positive influences
    
    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
    """
    
    def __init__(self,
                 model: nn.Module,
                 target_layer: Optional[nn.Module] = None,
                 target_layer_name: Optional[str] = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The neural network model
            target_layer: The convolutional layer to compute Grad-CAM for
            target_layer_name: Name of target layer (alternative to target_layer)
        """
        self.model = model
        self.model.eval()
        
        # Find target layer
        if target_layer is not None:
            self.target_layer = target_layer
        elif target_layer_name is not None:
            self.target_layer = self._find_layer_by_name(target_layer_name)
        else:
            self.target_layer = self._find_last_conv_layer()
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _find_layer_by_name(self, name: str) -> nn.Module:
        """Find layer by name."""
        for n, module in self.model.named_modules():
            if n == name:
                return module
        raise ValueError(f"Layer not found: {name}")
    
    def _find_last_conv_layer(self) -> nn.Module:
        """Find the last convolutional layer in the model."""
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        if last_conv is None:
            raise ValueError("No convolutional layer found in model")
        
        return last_conv
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self,
                 input_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None = use predicted class)
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        # Ensure input is on same device as model
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of feature maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def generate_batch(self,
                       input_batch: torch.Tensor,
                       target_classes: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Generate Grad-CAM for a batch of images.
        
        Args:
            input_batch: Batch of images (N, C, H, W)
            target_classes: List of target classes (one per image)
            
        Returns:
            List of heatmaps
        """
        heatmaps = []
        
        if target_classes is None:
            target_classes = [None] * input_batch.shape[0]
        
        for i in range(input_batch.shape[0]):
            heatmap = self.generate(
                input_batch[i:i+1],
                target_classes[i]
            )
            heatmaps.append(heatmap)
        
        return heatmaps


def overlay_heatmap(image: np.ndarray,
                   heatmap: np.ndarray,
                   alpha: float = 0.5,
                   colormap: str = 'jet') -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image (H, W) or (H, W, C)
        heatmap: Grad-CAM heatmap (H, W)
        alpha: Opacity of heatmap overlay
        colormap: Matplotlib colormap name
        
    Returns:
        Overlay image as RGB numpy array
    """
    # Resize heatmap to match image size
    if heatmap.shape != image.shape[:2]:
        heatmap = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]), Image.Resampling.BILINEAR
        )) / 255.0
    
    # Apply colormap to heatmap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # RGB, drop alpha
    
    # Normalize image to 0-1
    if image.max() > 1:
        image = image / 255.0
    
    # If grayscale, convert to RGB
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Blend
    overlay = (1 - alpha) * image + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    return (overlay * 255).astype(np.uint8)


def visualize_gradcam(model: nn.Module,
                     image: Union[torch.Tensor, np.ndarray],
                     class_names: List[str],
                     target_class: Optional[int] = None,
                     figsize: Tuple[int, int] = (12, 4),
                     save_path: Optional[str] = None) -> None:
    """
    Visualize Grad-CAM for a single image.
    
    Args:
        model: Neural network model
        image: Input image (1, C, H, W) tensor or (H, W) numpy array
        class_names: List of class names
        target_class: Target class for Grad-CAM (None = predicted)
        figsize: Figure size
        save_path: Path to save the figure
    """
    # Prepare image
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        else:
            image = torch.from_numpy(image).unsqueeze(0).float()
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Get prediction
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        output = model(image.to(device))
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()
    
    # Generate Grad-CAM
    gradcam = GradCAM(model)
    heatmap = gradcam.generate(image, target_class)
    
    # Get original image
    orig_image = image.squeeze().cpu().numpy()
    
    # Create overlay
    overlay = overlay_heatmap(orig_image, heatmap, alpha=0.5)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(orig_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f'Prediction: {class_names[pred_class]}\nConfidence: {pred_prob:.2%}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to: {save_path}")
    
    plt.show()


def visualize_feature_maps(model: nn.Module,
                          image: torch.Tensor,
                          layer_name: str,
                          num_features: int = 16,
                          figsize: Tuple[int, int] = (12, 8),
                          save_path: Optional[str] = None) -> None:
    """
    Visualize feature maps from a specific layer.
    
    Args:
        model: Neural network model
        image: Input image tensor
        layer_name: Name of layer to visualize
        num_features: Number of feature maps to show
        figsize: Figure size
        save_path: Path to save figure
    """
    # Hook to capture activations
    activations = {}
    
    def hook(module, input, output):
        activations['features'] = output.detach()
    
    # Find and hook the layer
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook)
            break
    else:
        raise ValueError(f"Layer not found: {layer_name}")
    
    # Forward pass
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        model(image.to(device))
    
    handle.remove()
    
    # Get feature maps
    features = activations['features'][0].cpu().numpy()  # (C, H, W)
    
    # Select subset of features
    num_features = min(num_features, features.shape[0])
    selected_indices = np.linspace(0, features.shape[0]-1, num_features, dtype=int)
    
    # Plot
    grid_size = int(np.ceil(np.sqrt(num_features)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    for idx, feat_idx in enumerate(selected_indices):
        feat_map = features[feat_idx]
        feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
        
        axes[idx].imshow(feat_map, cmap='viridis')
        axes[idx].set_title(f'Feature {feat_idx}', fontsize=8)
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(num_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Feature Maps from {layer_name}', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_embeddings(features: np.ndarray,
                        labels: np.ndarray,
                        class_names: List[str],
                        method: str = 'tsne',
                        figsize: Tuple[int, int] = (12, 10),
                        save_path: Optional[str] = None) -> None:
    """
    Visualize feature embeddings using t-SNE or UMAP.
    
    Args:
        features: Feature vectors (N, D)
        labels: Class labels (N,)
        class_names: List of class names
        method: 'tsne' or 'umap'
        figsize: Figure size
        save_path: Path to save figure
    """
    from sklearn.manifold import TSNE
    
    print(f"Computing {method.upper()} embedding...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        except ImportError:
            print("UMAP not installed, falling back to t-SNE")
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embeddings = reducer.fit_transform(features)
    
    # Plot
    plt.figure(figsize=figsize)
    
    num_classes = len(class_names)
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    
    for i, class_name in enumerate(class_names):
        mask = labels == i
        if mask.sum() > 0:
            plt.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[colors[i]],
                label=class_name,
                alpha=0.6,
                s=20
            )
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'{method.upper()} Visualization of Malware Embeddings')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Embedding visualization saved to: {save_path}")
    
    plt.show()


def visualize_sample_grid(images: List[np.ndarray],
                         labels: List[int],
                         class_names: List[str],
                         predictions: Optional[List[int]] = None,
                         probabilities: Optional[List[float]] = None,
                         figsize: Tuple[int, int] = (15, 12),
                         save_path: Optional[str] = None) -> None:
    """
    Visualize a grid of sample images with labels.
    
    Args:
        images: List of images
        labels: True labels
        class_names: Class names
        predictions: Optional predicted labels
        probabilities: Optional prediction probabilities
        figsize: Figure size
        save_path: Path to save figure
    """
    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    for idx in range(num_images):
        ax = axes[idx]
        ax.imshow(images[idx], cmap='gray')
        
        title = f"True: {class_names[labels[idx]]}"
        if predictions is not None:
            pred = predictions[idx]
            correct = pred == labels[idx]
            title += f"\nPred: {class_names[pred]}"
            if probabilities is not None:
                title += f" ({probabilities[idx]:.1%})"
            
            # Color based on correctness
            color = 'green' if correct else 'red'
            ax.set_title(title, fontsize=8, color=color)
        else:
            ax.set_title(title, fontsize=8)
        
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample grid saved to: {save_path}")
    
    plt.show()


def create_malware_family_comparison(dataset,
                                    class_names: List[str],
                                    samples_per_family: int = 3,
                                    figsize: Tuple[int, int] = (15, 20),
                                    save_path: Optional[str] = None) -> None:
    """
    Create a comparison visualization of samples from each malware family.
    
    Args:
        dataset: PyTorch dataset
        class_names: List of class names
        samples_per_family: Number of samples to show per family
        figsize: Figure size
        save_path: Path to save figure
    """
    num_classes = len(class_names)
    
    fig, axes = plt.subplots(num_classes, samples_per_family, figsize=figsize)
    
    # Collect samples by class
    samples_by_class = {i: [] for i in range(num_classes)}
    
    for img, label in dataset:
        if len(samples_by_class[label]) < samples_per_family:
            samples_by_class[label].append(img)
    
    # Plot
    for class_idx in range(num_classes):
        for sample_idx in range(samples_per_family):
            ax = axes[class_idx, sample_idx]
            
            if sample_idx < len(samples_by_class[class_idx]):
                img = samples_by_class[class_idx][sample_idx]
                if isinstance(img, torch.Tensor):
                    img = img.squeeze().numpy()
                ax.imshow(img, cmap='gray')
            
            ax.axis('off')
            
            if sample_idx == 0:
                ax.set_ylabel(class_names[class_idx], fontsize=8, rotation=0,
                            labelpad=60, va='center')
    
    plt.suptitle('Malware Family Visual Comparison', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Family comparison saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization Utilities for Malware Classification")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - GradCAM: Generate Grad-CAM heatmaps")
    print("  - overlay_heatmap: Overlay heatmap on image")
    print("  - visualize_gradcam: Full Grad-CAM visualization")
    print("  - visualize_feature_maps: Show CNN feature maps")
    print("  - visualize_embeddings: t-SNE/UMAP plots")
    print("  - visualize_sample_grid: Sample image grid")
    print("  - create_malware_family_comparison: Compare families")
