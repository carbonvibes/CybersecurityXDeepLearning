"""
PyTorch Data Loading Utilities for Malware Detection

This module provides:
1. MalwareDataset - Custom PyTorch Dataset for malware images
2. Data augmentation transforms specific to malware visualization
3. Data loader factory functions
4. Class balancing utilities

Design Decisions:
1. Grayscale input (1 channel) - Binary data is naturally single-channel
2. Conservative augmentation - Heavy transforms may corrupt malware semantics
3. Weighted sampling - Address class imbalance in Malimg dataset
4. Stratified splits - Ensure all classes represented in train/val/test

Author: ML/Security Research Team
Date: November 26, 2025
"""

import os
import csv
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Callable, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as transforms


class MalwareDataset(Dataset):
    """
    PyTorch Dataset for malware visualization images.
    
    Supports:
    - Loading from CSV split files (train.csv, val.csv, test.csv)
    - Loading directly from directory structure
    - Custom transforms for data augmentation
    - Return of image, label, and optionally file path
    
    Attributes:
        samples: List of (image_path, label_idx) tuples
        classes: List of class names (malware families)
        class_to_idx: Dictionary mapping class name to index
        transform: Optional transforms to apply to images
    """
    
    def __init__(self,
                 data_dir: Optional[Union[str, Path]] = None,
                 csv_file: Optional[Union[str, Path]] = None,
                 transform: Optional[Callable] = None,
                 return_path: bool = False):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing class subdirectories
                     (mutually exclusive with csv_file)
            csv_file: Path to CSV file with columns [image_path, family_name, label_idx]
                     (mutually exclusive with data_dir)
            transform: Transforms to apply to images
            return_path: If True, __getitem__ returns (image, label, path)
        """
        self.transform = transform
        self.return_path = return_path
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        if csv_file is not None:
            self._load_from_csv(csv_file)
        elif data_dir is not None:
            self._load_from_directory(data_dir)
        else:
            raise ValueError("Either data_dir or csv_file must be provided")
    
    def _load_from_csv(self, csv_file: Union[str, Path]) -> None:
        """Load dataset from a CSV split file."""
        csv_file = Path(csv_file)
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = row['image_path']
                family_name = row['family_name']
                label_idx = int(row['label_idx'])
                
                self.samples.append((image_path, label_idx))
                
                if family_name not in self.class_to_idx:
                    self.class_to_idx[family_name] = label_idx
                    self.classes.append(family_name)
        
        # Sort classes by index
        self.classes = [c for c, _ in sorted(self.class_to_idx.items(), 
                                              key=lambda x: x[1])]
    
    def _load_from_directory(self, data_dir: Union[str, Path]) -> None:
        """Load dataset from directory structure."""
        data_dir = Path(data_dir)
        
        # Get class directories
        class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx
            
            # Get images in this class
            for img_path in class_dir.glob('*.png'):
                self.samples.append((str(img_path), idx))
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((str(img_path), idx))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            If return_path=False: (image_tensor, label)
            If return_path=True: (image_tensor, label, image_path)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')  # Ensure grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.return_path:
            return image, label, img_path
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced data.
        
        Uses inverse frequency weighting:
        weight[c] = total_samples / (num_classes * count[c])
        
        Returns:
            Tensor of class weights
        """
        # Count samples per class
        class_counts = np.zeros(len(self.classes))
        for _, label in self.samples:
            class_counts[label] += 1
        
        # Calculate weights (inverse frequency)
        total = len(self.samples)
        num_classes = len(self.classes)
        weights = total / (num_classes * class_counts)
        
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Get weight for each sample (for WeightedRandomSampler).
        
        Returns:
            Tensor of sample weights
        """
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label] for _, label in self.samples]
        return torch.FloatTensor(sample_weights)


# ============================================================================
# Data Augmentation Transforms
# ============================================================================

def get_train_transforms(image_size: int = 224,
                         augmentation_level: str = 'light') -> transforms.Compose:
    """
    Get transforms for training data.
    
    Augmentation Philosophy for Malware Images:
    - Light augmentation: Minimal changes, preserve binary structure
    - Medium augmentation: More variation, may help generalization
    - Heavy augmentation: Risk of corrupting malware semantics
    
    Args:
        image_size: Target image size (square)
        augmentation_level: 'none', 'light', 'medium', or 'heavy'
        
    Returns:
        Composed transforms
    """
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    
    if augmentation_level == 'none':
        # No augmentation - just resize and convert
        return transforms.Compose(base_transforms)
    
    elif augmentation_level == 'light':
        # Conservative augmentation - recommended for malware
        augmentations = [
            transforms.Resize((image_size + 8, image_size + 8)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
        ]
        
    elif augmentation_level == 'medium':
        # Moderate augmentation
        augmentations = [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ]
        
    elif augmentation_level == 'heavy':
        # Heavy augmentation - use with caution
        augmentations = [
            transforms.Resize((image_size + 24, image_size + 24)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ]
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    
    return transforms.Compose(augmentations)


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get transforms for validation/test data.
    
    No augmentation - just resize and normalize.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


# ============================================================================
# Data Loader Factory Functions
# ============================================================================

def create_data_loaders(data_dir: Union[str, Path],
                       splits_dir: Union[str, Path],
                       batch_size: int = 32,
                       image_size: int = 224,
                       augmentation_level: str = 'light',
                       num_workers: int = 4,
                       use_weighted_sampling: bool = True,
                       pin_memory: bool = True) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Root directory containing the dataset
        splits_dir: Directory containing train.csv, val.csv, test.csv
        batch_size: Batch size for training
        image_size: Target image size
        augmentation_level: Level of data augmentation
        num_workers: Number of worker processes for data loading
        use_weighted_sampling: If True, use weighted sampling for class balance
        pin_memory: If True, pin memory for faster GPU transfer
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    splits_dir = Path(splits_dir)
    
    # Create datasets
    train_dataset = MalwareDataset(
        csv_file=splits_dir / 'train.csv',
        transform=get_train_transforms(image_size, augmentation_level)
    )
    
    val_dataset = MalwareDataset(
        csv_file=splits_dir / 'val.csv',
        transform=get_val_transforms(image_size)
    )
    
    test_dataset = MalwareDataset(
        csv_file=splits_dir / 'test.csv',
        transform=get_val_transforms(image_size)
    )
    
    # Create samplers
    train_sampler = None
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
    
    # Create data loaders
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),  # Only shuffle if no sampler
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # Drop last incomplete batch for training
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }
    
    return loaders


def get_class_names(splits_dir: Union[str, Path]) -> List[str]:
    """
    Get list of class names from label mapping file.
    
    Args:
        splits_dir: Directory containing label_mapping.csv
        
    Returns:
        List of class names ordered by label index
    """
    splits_dir = Path(splits_dir)
    mapping_file = splits_dir / 'label_mapping.csv'
    
    classes = []
    with open(mapping_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes.append(row['family_name'])
    
    return classes


def get_dataset_statistics(data_loader: DataLoader) -> Dict[str, float]:
    """
    Calculate dataset statistics (mean and std) for normalization.
    
    Args:
        data_loader: DataLoader to compute statistics from
        
    Returns:
        Dictionary with 'mean' and 'std' values
    """
    print("Computing dataset statistics...")
    
    mean = 0.0
    std = 0.0
    n_samples = 0
    
    for images, _ in data_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += batch_samples
    
    mean /= n_samples
    std /= n_samples
    
    return {
        'mean': mean.item(),
        'std': std.item()
    }


# ============================================================================
# Utility Functions
# ============================================================================

def visualize_batch(data_loader: DataLoader,
                   class_names: List[str],
                   num_images: int = 16,
                   figsize: Tuple[int, int] = (12, 12)) -> None:
    """
    Visualize a batch of images from the data loader.
    
    Args:
        data_loader: DataLoader to sample from
        class_names: List of class names
        num_images: Number of images to display
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    # Get a batch
    images, labels = next(iter(data_loader))
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    for idx in range(min(num_images, len(images))):
        img = images[idx].squeeze().numpy()
        label = labels[idx].item()
        
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(class_names[label], fontsize=8)
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test data loading
    print("Testing Data Loading Utilities")
    print("=" * 50)
    
    # Example paths
    splits_dir = Path(__file__).parent.parent / "data" / "splits"
    
    if splits_dir.exists():
        # Load class names
        classes = get_class_names(splits_dir)
        print(f"\nNumber of classes: {len(classes)}")
        print(f"Classes: {classes[:5]}...")
        
        # Create a dataset
        train_dataset = MalwareDataset(
            csv_file=splits_dir / 'train.csv',
            transform=get_val_transforms(224)
        )
        
        print(f"\nTraining samples: {len(train_dataset)}")
        
        # Get class weights
        weights = train_dataset.get_class_weights()
        print(f"\nClass weights (first 5): {weights[:5]}")
        
        # Test getting a sample
        img, label = train_dataset[0]
        print(f"\nSample image shape: {img.shape}")
        print(f"Sample label: {label} ({classes[label]})")
    else:
        print(f"\nSplits directory not found: {splits_dir}")
        print("Run data/prepare_dataset.py first to create splits.")
