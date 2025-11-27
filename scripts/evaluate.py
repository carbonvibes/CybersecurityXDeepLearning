"""
Evaluation Script for Malware Classification Models

This script provides comprehensive evaluation of trained models:
1. Test set evaluation with all metrics
2. Per-class performance analysis
3. Confusion matrix visualization
4. ROC curves and AUC
5. Grad-CAM visualizations
6. Error analysis

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/model/checkpoint_best.pth
    python scripts/evaluate.py --checkpoint checkpoints/model/checkpoint_best.pth --gradcam

Author: ML/Security Research Team
Date: November 26, 2025
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data_loader import MalwareDataset, get_val_transforms, get_class_names
from utils.metrics import (
    MetricsCalculator,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_per_class_performance,
    generate_evaluation_report
)
from utils.visualization import (
    GradCAM,
    visualize_gradcam,
    overlay_heatmap,
    visualize_embeddings
)
from models.resnet import create_resnet_model
from models.custom_cnn import create_custom_cnn


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model_name = config['model_name']
    num_classes = config['num_classes']
    
    if 'resnet' in model_name:
        variant = 'resnet50' if '50' in model_name else 'resnet18'
        model = create_resnet_model(
            num_classes=num_classes,
            variant=variant,
            pretrained=False
        )
    elif 'custom' in model_name:
        size = 'small' if 'small' in model_name else 'medium'
        model = create_custom_cnn(
            num_classes=num_classes,
            model_size=size
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    return model, config


def evaluate_model(model: nn.Module,
                  test_loader: DataLoader,
                  class_names: List[str],
                  device: torch.device) -> Tuple[MetricsCalculator, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on test set.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        class_names: List of class names
        device: Device
        
    Returns:
        Tuple of (metrics_calculator, predictions, labels, probabilities)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    print("\n📊 Evaluating model on test set...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if len(batch) == 3:
                images, labels, paths = batch
                all_paths.extend(paths)
            else:
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to arrays
    predictions = np.array(all_preds)
    labels = np.array(all_labels)
    probabilities = np.array(all_probs)
    
    # Create metrics calculator
    calc = MetricsCalculator(class_names)
    calc.update(predictions, labels, probabilities)
    
    return calc, predictions, labels, probabilities


def analyze_errors(predictions: np.ndarray,
                  labels: np.ndarray,
                  probabilities: np.ndarray,
                  class_names: List[str],
                  dataset: MalwareDataset,
                  top_k: int = 10) -> List[Dict]:
    """
    Analyze misclassified samples.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        probabilities: Prediction probabilities
        class_names: Class names
        dataset: Test dataset
        top_k: Number of errors to analyze
        
    Returns:
        List of error dictionaries
    """
    # Find errors
    errors_mask = predictions != labels
    error_indices = np.where(errors_mask)[0]
    
    if len(error_indices) == 0:
        print("No errors to analyze!")
        return []
    
    # Get confidence of errors
    error_confidences = probabilities[error_indices, predictions[error_indices]]
    
    # Sort by confidence (high confidence errors are more concerning)
    sorted_indices = np.argsort(-error_confidences)[:top_k]
    
    errors = []
    for idx in sorted_indices:
        error_idx = error_indices[idx]
        
        error_info = {
            'sample_idx': int(error_idx),
            'true_label': class_names[labels[error_idx]],
            'predicted_label': class_names[predictions[error_idx]],
            'confidence': float(probabilities[error_idx, predictions[error_idx]]),
            'true_prob': float(probabilities[error_idx, labels[error_idx]]),
        }
        
        if hasattr(dataset, 'samples'):
            error_info['image_path'] = dataset.samples[error_idx][0]
        
        errors.append(error_info)
    
    return errors


def generate_gradcam_samples(model: nn.Module,
                            dataset: MalwareDataset,
                            class_names: List[str],
                            device: torch.device,
                            save_dir: Path,
                            samples_per_class: int = 2) -> None:
    """
    Generate Grad-CAM visualizations for sample images.
    
    Args:
        model: Neural network model
        dataset: Test dataset
        class_names: Class names
        device: Device
        save_dir: Directory to save visualizations
        samples_per_class: Number of samples per class
    """
    import matplotlib.pyplot as plt
    
    gradcam_dir = save_dir / 'gradcam'
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🔍 Generating Grad-CAM visualizations...")
    
    # Collect samples by class
    samples_by_class = {i: [] for i in range(len(class_names))}
    
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if len(samples_by_class[label]) < samples_per_class:
            samples_by_class[label].append((img, idx))
    
    # Create Grad-CAM generator
    gradcam = GradCAM(model)
    
    for class_idx, samples in tqdm(samples_by_class.items(), desc="Generating Grad-CAM"):
        class_name = class_names[class_idx]
        
        for sample_idx, (img, dataset_idx) in enumerate(samples):
            # Generate heatmap
            img_tensor = img.unsqueeze(0).to(device)
            heatmap = gradcam.generate(img_tensor, class_idx)
            
            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                pred_class = output.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()
            
            # Create visualization
            orig_img = img.squeeze().numpy()
            overlay = overlay_heatmap(orig_img, heatmap, alpha=0.5)
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(orig_img, cmap='gray')
            axes[0].set_title(f'Original: {class_name}')
            axes[0].axis('off')
            
            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title(f'Pred: {class_names[pred_class]} ({confidence:.1%})')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save
            save_path = gradcam_dir / f'{class_name}_{sample_idx}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"Grad-CAM visualizations saved to: {gradcam_dir}")


def extract_and_visualize_embeddings(model: nn.Module,
                                     test_loader: DataLoader,
                                     class_names: List[str],
                                     device: torch.device,
                                     save_path: Path) -> None:
    """
    Extract features and visualize with t-SNE.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        class_names: Class names
        device: Device
        save_path: Path to save visualization
    """
    print("\n📈 Extracting features for embedding visualization...")
    
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, batch_labels in tqdm(test_loader, desc="Extracting features"):
            images = images.to(device)
            
            # Get features before final classification
            if hasattr(model, 'extract_features'):
                batch_features = model.extract_features(images)
            else:
                # Fallback: get output before softmax
                batch_features = model.backbone(images)
            
            features.extend(batch_features.cpu().numpy())
            labels.extend(batch_labels.numpy())
    
    features = np.array(features)
    labels = np.array(labels)
    
    # Visualize
    visualize_embeddings(
        features, labels, class_names,
        method='tsne',
        save_path=str(save_path)
    )


def main(args):
    """Main evaluation function."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\n📦 Loading model from: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    splits_dir = project_root / 'data' / 'splits'
    
    # Create output directory
    checkpoint_dir = Path(args.checkpoint).parent
    eval_dir = checkpoint_dir / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Get class names
    class_names = get_class_names(splits_dir)
    
    # Create test dataset and loader
    test_dataset = MalwareDataset(
        csv_file=splits_dir / 'test.csv',
        transform=get_val_transforms(config.get('image_size', 224)),
        return_path=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    calc, predictions, labels, probabilities = evaluate_model(
        model, test_loader, class_names, device
    )
    
    # Generate report
    print("\n" + "=" * 60)
    report = generate_evaluation_report(calc, save_dir=str(eval_dir))
    print(report)
    
    # Plot confusion matrix
    cm = calc.get_confusion_matrix()
    plot_confusion_matrix(
        cm, class_names,
        normalize=True,
        save_path=str(eval_dir / 'confusion_matrix_normalized.png')
    )
    
    # Plot ROC curves
    plot_roc_curves(
        labels, probabilities, class_names,
        save_path=str(eval_dir / 'roc_curves.png')
    )
    
    # Per-class performance
    per_class = calc.get_per_class_metrics()
    plot_per_class_performance(
        per_class, metric='f1',
        save_path=str(eval_dir / 'per_class_f1.png')
    )
    
    # Error analysis
    print("\n🔍 Analyzing misclassified samples...")
    errors = analyze_errors(
        predictions, labels, probabilities,
        class_names, test_dataset, top_k=20
    )
    
    print("\nTop high-confidence errors:")
    for i, error in enumerate(errors[:10]):
        print(f"  {i+1}. {error['true_label']} → {error['predicted_label']} "
              f"(conf: {error['confidence']:.1%})")
    
    # Save errors
    with open(eval_dir / 'error_analysis.json', 'w') as f:
        json.dump(errors, f, indent=2)
    
    # Grad-CAM visualizations
    if args.gradcam:
        # Reload dataset without paths for gradcam
        test_dataset_gradcam = MalwareDataset(
            csv_file=splits_dir / 'test.csv',
            transform=get_val_transforms(config.get('image_size', 224)),
            return_path=False
        )
        
        generate_gradcam_samples(
            model, test_dataset_gradcam, class_names,
            device, eval_dir, samples_per_class=2
        )
    
    # t-SNE visualization
    if args.embeddings:
        # Create loader without paths
        test_loader_emb = DataLoader(
            MalwareDataset(
                csv_file=splits_dir / 'test.csv',
                transform=get_val_transforms(config.get('image_size', 224))
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        extract_and_visualize_embeddings(
            model, test_loader_emb, class_names,
            device, eval_dir / 'tsne_embeddings.png'
        )
    
    print(f"\n✅ Evaluation complete! Results saved to: {eval_dir}")
    
    # Summary
    metrics = calc.compute()
    print("\n📊 Final Results:")
    print(f"  Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate malware classification model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--gradcam', action='store_true',
                       help='Generate Grad-CAM visualizations')
    parser.add_argument('--embeddings', action='store_true',
                       help='Generate t-SNE embedding visualization')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
