"""
Evaluation Metrics for Malware Classification

This module provides comprehensive evaluation metrics for security applications:
1. Standard ML metrics (accuracy, precision, recall, F1)
2. Confusion matrix analysis
3. ROC curves and AUC scores
4. Per-class performance breakdown
5. Security-specific metrics

Design Decisions:
- High recall prioritized (minimize false negatives - missed malware)
- Per-class metrics for understanding failure modes
- Multi-class ROC using one-vs-rest strategy

Author: ML/Security Research Team
Date: November 26, 2025
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """
    Calculator for comprehensive evaluation metrics.
    
    Computes and stores:
    - Overall accuracy, precision, recall, F1
    - Per-class metrics
    - Confusion matrix
    - ROC curves and AUC
    """
    
    def __init__(self, 
                 class_names: List[str],
                 average: str = 'weighted'):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: List of class names (malware families)
            average: Averaging strategy ('micro', 'macro', 'weighted')
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.average = average
        
        # Storage for predictions
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def reset(self) -> None:
        """Reset stored predictions."""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self,
               predictions: Union[np.ndarray, torch.Tensor],
               labels: Union[np.ndarray, torch.Tensor],
               probabilities: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
        """
        Update with a batch of predictions.
        
        Args:
            predictions: Predicted class indices
            labels: Ground truth labels
            probabilities: Class probabilities (for ROC curves)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        self.all_preds.extend(predictions.flatten())
        self.all_labels.extend(labels.flatten())
        
        if probabilities is not None:
            self.all_probs.extend(probabilities)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric values
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
            'precision_weighted': precision_score(labels, preds, average='weighted', zero_division=0),
            'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
            'recall_weighted': recall_score(labels, preds, average='weighted', zero_division=0),
            'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
        }
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get the confusion matrix."""
        return confusion_matrix(self.all_labels, self.all_preds)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        return classification_report(
            self.all_labels,
            self.all_preds,
            target_names=self.class_names,
            zero_division=0
        )
    
    def get_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get metrics for each class.
        
        Returns:
            Dictionary mapping class names to their metrics
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        # Per-class precision, recall, F1
        precision = precision_score(labels, preds, average=None, zero_division=0)
        recall = recall_score(labels, preds, average=None, zero_division=0)
        f1 = f1_score(labels, preds, average=None, zero_division=0)
        
        # Support (number of true instances per class)
        cm = confusion_matrix(labels, preds)
        support = cm.sum(axis=1)
        
        # Per-class accuracy
        per_class_acc = cm.diagonal() / support
        
        per_class = {}
        for i, name in enumerate(self.class_names):
            per_class[name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'accuracy': per_class_acc[i] if support[i] > 0 else 0.0,
                'support': int(support[i])
            }
        
        return per_class
    
    def compute_roc_auc(self) -> Dict[str, float]:
        """
        Compute ROC-AUC scores (requires probabilities).
        
        Returns:
            Dictionary with per-class and average AUC scores
        """
        if len(self.all_probs) == 0:
            return {'error': 'No probabilities available'}
        
        probs = np.array(self.all_probs)
        labels = np.array(self.all_labels)
        
        # One-hot encode labels
        labels_onehot = np.zeros((len(labels), self.num_classes))
        labels_onehot[np.arange(len(labels)), labels] = 1
        
        auc_scores = {}
        
        # Per-class AUC
        for i, name in enumerate(self.class_names):
            try:
                fpr, tpr, _ = roc_curve(labels_onehot[:, i], probs[:, i])
                auc_scores[name] = auc(fpr, tpr)
            except:
                auc_scores[name] = float('nan')
        
        # Macro average AUC
        valid_aucs = [v for v in auc_scores.values() if not np.isnan(v)]
        auc_scores['macro_avg'] = np.mean(valid_aucs) if valid_aucs else 0.0
        
        return auc_scores


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: List[str],
                         figsize: Tuple[int, int] = (12, 10),
                         normalize: bool = True,
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        figsize: Figure size
        normalize: If True, normalize by row (true labels)
        save_path: Path to save the figure
    """
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
        cm_display = cm_normalized
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_roc_curves(labels: np.ndarray,
                   probs: np.ndarray,
                   class_names: List[str],
                   figsize: Tuple[int, int] = (10, 8),
                   save_path: Optional[str] = None) -> None:
    """
    Plot multi-class ROC curves.
    
    Args:
        labels: True labels
        probs: Predicted probabilities
        class_names: Class names
        figsize: Figure size
        save_path: Path to save figure
    """
    num_classes = len(class_names)
    
    # One-hot encode labels
    labels_onehot = np.zeros((len(labels), num_classes))
    labels_onehot[np.arange(len(labels)), labels] = 1
    
    plt.figure(figsize=figsize)
    
    # Plot ROC for each class
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    
    for i, (name, color) in enumerate(zip(class_names, colors)):
        try:
            fpr, tpr, _ = roc_curve(labels_onehot[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=1.5,
                    label=f'{name} (AUC = {roc_auc:.3f})')
        except:
            pass
    
    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curves', fontsize=14)
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")
    
    plt.show()


def plot_per_class_performance(per_class_metrics: Dict[str, Dict[str, float]],
                               metric: str = 'f1',
                               figsize: Tuple[int, int] = (12, 6),
                               save_path: Optional[str] = None) -> None:
    """
    Plot per-class performance as bar chart.
    
    Args:
        per_class_metrics: Dictionary from get_per_class_metrics
        metric: Which metric to plot ('precision', 'recall', 'f1', 'accuracy')
        figsize: Figure size
        save_path: Path to save figure
    """
    classes = list(per_class_metrics.keys())
    values = [per_class_metrics[c][metric] for c in classes]
    supports = [per_class_metrics[c]['support'] for c in classes]
    
    # Sort by value
    sorted_indices = np.argsort(values)
    classes = [classes[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    supports = [supports[i] for i in sorted_indices]
    
    plt.figure(figsize=figsize)
    
    # Color by support (more samples = darker)
    colors = plt.cm.Blues(np.array(supports) / max(supports))
    
    bars = plt.barh(classes, values, color=colors)
    
    # Add value labels
    for bar, val, sup in zip(bars, values, supports):
        plt.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f} (n={sup})', va='center', fontsize=8)
    
    plt.xlabel(metric.capitalize(), fontsize=12)
    plt.title(f'Per-Class {metric.capitalize()} Score', fontsize=14)
    plt.xlim(0, 1.15)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class plot saved to: {save_path}")
    
    plt.show()


def plot_training_history(history: Dict[str, List[float]],
                         figsize: Tuple[int, int] = (14, 5),
                         save_path: Optional[str] = None) -> None:
    """
    Plot training history (loss, accuracy, learning rate).
    
    Args:
        history: Training history dictionary
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in history:
        axes[2].plot(epochs, history['lr'], 'g-')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
    
    plt.show()


def generate_evaluation_report(metrics_calculator: MetricsCalculator,
                              save_dir: Optional[str] = None) -> str:
    """
    Generate comprehensive evaluation report.
    
    Args:
        metrics_calculator: MetricsCalculator with computed metrics
        save_dir: Directory to save plots
        
    Returns:
        Report as string
    """
    report = []
    report.append("=" * 60)
    report.append("MALWARE CLASSIFICATION EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Overall metrics
    metrics = metrics_calculator.compute()
    report.append("OVERALL METRICS")
    report.append("-" * 40)
    report.append(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    report.append(f"Precision (macro):  {metrics['precision_macro']:.4f}")
    report.append(f"Precision (weight): {metrics['precision_weighted']:.4f}")
    report.append(f"Recall (macro):     {metrics['recall_macro']:.4f}")
    report.append(f"Recall (weighted):  {metrics['recall_weighted']:.4f}")
    report.append(f"F1 (macro):         {metrics['f1_macro']:.4f}")
    report.append(f"F1 (weighted):      {metrics['f1_weighted']:.4f}")
    report.append("")
    
    # Classification report
    report.append("DETAILED CLASSIFICATION REPORT")
    report.append("-" * 40)
    report.append(metrics_calculator.get_classification_report())
    
    # ROC-AUC if available
    auc_scores = metrics_calculator.compute_roc_auc()
    if 'error' not in auc_scores:
        report.append("ROC-AUC SCORES")
        report.append("-" * 40)
        for name, score in sorted(auc_scores.items()):
            if name != 'macro_avg':
                report.append(f"  {name:20s}: {score:.4f}")
        report.append(f"  {'Macro Average':20s}: {auc_scores['macro_avg']:.4f}")
        report.append("")
    
    # Find worst performing classes
    per_class = metrics_calculator.get_per_class_metrics()
    sorted_by_f1 = sorted(per_class.items(), key=lambda x: x[1]['f1'])
    
    report.append("CLASSES WITH LOWEST F1 SCORE")
    report.append("-" * 40)
    for name, m in sorted_by_f1[:5]:
        report.append(f"  {name:20s}: F1={m['f1']:.3f}, Recall={m['recall']:.3f}, Support={m['support']}")
    report.append("")
    
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    # Save report
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / 'evaluation_report.txt', 'w') as f:
            f.write(report_text)
        
        # Generate and save plots
        cm = metrics_calculator.get_confusion_matrix()
        plot_confusion_matrix(
            cm, metrics_calculator.class_names,
            save_path=str(save_dir / 'confusion_matrix.png')
        )
        
        plot_per_class_performance(
            per_class, metric='f1',
            save_path=str(save_dir / 'per_class_f1.png')
        )
    
    return report_text


if __name__ == "__main__":
    # Test metrics calculator
    print("Testing Metrics Calculator")
    print("=" * 50)
    
    # Simulate predictions
    np.random.seed(42)
    
    class_names = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'VB.AT']
    num_classes = len(class_names)
    num_samples = 500
    
    # Generate random predictions (simulate 85% accuracy)
    true_labels = np.random.randint(0, num_classes, num_samples)
    predictions = true_labels.copy()
    
    # Add some errors
    error_mask = np.random.random(num_samples) < 0.15
    predictions[error_mask] = np.random.randint(0, num_classes, error_mask.sum())
    
    # Generate probabilities
    probs = np.random.dirichlet(np.ones(num_classes), num_samples)
    
    # Create calculator
    calc = MetricsCalculator(class_names)
    calc.update(predictions, true_labels, probs)
    
    # Compute metrics
    metrics = calc.compute()
    print("\nOverall Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(calc.get_classification_report())
    
    # Per-class metrics
    print("\nPer-class F1 Scores:")
    per_class = calc.get_per_class_metrics()
    for name, m in per_class.items():
        print(f"  {name}: {m['f1']:.3f}")
