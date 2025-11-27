"""
Training Script for Malware Classification Models

This script provides a complete training pipeline for malware classification:
1. Model initialization (ResNet-50 or Custom CNN)
2. Data loading with augmentation
3. Training loop with validation
4. Early stopping and checkpointing
5. TensorBoard logging
6. Learning rate scheduling

Usage:
    python scripts/train.py --model resnet50 --epochs 50 --batch_size 32
    python scripts/train.py --model custom_cnn --epochs 100 --lr 0.001

Author: ML/Security Research Team
Date: November 26, 2025
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data_loader import create_data_loaders, get_class_names
from models.resnet import create_resnet_model
from models.custom_cnn import create_custom_cnn


class Trainer:
    """
    Trainer class for malware classification models.
    
    Handles:
    - Training and validation loops
    - Early stopping
    - Model checkpointing
    - TensorBoard logging
    - Learning rate scheduling
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler],
                 device: torch.device,
                 config: dict):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            config: Training configuration
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Create directories
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100. * correct / total:.2f}%"
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model (Val Acc: {self.best_val_acc:.2f}%)")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self,
              num_epochs: int,
              early_stopping_patience: int = 10) -> dict:
        """
        Main training loop.
        
        Args:
            num_epochs: Maximum number of epochs
            early_stopping_patience: Epochs to wait before early stopping
            
        Returns:
            Training history dictionary
        """
        print("\n" + "=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)
        print(f"Model: {self.config['model_name']}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch Size: {self.config['batch_size']}")
        print(f"Learning Rate: {self.config['learning_rate']}")
        print(f"Early Stopping Patience: {early_stopping_patience}")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
            print(f"  Best Val Acc: {self.best_val_acc:.2f}%")
            
            if is_best:
                print("  ⭐ New best model!")
            
            # Early stopping check
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠️ Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Total Time: {total_time / 60:.1f} minutes")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"Checkpoint saved to: {self.checkpoint_dir}")
        print("=" * 60)
        
        # Save final history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.writer.close()
        
        return self.history


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def main(args):
    """Main training function."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = get_device()
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'raw' / 'malimg'
    splits_dir = project_root / 'data' / 'splits'
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.model}_{timestamp}"
    
    # Create data loaders
    print("\n📦 Loading data...")
    loaders = create_data_loaders(
        data_dir=data_dir,
        splits_dir=splits_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        augmentation_level=args.augmentation,
        num_workers=args.num_workers,
        use_weighted_sampling=True
    )
    
    # Get class names and count
    class_names = get_class_names(splits_dir)
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")
    print(f"Training samples: {len(loaders['train'].dataset)}")
    print(f"Validation samples: {len(loaders['val'].dataset)}")
    
    # Create model
    print(f"\n🏗️ Creating model: {args.model}")
    if args.model == 'resnet50':
        model = create_resnet_model(
            num_classes=num_classes,
            variant='resnet50',
            pretrained=True,
            freeze_backbone=args.freeze_backbone
        )
    elif args.model == 'resnet18':
        model = create_resnet_model(
            num_classes=num_classes,
            variant='resnet18',
            pretrained=True,
            freeze_backbone=args.freeze_backbone
        )
    elif args.model == 'custom_cnn':
        model = create_custom_cnn(
            num_classes=num_classes,
            model_size='medium',
            use_attention=True
        )
    elif args.model == 'custom_cnn_small':
        model = create_custom_cnn(
            num_classes=num_classes,
            model_size='small'
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Print model info
    if hasattr(model, 'get_model_size'):
        info = model.get_model_size()
        print(f"Model parameters: {info['total_params']:,}")
        print(f"Trainable parameters: {info['trainable_params']:,}")
        print(f"Model size: {info['size_mb']:.2f} MB")
    
    # Loss function with class weights
    if args.use_class_weights:
        class_weights = loaders['train'].dataset.get_class_weights()
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted cross-entropy loss")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    print(f"Optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    else:
        scheduler = None
    
    # Training configuration
    config = {
        'model_name': args.model,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'augmentation': args.augmentation,
        'image_size': args.image_size,
        'num_classes': num_classes,
        'checkpoint_dir': str(project_root / 'checkpoints' / run_name),
        'log_dir': str(project_root / 'results' / 'logs' / run_name),
        'timestamp': timestamp
    }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience
    )
    
    return history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train malware classification model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'resnet18', 'custom_cnn', 'custom_cnn_small'],
                       help='Model architecture')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze backbone for transfer learning')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', '--lr', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'step', 'none'],
                       help='Learning rate scheduler')
    
    # Data arguments
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--augmentation', type=str, default='light',
                       choices=['none', 'light', 'medium', 'heavy'],
                       help='Data augmentation level')
    parser.add_argument('--use-class-weights', action='store_true',
                       help='Use class weights for imbalanced data')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
