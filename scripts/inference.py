"""
Inference Pipeline for Malware Classification

This script provides a production-ready inference pipeline:
1. Load trained model
2. Process new binary files or images
3. Classify with confidence scores
4. Generate Grad-CAM explanations
5. Output structured results

Usage:
    # Classify a single image
    python scripts/inference.py --input sample.png --checkpoint checkpoint_best.pth
    
    # Classify a directory of images
    python scripts/inference.py --input samples/ --checkpoint checkpoint_best.pth
    
    # With Grad-CAM explanation
    python scripts/inference.py --input sample.png --checkpoint checkpoint_best.pth --explain

Author: ML/Security Research Team
Date: November 26, 2025
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data_loader import get_val_transforms, get_class_names
from utils.visualization import GradCAM, overlay_heatmap
from utils.binary_to_image import binary_file_to_image
from models.resnet import create_resnet_model
from models.custom_cnn import create_custom_cnn


class MalwareClassifier:
    """
    Production-ready malware classifier.
    
    Features:
    - Load trained models
    - Process images or binary files
    - Return predictions with confidence
    - Generate visual explanations (Grad-CAM)
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 device: Optional[str] = None,
                 class_names: Optional[List[str]] = None):
        """
        Initialize classifier.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
            class_names: List of class names (None = load from checkpoint)
        """
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model, self.config = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Get class names
        if class_names is not None:
            self.class_names = class_names
        else:
            # Try to load from splits directory
            splits_dir = project_root / 'data' / 'splits'
            if (splits_dir / 'label_mapping.csv').exists():
                self.class_names = get_class_names(splits_dir)
            else:
                self.class_names = [f"Class_{i}" for i in range(self.config['num_classes'])]
        
        # Setup transforms
        self.image_size = self.config.get('image_size', 224)
        self.transform = get_val_transforms(self.image_size)
        
        # Setup Grad-CAM
        self.gradcam = None
        
        print(f"Model loaded: {self.config['model_name']}")
        print(f"Classes: {len(self.class_names)}")
    
    def _load_model(self, checkpoint_path: str) -> Tuple[nn.Module, dict]:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # Create model architecture
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
        model = model.to(self.device)
        
        return model, config
    
    def _preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            Preprocessed tensor
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            
            # Check if it's a binary file (need to convert)
            if image_path.suffix.lower() in ['.exe', '.dll', '.bin', '.elf']:
                pil_image = binary_file_to_image(image_path, (self.image_size, self.image_size))
            else:
                pil_image = Image.open(image_path).convert('L')
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert('L')
        elif isinstance(image, Image.Image):
            pil_image = image.convert('L')
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict(self, 
                image: Union[str, Path, Image.Image, np.ndarray],
                top_k: int = 5) -> Dict:
        """
        Predict malware family for an image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Preprocess
        input_tensor = self._preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
        
        # Get top-k predictions
        probs_np = probabilities.cpu().numpy()[0]
        top_indices = np.argsort(probs_np)[::-1][:top_k]
        
        predictions = []
        for idx in top_indices:
            predictions.append({
                'class_name': self.class_names[idx],
                'class_idx': int(idx),
                'confidence': float(probs_np[idx])
            })
        
        # Timing
        inference_time = time.time() - start_time
        
        result = {
            'predicted_class': predictions[0]['class_name'],
            'predicted_idx': predictions[0]['class_idx'],
            'confidence': predictions[0]['confidence'],
            'top_predictions': predictions,
            'inference_time_ms': inference_time * 1000
        }
        
        return result
    
    def predict_with_explanation(self,
                                  image: Union[str, Path, Image.Image, np.ndarray],
                                  save_path: Optional[str] = None) -> Tuple[Dict, np.ndarray]:
        """
        Predict with Grad-CAM explanation.
        
        Args:
            image: Input image
            save_path: Path to save visualization
            
        Returns:
            Tuple of (prediction_result, heatmap_overlay)
        """
        import matplotlib.pyplot as plt
        
        # Get prediction
        result = self.predict(image)
        
        # Setup Grad-CAM if not already
        if self.gradcam is None:
            self.gradcam = GradCAM(self.model)
        
        # Generate heatmap
        input_tensor = self._preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        heatmap = self.gradcam.generate(input_tensor, result['predicted_idx'])
        
        # Create overlay
        orig_image = input_tensor.squeeze().cpu().numpy()
        overlay = overlay_heatmap(orig_image, heatmap, alpha=0.5)
        
        # Save visualization if requested
        if save_path:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(orig_image, cmap='gray')
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Attention Heatmap')
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title(f"Prediction: {result['predicted_class']}\n"
                            f"Confidence: {result['confidence']:.1%}")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            result['explanation_path'] = save_path
        
        return result, overlay
    
    def predict_batch(self,
                      images: List[Union[str, Path]],
                      batch_size: int = 32) -> List[Dict]:
        """
        Predict on a batch of images.
        
        Args:
            images: List of image paths
            batch_size: Batch size for inference
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_paths = images[i:i+batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img_path in batch_paths:
                try:
                    tensor = self._preprocess_image(img_path)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    results.append({
                        'image_path': str(img_path),
                        'error': str(e)
                    })
                    continue
            
            if not batch_tensors:
                continue
            
            # Stack and predict
            batch = torch.cat(batch_tensors, dim=0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1)
            
            # Process results
            for j, (path, prob) in enumerate(zip(batch_paths[:len(batch_tensors)], probs)):
                prob_np = prob.cpu().numpy()
                pred_idx = prob_np.argmax()
                
                results.append({
                    'image_path': str(path),
                    'predicted_class': self.class_names[pred_idx],
                    'predicted_idx': int(pred_idx),
                    'confidence': float(prob_np[pred_idx])
                })
        
        return results


def main(args):
    """Main inference function."""
    
    # Initialize classifier
    classifier = MalwareClassifier(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Get input files
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        image_paths = [input_path]
    elif input_path.is_dir():
        # Directory
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.exe', '.dll', '.bin']
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_path.glob(f'*{ext}'))
            image_paths.extend(input_path.glob(f'*{ext.upper()}'))
    else:
        print(f"Error: Input not found: {input_path}")
        return
    
    print(f"\nProcessing {len(image_paths)} file(s)...")
    
    # Create output directory
    output_dir = Path(args.output) if args.output else input_path.parent / 'predictions'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    all_results = []
    
    for img_path in image_paths:
        print(f"\n📄 Processing: {img_path.name}")
        
        try:
            if args.explain:
                # With Grad-CAM explanation
                save_path = output_dir / f"{img_path.stem}_explanation.png"
                result, _ = classifier.predict_with_explanation(
                    img_path, save_path=str(save_path)
                )
            else:
                result = classifier.predict(img_path, top_k=args.top_k)
            
            result['image_path'] = str(img_path)
            
            # Print result
            print(f"  Prediction: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Inference time: {result['inference_time_ms']:.1f}ms")
            
            if args.top_k > 1:
                print("  Top predictions:")
                for pred in result['top_predictions'][:args.top_k]:
                    print(f"    - {pred['class_name']}: {pred['confidence']:.2%}")
            
            all_results.append(result)
            
        except Exception as e:
            print(f"  Error: {e}")
            all_results.append({
                'image_path': str(img_path),
                'error': str(e)
            })
    
    # Save results
    results_path = output_dir / 'predictions.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    # Summary
    successful = [r for r in all_results if 'error' not in r]
    print(f"\nSummary: {len(successful)}/{len(all_results)} files processed successfully")
    
    if successful:
        avg_time = np.mean([r['inference_time_ms'] for r in successful])
        avg_conf = np.mean([r['confidence'] for r in successful])
        print(f"Average inference time: {avg_time:.1f}ms")
        print(f"Average confidence: {avg_conf:.2%}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Malware classification inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input file or directory')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictions to show')
    parser.add_argument('--explain', action='store_true',
                       help='Generate Grad-CAM explanations')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
