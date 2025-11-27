"""
Binary-to-Image Conversion Utilities for Malware Detection

This module provides functions to convert binary executables (PE, ELF files)
into grayscale images for CNN-based malware classification.

The core idea: Every executable is a sequence of bytes (0-255), which can be
interpreted as pixel intensities to create a visual representation.

Design Decisions:
1. Fixed 224×224 output - Compatible with ImageNet pre-trained models
2. Grayscale - Binary data naturally maps to single-channel images
3. Resize with padding - Preserves aspect ratio, adds zero padding
4. Anti-aliasing - Smooth downscaling to preserve patterns

Author: ML/Security Research Team
Date: November 26, 2025
"""

import os
import math
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
from PIL import Image


def calculate_image_dimensions(file_size: int, 
                               max_width: int = 1024) -> Tuple[int, int]:
    """
    Calculate optimal 2D dimensions for a binary file.
    
    Uses a heuristic based on file size to determine width,
    then calculates height to fit all bytes.
    
    The width selection follows the original Malimg paper methodology:
    - Files < 10KB: width = 32
    - Files 10KB - 30KB: width = 64
    - Files 30KB - 60KB: width = 128
    - Files 60KB - 100KB: width = 256
    - Files 100KB - 200KB: width = 384
    - Files 200KB - 500KB: width = 512
    - Files 500KB - 1MB: width = 768
    - Files > 1MB: width = 1024
    
    Args:
        file_size: Size of the binary file in bytes
        max_width: Maximum allowed width
        
    Returns:
        Tuple of (width, height)
    """
    # Width selection based on file size (from Malimg paper)
    if file_size < 10 * 1024:
        width = 32
    elif file_size < 30 * 1024:
        width = 64
    elif file_size < 60 * 1024:
        width = 128
    elif file_size < 100 * 1024:
        width = 256
    elif file_size < 200 * 1024:
        width = 384
    elif file_size < 500 * 1024:
        width = 512
    elif file_size < 1024 * 1024:
        width = 768
    else:
        width = min(1024, max_width)
    
    # Calculate height
    height = math.ceil(file_size / width)
    
    return width, height


def bytes_to_image(data: bytes, 
                   target_size: Tuple[int, int] = (224, 224),
                   keep_aspect_ratio: bool = True,
                   use_heuristic_dims: bool = True) -> Image.Image:
    """
    Convert raw bytes to a grayscale PIL Image.
    
    Args:
        data: Raw bytes from a binary file
        target_size: Target output dimensions (width, height)
        keep_aspect_ratio: If True, pad to maintain aspect ratio
        use_heuristic_dims: If True, use Malimg paper heuristics for initial dimensions
        
    Returns:
        PIL Image in grayscale mode ('L')
    """
    # Convert bytes to numpy array
    byte_array = np.frombuffer(data, dtype=np.uint8)
    file_size = len(byte_array)
    
    if file_size == 0:
        # Return blank image for empty files
        return Image.new('L', target_size, 0)
    
    # Calculate dimensions
    if use_heuristic_dims:
        width, height = calculate_image_dimensions(file_size)
    else:
        # Square-ish dimensions
        width = int(math.sqrt(file_size))
        height = math.ceil(file_size / width)
    
    # Pad array to fit dimensions
    total_pixels = width * height
    if file_size < total_pixels:
        # Pad with zeros
        padded = np.zeros(total_pixels, dtype=np.uint8)
        padded[:file_size] = byte_array
        byte_array = padded
    else:
        # Truncate if needed
        byte_array = byte_array[:total_pixels]
    
    # Reshape to 2D
    image_array = byte_array.reshape((height, width))
    
    # Convert to PIL Image
    img = Image.fromarray(image_array, mode='L')
    
    # Resize to target size
    if keep_aspect_ratio:
        img = resize_with_padding(img, target_size)
    else:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    return img


def resize_with_padding(img: Image.Image, 
                        target_size: Tuple[int, int],
                        pad_color: int = 0) -> Image.Image:
    """
    Resize image to target size while maintaining aspect ratio.
    Pads with specified color to fill remaining space.
    
    Args:
        img: Input PIL Image
        target_size: Target dimensions (width, height)
        pad_color: Padding pixel value (0-255)
        
    Returns:
        Resized and padded PIL Image
    """
    target_w, target_h = target_size
    orig_w, orig_h = img.size
    
    # Calculate scaling factor
    scale = min(target_w / orig_w, target_h / orig_h)
    
    # New dimensions
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # Resize with anti-aliasing
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create padded image
    padded = Image.new('L', target_size, pad_color)
    
    # Center the resized image
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))
    
    return padded


def binary_file_to_image(file_path: Union[str, Path],
                         target_size: Tuple[int, int] = (224, 224),
                         save_path: Optional[Union[str, Path]] = None) -> Image.Image:
    """
    Convert a binary file to a grayscale image.
    
    This is the main entry point for binary-to-image conversion.
    
    Args:
        file_path: Path to the binary file (PE, ELF, etc.)
        target_size: Target output dimensions
        save_path: Optional path to save the image
        
    Returns:
        PIL Image in grayscale mode
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        IOError: If the file cannot be read
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read binary data
    with open(file_path, 'rb') as f:
        data = f.read()
    
    # Convert to image
    img = bytes_to_image(data, target_size)
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
    
    return img


def image_to_tensor(img: Image.Image, 
                    normalize: bool = True) -> np.ndarray:
    """
    Convert PIL Image to numpy array suitable for PyTorch.
    
    Args:
        img: PIL Image in grayscale mode
        normalize: If True, normalize to [0, 1] range
        
    Returns:
        NumPy array of shape (1, H, W) for single-channel
    """
    # Convert to numpy
    arr = np.array(img, dtype=np.float32)
    
    # Normalize
    if normalize:
        arr = arr / 255.0
    
    # Add channel dimension
    arr = np.expand_dims(arr, axis=0)
    
    return arr


def visualize_binary_structure(file_path: Union[str, Path],
                               figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Create a detailed visualization of a binary file's structure.
    
    Shows:
    1. Raw byte visualization
    2. Entropy distribution
    3. Byte frequency histogram
    
    Args:
        file_path: Path to the binary file
        figsize: Figure size for matplotlib
    """
    import matplotlib.pyplot as plt
    from scipy import ndimage
    
    file_path = Path(file_path)
    
    # Read file
    with open(file_path, 'rb') as f:
        data = f.read()
    
    byte_array = np.frombuffer(data, dtype=np.uint8)
    file_size = len(byte_array)
    
    # Calculate dimensions
    width, height = calculate_image_dimensions(file_size)
    
    # Pad and reshape
    total = width * height
    if file_size < total:
        padded = np.zeros(total, dtype=np.uint8)
        padded[:file_size] = byte_array
        byte_array = padded
    else:
        byte_array = byte_array[:total]
    
    img_array = byte_array.reshape((height, width))
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Raw visualization
    axes[0].imshow(img_array, cmap='gray', aspect='auto')
    axes[0].set_title(f'Binary Visualization\n{file_path.name}\nSize: {file_size:,} bytes')
    axes[0].axis('off')
    
    # 2. Local entropy (using variance as proxy)
    entropy = ndimage.generic_filter(
        img_array.astype(float),
        np.var,
        size=16
    )
    axes[1].imshow(entropy, cmap='hot', aspect='auto')
    axes[1].set_title('Local Variance (Entropy Proxy)\nHigh = Complex code/packed')
    axes[1].axis('off')
    
    # 3. Byte frequency histogram
    hist, bins = np.histogram(byte_array, bins=256, range=(0, 256))
    axes[2].bar(range(256), hist, width=1, color='steelblue')
    axes[2].set_xlabel('Byte Value')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Byte Distribution\nSpikes indicate structure')
    
    plt.tight_layout()
    plt.show()


def batch_convert_directory(input_dir: Union[str, Path],
                           output_dir: Union[str, Path],
                           target_size: Tuple[int, int] = (224, 224),
                           extensions: Tuple[str, ...] = ('.exe', '.dll', '.bin')) -> int:
    """
    Convert all binary files in a directory to images.
    
    Maintains directory structure in output.
    
    Args:
        input_dir: Directory containing binary files
        output_dir: Directory to save converted images
        target_size: Target image dimensions
        extensions: File extensions to process
        
    Returns:
        Number of files processed
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    processed = 0
    
    for file_path in input_dir.rglob('*'):
        if file_path.suffix.lower() in extensions:
            # Calculate output path
            relative = file_path.relative_to(input_dir)
            output_path = output_dir / relative.with_suffix('.png')
            
            try:
                binary_file_to_image(file_path, target_size, output_path)
                processed += 1
                
                if processed % 100 == 0:
                    print(f"Processed {processed} files...")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Total files processed: {processed}")
    return processed


# ============================================================================
# PE File Analysis Utilities (for advanced features)
# ============================================================================

def extract_pe_features(file_path: Union[str, Path]) -> dict:
    """
    Extract static features from a PE (Portable Executable) file.
    
    Features extracted:
    - Section information (names, sizes, entropy)
    - Import table statistics
    - File header characteristics
    - Entropy of different regions
    
    Requires 'pefile' library.
    
    Args:
        file_path: Path to PE file
        
    Returns:
        Dictionary of extracted features
    """
    try:
        import pefile
    except ImportError:
        raise ImportError("pefile library required: pip install pefile")
    
    file_path = Path(file_path)
    features = {}
    
    try:
        pe = pefile.PE(str(file_path))
        
        # Basic info
        features['is_dll'] = pe.is_dll()
        features['is_exe'] = pe.is_exe()
        
        # Section analysis
        features['num_sections'] = len(pe.sections)
        section_entropies = []
        section_names = []
        
        for section in pe.sections:
            name = section.Name.decode().strip('\x00')
            section_names.append(name)
            section_entropies.append(section.get_entropy())
        
        features['section_names'] = section_names
        features['section_entropies'] = section_entropies
        features['avg_section_entropy'] = np.mean(section_entropies) if section_entropies else 0
        features['max_section_entropy'] = max(section_entropies) if section_entropies else 0
        
        # Import analysis
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            imports = []
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode()
                imports.append({
                    'dll': dll_name,
                    'num_functions': len(entry.imports)
                })
            features['imports'] = imports
            features['num_imported_dlls'] = len(imports)
            features['total_imported_functions'] = sum(i['num_functions'] for i in imports)
        else:
            features['imports'] = []
            features['num_imported_dlls'] = 0
            features['total_imported_functions'] = 0
        
        pe.close()
        
    except Exception as e:
        features['error'] = str(e)
    
    return features


def calculate_file_entropy(file_path: Union[str, Path]) -> float:
    """
    Calculate Shannon entropy of a file.
    
    Entropy measures randomness/unpredictability:
    - Low entropy (0-4): Structured data, lots of patterns
    - Medium entropy (4-6): Normal executable code
    - High entropy (6-8): Compressed, encrypted, or packed data
    
    Args:
        file_path: Path to the file
        
    Returns:
        Entropy value between 0 and 8
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    
    if len(data) == 0:
        return 0.0
    
    # Count byte frequencies
    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    
    # Calculate probabilities
    probabilities = byte_counts / len(data)
    
    # Remove zero probabilities
    probabilities = probabilities[probabilities > 0]
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy


if __name__ == "__main__":
    # Example usage and testing
    print("Binary-to-Image Conversion Utilities")
    print("=" * 50)
    
    # Test with a sample file if available
    import sys
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        if test_file.exists():
            print(f"\nProcessing: {test_file}")
            
            # Convert to image
            img = binary_file_to_image(test_file)
            print(f"Output size: {img.size}")
            
            # Calculate entropy
            entropy = calculate_file_entropy(test_file)
            print(f"File entropy: {entropy:.2f}")
            
            # Show visualization
            visualize_binary_structure(test_file)
        else:
            print(f"File not found: {test_file}")
    else:
        print("\nUsage: python binary_to_image.py <binary_file>")
        print("\nModule can be imported and used as:")
        print("  from utils.binary_to_image import binary_file_to_image")
        print("  img = binary_file_to_image('malware.exe')")
