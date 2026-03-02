"""
Image processing utilities
Common functions for image loading, preprocessing, and metadata extraction
"""

import cv2
import numpy as np
from PIL import Image, ExifTags
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import os


def get_image_metadata(image_path: Union[str, Path]) -> Dict:
    """
    Extract metadata from image file
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image metadata
    """
    image_path = Path(image_path)
    metadata = {}
    
    try:
        # Basic file info
        stat = image_path.stat()
        metadata['file_size'] = stat.st_size
        metadata['file_name'] = image_path.name
        metadata['file_extension'] = image_path.suffix.lower()
        
        # PIL Image info
        with Image.open(image_path) as img:
            metadata['width'] = img.width
            metadata['height'] = img.height
            metadata['mode'] = img.mode
            metadata['format'] = img.format
            
            # EXIF data if available
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if isinstance(value, (str, int, float)):
                        metadata[f'exif_{tag}'] = value
        
        # OpenCV info (color channels, etc.)
        cv_img = cv2.imread(str(image_path))
        if cv_img is not None:
            metadata['channels'] = cv_img.shape[2] if len(cv_img.shape) > 2 else 1
            metadata['dtype'] = str(cv_img.dtype)
            
    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")
        metadata['error'] = str(e)
    
    return metadata


def validate_image(image_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Validate if file is a valid image
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    image_path = Path(image_path)
    
    # Check file exists
    if not image_path.exists():
        return False, f"File does not exist: {image_path}"
    
    # Check file size
    if image_path.stat().st_size == 0:
        return False, f"Empty file: {image_path}"
    
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    if image_path.suffix.lower() not in valid_extensions:
        return False, f"Invalid file extension: {image_path.suffix}"
    
    # Try to open with PIL
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def resize_image_keep_aspect(image_path: Union[str, Path], 
                           target_size: Tuple[int, int] = (512, 512),
                           output_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image_path: Path to input image
        target_size: Target maximum size (width, height)
        output_path: Path for output image (default: same as input with _resized suffix)
        
    Returns:
        Path to resized image
    """
    image_path = Path(image_path)
    
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_resized{image_path.suffix}"
    
    output_path = Path(output_path)
    
    try:
        with Image.open(image_path) as img:
            # Calculate new size maintaining aspect ratio
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Save resized image
            img.save(output_path, quality=95, optimize=True)
        
        return output_path
        
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")
        raise


def crop_center(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Crop center of image
    
    Args:
        image: Input image as numpy array
        crop_size: Target crop size (width, height)
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    crop_w, crop_h = crop_size
    
    # Calculate crop coordinates
    start_x = max(0, (w - crop_w) // 2)
    start_y = max(0, (h - crop_h) // 2)
    
    end_x = min(w, start_x + crop_w)
    end_y = min(h, start_y + crop_h)
    
    return image[start_y:end_y, start_x:end_x]


def enhance_image_quality(image_path: Union[str, Path]) -> np.ndarray:
    """
    Enhance image quality for better embedding extraction
    
    Args:
        image_path: Path to image file
        
    Returns:
        Enhanced image as numpy array
    """
    image_path = Path(image_path)
    
    try:
        # Load with OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply enhancements
        # 1. Denoising
        denoised = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)
        
        # 2. Contrast enhancement (CLAHE)
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
        
    except Exception as e:
        print(f"Error enhancing image {image_path}: {e}")
        # Return original image if enhancement fails
        return cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)


def batch_process_images(image_dir: Union[str, Path], 
                        output_dir: Optional[Union[str, Path]] = None,
                        target_size: Tuple[int, int] = (512, 512)) -> Dict[str, Path]:
    """
    Batch process images in a directory
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory for processed images (default: same as input)
        target_size: Target size for resizing
        
    Returns:
        Dictionary mapping original paths to processed paths
    """
    image_dir = Path(image_dir)
    
    if output_dir is None:
        output_dir = image_dir / "processed"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_files = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    image_files = [f for f in image_dir.iterdir() 
                  if f.suffix.lower() in valid_extensions and f.is_file()]
    
    print(f"Processing {len(image_files)} images...")
    
    for image_file in image_files:
        try:
            # Validate image
            is_valid, error = validate_image(image_file)
            if not is_valid:
                print(f"Skipping {image_file}: {error}")
                continue
            
            # Resize image
            output_path = output_dir / image_file.name
            resized_path = resize_image_keep_aspect(image_file, target_size, output_path)
            
            processed_files[str(image_file)] = resized_path
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    print(f"Successfully processed {len(processed_files)} images")
    return processed_files


def create_image_collage(image_paths: List[Union[str, Path]], 
                        output_path: Union[str, Path],
                        grid_size: Tuple[int, int] = (3, 3),
                        thumb_size: Tuple[int, int] = (200, 200)) -> Path:
    """
    Create a collage from multiple images
    
    Args:
        image_paths: List of image paths
        output_path: Path for output collage
        grid_size: Grid dimensions (cols, rows)
        thumb_size: Thumbnail size for each image
        
    Returns:
        Path to created collage
    """
    output_path = Path(output_path)
    cols, rows = grid_size
    thumb_w, thumb_h = thumb_size
    
    # Create blank canvas
    collage = Image.new('RGB', (cols * thumb_w, rows * thumb_h), color='white')
    
    for i, image_path in enumerate(image_paths[:cols * rows]):
        try:
            with Image.open(image_path) as img:
                # Resize to thumbnail
                img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                
                # Calculate position
                x = (i % cols) * thumb_w
                y = (i // cols) * thumb_h
                
                # Paste onto collage (centered)
                offset_x = x + (thumb_w - img.width) // 2
                offset_y = y + (thumb_h - img.height) // 2
                collage.paste(img, (offset_x, offset_y))
                
        except Exception as e:
            print(f"Error adding {image_path} to collage: {e}")
    
    collage.save(output_path, quality=95)
    return output_path


# Test function
def test_image_processing():
    """Test image processing utilities"""
    print("Testing image processing utilities...")
    
    # Test metadata extraction
    sample_dir = Path(__file__).parent.parent.parent / "data" / "sample_images"
    
    if sample_dir.exists():
        image_files = list(sample_dir.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
        
        if image_files:
            test_image = image_files[0]
            print(f"Testing with: {test_image}")
            
            # Test metadata
            metadata = get_image_metadata(test_image)
            print(f"Metadata keys: {list(metadata.keys())}")
            
            # Test validation
            is_valid, error = validate_image(test_image)
            print(f"Valid image: {is_valid}, Error: {error}")
            
            # Test enhancement
            enhanced = enhance_image_quality(test_image)
            print(f"Enhanced image shape: {enhanced.shape}")
            
        else:
            print("No sample images found")
    else:
        print("Sample images directory not found")
    
    print("Image processing test completed!")


if __name__ == "__main__":
    test_image_processing()
