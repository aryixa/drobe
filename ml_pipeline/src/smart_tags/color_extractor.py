"""
Task 12 - Smart Tags System: Color Extraction
Extract dominant colors from clothing images using OpenCV
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
import json
from dataclasses import dataclass
from enum import Enum

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.image_processing import enhance_image_quality


class ColorCategory(Enum):
    """Color categories for clothing"""
    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    PURPLE = "purple"
    PINK = "pink"
    BROWN = "brown"
    BLACK = "black"
    WHITE = "white"
    GRAY = "gray"
    BEIGE = "beige"
    CREAM = "cream"
    NAVY = "navy"
    MAROON = "maroon"
    OLIVE = "olive"
    TEAL = "teal"
    MULTI = "multi"
    NEUTRAL = "neutral"


@dataclass
class ColorInfo:
    """Color information structure"""
    rgb: Tuple[int, int, int]
    hsv: Tuple[float, float, float]
    category: ColorCategory
    percentage: float
    confidence: float


class ColorExtractor:
    """
    Advanced color extraction for clothing images
    - Dominant color detection
    - Color categorization
    - Confidence scoring
    """
    
    def __init__(self):
        # Color mapping ranges (HSV)
        self.color_ranges = self._init_color_ranges()
        
        # Color names for common colors
        self.color_names = self._init_color_names()
        
        # Analysis parameters
        self.n_clusters = 5  # Number of color clusters
        self.min_color_percentage = 0.05  # Minimum percentage to consider
        
        print("Color Extractor initialized")
    
    def _init_color_ranges(self) -> Dict[ColorCategory, Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """Initialize HSV color ranges for categorization"""
        return {
            ColorCategory.RED: ((0, 50, 50), (10, 255, 255)),
            ColorCategory.ORANGE: ((10, 50, 50), (20, 255, 255)),
            ColorCategory.YELLOW: ((20, 50, 50), (35, 255, 255)),
            ColorCategory.GREEN: ((35, 50, 50), (85, 255, 255)),
            ColorCategory.BLUE: ((85, 50, 50), (135, 255, 255)),
            ColorCategory.PURPLE: ((135, 50, 50), (170, 255, 255)),
            ColorCategory.PINK: ((170, 50, 50), (180, 255, 255)),
            ColorCategory.BROWN: ((8, 50, 20), (25, 255, 200)),
            ColorCategory.BLACK: ((0, 0, 0), (180, 255, 50)),
            ColorCategory.WHITE: ((0, 0, 200), (180, 30, 255)),
            ColorCategory.GRAY: ((0, 0, 50), (180, 30, 200)),
            ColorCategory.BEIGE: ((20, 20, 150), (40, 60, 255)),
            ColorCategory.NAVY: ((85, 50, 50), (115, 255, 150)),
            ColorCategory.MAROON: ((0, 50, 50), (10, 255, 150)),
            ColorCategory.OLIVE: ((35, 50, 50), (70, 255, 150)),
            ColorCategory.TEAL: ((70, 50, 50), (100, 255, 255)),
        }
    
    def _init_color_names(self) -> Dict[ColorCategory, str]:
        """Initialize display names for colors"""
        return {
            ColorCategory.RED: "red",
            ColorCategory.ORANGE: "orange",
            ColorCategory.YELLOW: "yellow",
            ColorCategory.GREEN: "green",
            ColorCategory.BLUE: "blue",
            ColorCategory.PURPLE: "purple",
            ColorCategory.PINK: "pink",
            ColorCategory.BROWN: "brown",
            ColorCategory.BLACK: "black",
            ColorCategory.WHITE: "white",
            ColorCategory.GRAY: "gray",
            ColorCategory.BEIGE: "beige",
            ColorCategory.CREAM: "cream",
            ColorCategory.NAVY: "navy",
            ColorCategory.MAROON: "maroon",
            ColorCategory.OLIVE: "olive",
            ColorCategory.TEAL: "teal",
            ColorCategory.MULTI: "multi",
            ColorCategory.NEUTRAL: "neutral"
        }
    
    def extract_colors(self, image_path: Union[str, Path], top_k: int = 3) -> List[ColorInfo]:
        """
        Extract dominant colors from image
        
        Args:
            image_path: Path to image file
            top_k: Number of top colors to return
            
        Returns:
            List of color information
        """
        image_path = Path(image_path)
        
        try:
            # Load and enhance image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Enhance image quality
            enhanced = enhance_image_quality(image_path)
            
            # Convert to RGB for processing
            rgb_image = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            
            # Reshape for clustering
            pixels = rgb_image.reshape(-1, 3)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers and labels
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Calculate color percentages
            label_counts = Counter(labels)
            total_pixels = len(pixels)
            
            # Create color info list
            color_infos = []
            for i, color in enumerate(colors):
                percentage = label_counts[i] / total_pixels
                
                if percentage >= self.min_color_percentage:
                    # Convert to HSV for categorization
                    hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
                    
                    # Categorize color
                    category = self._categorize_color(hsv_color)
                    
                    # Calculate confidence based on cluster purity
                    confidence = self._calculate_color_confidence(
                        pixels[labels == i], color
                    )
                    
                    color_info = ColorInfo(
                        rgb=tuple(map(int, color)),
                        hsv=tuple(map(float, hsv_color)),
                        category=category,
                        percentage=percentage,
                        confidence=confidence
                    )
                    
                    color_infos.append(color_info)
            
            # Sort by percentage
            color_infos.sort(key=lambda x: x.percentage, reverse=True)
            
            # Return top k colors
            return color_infos[:top_k]
            
        except Exception as e:
            print(f"Error extracting colors from {image_path}: {e}")
            return []
    
    def _categorize_color(self, hsv_color: Tuple[float, float, float]) -> ColorCategory:
        """Categorize color based on HSV values"""
        h, s, v = hsv_color
        
        # Check for special cases first
        if v < 50:  # Very dark
            return ColorCategory.BLACK
        elif v > 200 and s < 30:  # Very light and low saturation
            return ColorCategory.WHITE
        elif s < 30:  # Low saturation
            if v < 100:
                return ColorCategory.GRAY
            elif v > 150:
                return ColorCategory.BEIGE
            else:
                return ColorCategory.GRAY
        
        # Check color ranges
        for category, (lower, upper) in self.color_ranges.items():
            if (lower[0] <= h <= upper[0] and 
                lower[1] <= s <= upper[1] and 
                lower[2] <= v <= upper[2]):
                return category
        
        # Default to multi if no category matches
        return ColorCategory.MULTI
    
    def _calculate_color_confidence(self, cluster_pixels: np.ndarray, center_color: np.ndarray) -> float:
        """Calculate confidence score for color detection"""
        if len(cluster_pixels) == 0:
            return 0.0
        
        # Calculate standard deviation from center
        distances = np.linalg.norm(cluster_pixels - center_color, axis=1)
        std_distance = np.std(distances)
        
        # Convert to confidence (lower std = higher confidence)
        max_std = 100.0  # Maximum expected standard deviation
        confidence = max(0.0, 1.0 - (std_distance / max_std))
        
        return float(confidence)
    
    def get_primary_color(self, image_path: Union[str, Path]) -> Optional[ColorInfo]:
        """Get the primary (dominant) color from image"""
        colors = self.extract_colors(image_path, top_k=1)
        return colors[0] if colors else None
    
    def get_color_palette(self, image_path: Union[str, Path], n_colors: int = 5) -> List[ColorInfo]:
        """Get color palette from image"""
        return self.extract_colors(image_path, top_k=n_colors)
    
    def analyze_color_harmony(self, image_path: Union[str, Path]) -> Dict[str, float]:
        """Analyze color harmony in the image"""
        colors = self.extract_colors(image_path, top_k=5)
        
        if len(colors) < 2:
            return {'harmony_score': 1.0, 'harmony_type': 'monochrome'}
        
        # Get color categories
        categories = [color.category for color in colors]
        
        # Analyze harmony
        harmony_score = 0.0
        harmony_type = 'mixed'
        
        # Monochromatic (all same category)
        if len(set(categories)) == 1:
            harmony_score = 0.9
            harmony_type = 'monochromatic'
        
        # Complementary colors (opposite on color wheel)
        complementary_pairs = [
            (ColorCategory.RED, ColorCategory.GREEN),
            (ColorCategory.BLUE, ColorCategory.ORANGE),
            (ColorCategory.YELLOW, ColorCategory.PURPLE)
        ]
        
        for pair in complementary_pairs:
            if all(cat in categories for cat in pair):
                harmony_score = max(harmony_score, 0.8)
                harmony_type = 'complementary'
        
        # Analogous colors (adjacent on color wheel)
        analogous_groups = [
            [ColorCategory.RED, ColorCategory.ORANGE, ColorCategory.YELLOW],
            [ColorCategory.YELLOW, ColorCategory.GREEN, ColorCategory.BLUE],
            [ColorCategory.BLUE, ColorCategory.PURPLE, ColorCategory.RED]
        ]
        
        for group in analogous_groups:
            if len(set(categories) & set(group)) >= 2:
                harmony_score = max(harmony_score, 0.7)
                harmony_type = 'analogous'
        
        # Neutral base
        neutral_colors = [ColorCategory.BLACK, ColorCategory.WHITE, ColorCategory.GRAY, ColorCategory.BEIGE]
        neutral_count = sum(1 for cat in categories if cat in neutral_colors)
        
        if neutral_count >= 2 and len(categories) - neutral_count == 1:
            harmony_score = max(harmony_score, 0.85)
            harmony_type = 'neutral_base'
        
        # Default score based on color variety
        if harmony_score == 0.0:
            unique_colors = len(set(categories))
            harmony_score = max(0.3, 1.0 - (unique_colors - 1) * 0.2)
            harmony_type = 'varied'
        
        return {
            'harmony_score': harmony_score,
            'harmony_type': harmony_type,
            'color_count': len(set(categories)),
            'neutral_ratio': neutral_count / len(categories)
        }
    
    def detect_color_temperature(self, image_path: Union[str, Path]) -> str:
        """Detect if image has warm or cool colors"""
        colors = self.extract_colors(image_path, top_k=3)
        
        if not colors:
            return 'neutral'
        
        warm_colors = [ColorCategory.RED, ColorCategory.ORANGE, ColorCategory.YELLOW, 
                       ColorCategory.PINK, ColorCategory.BROWN]
        cool_colors = [ColorCategory.BLUE, ColorCategory.GREEN, ColorCategory.PURPLE, 
                      ColorCategory.TEAL, ColorCategory.NAVY]
        
        warm_weight = 0.0
        cool_weight = 0.0
        
        for color in colors:
            if color.category in warm_colors:
                warm_weight += color.percentage
            elif color.category in cool_colors:
                cool_weight += color.percentage
        
        if warm_weight > cool_weight * 1.2:
            return 'warm'
        elif cool_weight > warm_weight * 1.2:
            return 'cool'
        else:
            return 'neutral'
    
    def export_color_analysis(self, image_path: Union[str, Path], output_path: Union[str, Path]) -> Dict:
        """Export complete color analysis to JSON"""
        colors = self.extract_colors(image_path, top_k=5)
        harmony = self.analyze_color_harmony(image_path)
        temperature = self.detect_color_temperature(image_path)
        
        analysis = {
            'image_path': str(image_path),
            'colors': [
                {
                    'rgb': color.rgb,
                    'category': color.category.value,
                    'percentage': color.percentage,
                    'confidence': color.confidence
                }
                for color in colors
            ],
            'primary_color': colors[0].category.value if colors else None,
            'harmony': harmony,
            'temperature': temperature,
            'color_count': len(colors)
        }
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def batch_extract_colors(self, image_dir: Union[str, Path], output_dir: Union[str, Path]) -> Dict[str, Dict]:
        """Batch extract colors from directory of images"""
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Get image files
        image_files = list(image_dir.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
        
        print(f"Processing {len(image_files)} images for color extraction...")
        
        for image_file in image_files:
            try:
                analysis = self.export_color_analysis(
                    image_file, 
                    output_dir / f"{image_file.stem}_colors.json"
                )
                results[str(image_file)] = analysis
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                results[str(image_file)] = {'error': str(e)}
        
        print(f"Completed color extraction for {len(results)} images")
        return results
    
    def get_color_statistics(self, color_analyses: List[Dict]) -> Dict:
        """Get statistics from multiple color analyses"""
        if not color_analyses:
            return {}
        
        # Count color categories
        color_counts = Counter()
        temperature_counts = Counter()
        harmony_counts = Counter()
        
        for analysis in color_analyses:
            if 'primary_color' in analysis and analysis['primary_color']:
                color_counts[analysis['primary_color']] += 1
            
            if 'temperature' in analysis:
                temperature_counts[analysis['temperature']] += 1
            
            if 'harmony' in analysis and 'harmony_type' in analysis['harmony']:
                harmony_counts[analysis['harmony']['harmony_type']] += 1
        
        return {
            'total_analyzed': len(color_analyses),
            'color_distribution': dict(color_counts),
            'temperature_distribution': dict(temperature_counts),
            'harmony_distribution': dict(harmony_counts),
            'most_common_color': color_counts.most_common(1)[0] if color_counts else None,
            'dominant_temperature': temperature_counts.most_common(1)[0] if temperature_counts else None
        }


# Test function
def test_color_extractor():
    """Test the color extractor"""
    print("Testing Color Extractor...")
    
    extractor = ColorExtractor()
    
    # Test with sample images if available
    sample_dir = Path(__file__).parent.parent.parent / "data" / "sample_images"
    
    if sample_dir.exists():
        image_files = list(sample_dir.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
        
        if image_files:
            test_image = image_files[0]
            print(f"Testing with: {test_image}")
            
            # Test color extraction
            colors = extractor.extract_colors(test_image, top_k=3)
            print(f"Extracted {len(colors)} colors")
            
            for i, color in enumerate(colors):
                print(f"  {i+1}. {color.category.value} ({color.percentage:.1%}) - Confidence: {color.confidence:.2f}")
            
            # Test primary color
            primary = extractor.get_primary_color(test_image)
            if primary:
                print(f"Primary color: {primary.category.value}")
            
            # Test harmony analysis
            harmony = extractor.analyze_color_harmony(test_image)
            print(f"Harmony: {harmony['harmony_type']} (score: {harmony['harmony_score']:.2f})")
            
            # Test temperature
            temperature = extractor.detect_color_temperature(test_image)
            print(f"Temperature: {temperature}")
            
            # Test export
            output_path = sample_dir / f"{test_image.stem}_analysis.json"
            analysis = extractor.export_color_analysis(test_image, output_path)
            print(f"Analysis saved to: {output_path}")
            
        else:
            print("No sample images found")
    else:
        print("Sample images directory not found")
    
    print("Color extractor test completed!")


if __name__ == "__main__":
    test_color_extractor()
