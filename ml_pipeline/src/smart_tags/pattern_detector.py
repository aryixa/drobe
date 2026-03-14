"""
Task 12 - Smart Tags System: Pattern Detection
Detect patterns in clothing images using simple logic and computer vision
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
from collections import Counter
from skimage import feature, measure
from skimage.filters import gabor
import matplotlib.pyplot as plt

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.image_processing import enhance_image_quality


class PatternType(Enum):
    """Pattern types for clothing"""
    SOLID = "solid"
    STRIPED = "striped"
    CHECKERED = "checkered"
    POLKA_DOT = "polka_dot"
    FLORAL = "floral"
    GEOMETRIC = "geometric"
    PLAID = "plaid"
    ABSTRACT = "abstract"
    ANIMAL = "animal"
    CAMOUFLAGE = "camouflage"
    TIE_DYE = "tie_dye"
    MARBLE = "marble"
    TEXTURED = "textured"


@dataclass
class PatternInfo:
    """Pattern detection result"""
    pattern_type: PatternType
    confidence: float
    characteristics: Dict[str, float]
    evidence: Dict[str, float]


class PatternDetector:
    """
    Pattern detection system for clothing images
    - Edge detection for stripes and checks
    - Blob detection for dots
    - Texture analysis for patterns
    - Simple logic-based classification
    """
    
    def __init__(self):
        # Pattern detection parameters
        self.edge_threshold = 50
        self.blob_min_size = 10
        self.blob_max_size = 100
        self.texture_threshold = 0.1
        
        # Gabor filter parameters for texture analysis
        self.gabor_frequencies = [0.1, 0.3, 0.5]
        self.gabor_angles = [0, 45, 90, 135]
        
        print("Pattern Detector initialized")
    
    def detect_pattern(self, image_path: Union[str, Path]) -> PatternInfo:
        """
        Detect pattern type from image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Pattern detection result
        """
        image_path = Path(image_path)
        
        try:
            # Load and enhance image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhance image quality
            enhanced = enhance_image_quality(image_path)
            enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            
            # Analyze different pattern characteristics
            characteristics = {}
            evidence = {}
            
            # 1. Edge analysis (for stripes, checks)
            edge_score = self._analyze_edges(enhanced_gray)
            characteristics['edge_density'] = edge_score
            evidence['striped'] = edge_score * 0.7
            evidence['checkered'] = edge_score * 0.5
            
            # 2. Blob analysis (for dots, polka dots)
            blob_score = self._analyze_blobs(enhanced_gray)
            characteristics['blob_density'] = blob_score
            evidence['polka_dot'] = blob_score * 0.8
            evidence['floral'] = blob_score * 0.4
            
            # 3. Texture analysis (for various patterns)
            texture_score = self._analyze_texture(enhanced_gray)
            characteristics['texture_complexity'] = texture_score
            evidence['geometric'] = texture_score * 0.6
            evidence['abstract'] = texture_score * 0.5
            
            # 4. Periodicity analysis (for regular patterns)
            periodicity_score = self._analyze_periodicity(enhanced_gray)
            characteristics['periodicity'] = periodicity_score
            evidence['striped'] += periodicity_score * 0.8
            evidence['checkered'] += periodicity_score * 0.6
            evidence['plaid'] += periodicity_score * 0.7
            
            # 5. Color variation analysis
            color_variation = self._analyze_color_variation(image)
            characteristics['color_variation'] = color_variation
            evidence['floral'] += color_variation * 0.6
            evidence['tie_dye'] += color_variation * 0.8
            evidence['marble'] += color_variation * 0.5
            
            # 6. Symmetry analysis (for geometric patterns)
            symmetry_score = self._analyze_symmetry(enhanced_gray)
            characteristics['symmetry'] = symmetry_score
            evidence['geometric'] += symmetry_score * 0.5
            evidence['checkered'] += symmetry_score * 0.4
            
            # Determine primary pattern
            if evidence:
                sorted_patterns = sorted(evidence.items(), key=lambda x: x[1], reverse=True)
                primary_pattern_type = PatternType(sorted_patterns[0][0])
                confidence = min(1.0, sorted_patterns[0][1])
            else:
                # Default to solid if no patterns detected
                primary_pattern_type = PatternType.SOLID
                confidence = 0.8
                characteristics['solid_score'] = 0.8
            
            # Check if it's actually solid (low pattern evidence)
            total_pattern_evidence = sum(evidence.values())
            if total_pattern_evidence < 0.3:
                primary_pattern_type = PatternType.SOLID
                confidence = 0.9
                characteristics['solid_score'] = 0.9
            
            return PatternInfo(
                pattern_type=primary_pattern_type,
                confidence=confidence,
                characteristics=characteristics,
                evidence=evidence
            )
            
        except Exception as e:
            print(f"Error detecting pattern in {image_path}: {e}")
            return PatternInfo(
                pattern_type=PatternType.SOLID,
                confidence=0.0,
                characteristics={},
                evidence={}
            )
    
    def _analyze_edges(self, gray_image: np.ndarray) -> float:
        """Analyze edge density for stripe/checker patterns"""
        # Canny edge detection
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Calculate edge density
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        edge_density = edge_pixels / total_pixels
        
        # Normalize to 0-1 range
        return min(1.0, edge_density * 10)  # Scale up for better sensitivity
    
    def _analyze_blobs(self, gray_image: np.ndarray) -> float:
        """Analyze blob density for dot/floral patterns"""
        # Use blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = self.blob_min_size
        params.maxArea = self.blob_max_size
        params.filterByCircularity = True
        params.minCircularity = 0.5
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray_image)
        
        # Calculate blob density
        blob_count = len(keypoints)
        image_area = gray_image.size
        blob_density = blob_count / (image_area / 1000)  # Normalize by image size
        
        return min(1.0, blob_density)
    
    def _analyze_texture(self, gray_image: np.ndarray) -> float:
        """Analyze texture complexity using Gabor filters"""
        texture_responses = []
        
        for frequency in self.gabor_frequencies:
            for angle in self.gabor_angles:
                # Apply Gabor filter
                real, _ = gabor(gray_image, frequency=frequency, theta=np.radians(angle))
                response = np.mean(np.abs(real))
                texture_responses.append(response)
        
        # Calculate average texture response
        avg_texture = np.mean(texture_responses)
        
        # Normalize to 0-1 range
        return min(1.0, avg_texture / 0.5)  # Normalize by typical response
    
    def _analyze_periodicity(self, gray_image: np.ndarray) -> float:
        """Analyze periodic patterns (stripes, checks)"""
        # Use 2D FFT to detect periodicity
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Look for peaks in frequency domain (indicating periodicity)
        center = magnitude_spectrum.shape[0] // 2
        radius = 50  # Look at frequencies around center
        
        # Extract circular region around center
        y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        center_region = magnitude_spectrum[mask]
        
        # Calculate periodicity score based on peak prominence
        if len(center_region) > 0:
            peak_score = np.max(center_region) / np.mean(center_region + 1e-6)
            return min(1.0, peak_score / 10)  # Normalize
        
        return 0.0
    
    def _analyze_color_variation(self, image: np.ndarray) -> float:
        """Analyze color variation for complex patterns"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate standard deviation in each channel
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])
        
        # Combined variation score
        variation_score = (h_std + s_std + v_std) / (255 * 3)  # Normalize by max possible
        
        return min(1.0, variation_score * 3)  # Scale up for sensitivity
    
    def _analyze_symmetry(self, gray_image: np.ndarray) -> float:
        """Analyze symmetry for geometric patterns"""
        h, w = gray_image.shape
        
        # Horizontal symmetry
        left_half = gray_image[:, :w//2]
        right_half = cv2.flip(gray_image[:, w//2:], 1)
        
        # Make sure halves are same size
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Calculate symmetry score
        h_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
        h_symmetry = 1.0 - (h_diff / 255.0)
        
        # Vertical symmetry
        top_half = gray_image[:h//2, :]
        bottom_half = cv2.flip(gray_image[h//2:, :], 0)
        
        # Make sure halves are same size
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half = bottom_half[:min_height, :]
        
        v_diff = np.mean(np.abs(top_half.astype(float) - bottom_half.astype(float)))
        v_symmetry = 1.0 - (v_diff / 255.0)
        
        # Combined symmetry score
        return max(h_symmetry, v_symmetry)
    
    def detect_multiple_patterns(self, image_path: Union[str, Path], top_k: int = 3) -> List[PatternInfo]:
        """Detect multiple possible patterns with confidence scores"""
        primary_result = self.detect_pattern(image_path)
        
        # Get all pattern scores from evidence
        pattern_scores = []
        for pattern_name, score in primary_result.evidence.items():
            try:
                pattern_type = PatternType(pattern_name)
                pattern_info = PatternInfo(
                    pattern_type=pattern_type,
                    confidence=min(1.0, score),
                    characteristics=primary_result.characteristics,
                    evidence={pattern_name: score}
                )
                pattern_scores.append(pattern_info)
            except ValueError:
                continue
        
        # Add solid pattern if no strong patterns detected
        if primary_result.pattern_type == PatternType.SOLID:
            solid_info = PatternInfo(
                pattern_type=PatternType.SOLID,
                confidence=primary_result.confidence,
                characteristics=primary_result.characteristics,
                evidence={'solid': primary_result.confidence}
            )
            pattern_scores.append(solid_info)
        
        # Sort by confidence and return top k
        pattern_scores.sort(key=lambda x: x.confidence, reverse=True)
        return pattern_scores[:top_k]
    
    def batch_detect_patterns(self, image_dir: Union[str, Path]) -> Dict[str, PatternInfo]:
        """Batch detect patterns in directory"""
        image_dir = Path(image_dir)
        results = {}
        
        # Get image files
        image_files = list(image_dir.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
        
        print(f"Detecting patterns in {len(image_files)} images...")
        
        for image_file in image_files:
            try:
                pattern_info = self.detect_pattern(image_file)
                results[str(image_file)] = pattern_info
                
            except Exception as e:
                print(f"Error detecting pattern in {image_file}: {e}")
                results[str(image_file)] = None
        
        print(f"Completed pattern detection for {len(results)} images")
        return results
    
    def export_pattern_analysis(self, image_path: Union[str, Path], output_path: Union[str, Path]) -> Dict:
        """Export complete pattern analysis to JSON"""
        pattern_info = self.detect_pattern(image_path)
        multiple_patterns = self.detect_multiple_patterns(image_path, top_k=5)
        
        analysis = {
            'image_path': str(image_path),
            'primary_pattern': {
                'type': pattern_info.pattern_type.value,
                'confidence': pattern_info.confidence
            },
            'characteristics': pattern_info.characteristics,
            'evidence': {k: v for k, v in pattern_info.evidence.items()},
            'alternative_patterns': [
                {
                    'type': p.pattern_type.value,
                    'confidence': p.confidence
                }
                for p in multiple_patterns[1:3]  # Top 2 alternatives
            ],
            'all_patterns': [
                {
                    'type': p.pattern_type.value,
                    'confidence': p.confidence
                }
                for p in multiple_patterns
            ]
        }
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def get_pattern_statistics(self, pattern_analyses: List[Dict]) -> Dict:
        """Get statistics from pattern analyses"""
        if not pattern_analyses:
            return {}
        
        # Count pattern types
        pattern_counts = Counter()
        confidence_scores = []
        
        for analysis in pattern_analyses:
            if 'primary_pattern' in analysis:
                pattern_type = analysis['primary_pattern']['type']
                confidence = analysis['primary_pattern']['confidence']
                
                pattern_counts[pattern_type] += 1
                confidence_scores.append(confidence)
        
        return {
            'total_analyzed': len(pattern_analyses),
            'pattern_distribution': dict(pattern_counts),
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'confidence_std': np.std(confidence_scores) if confidence_scores else 0,
            'most_common_pattern': pattern_counts.most_common(1)[0] if pattern_counts else None
        }


# Test function
def test_pattern_detector():
    """Test the pattern detector"""
    print("Testing Pattern Detector...")
    
    detector = PatternDetector()
    
    # Test with sample images if available
    sample_dir = Path(__file__).parent.parent.parent / "data" / "sample_images"
    
    if sample_dir.exists():
        image_files = list(sample_dir.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
        
        if image_files:
            test_image = image_files[0]
            print(f"Testing with: {test_image}")
            
            # Test pattern detection
            pattern_info = detector.detect_pattern(test_image)
            print(f"Primary pattern: {pattern_info.pattern_type.value} (confidence: {pattern_info.confidence:.2f})")
            
            print(f"Characteristics: {pattern_info.characteristics}")
            print(f"Evidence: {pattern_info.evidence}")
            
            # Test multiple patterns
            multiple_patterns = detector.detect_multiple_patterns(test_image, top_k=3)
            print(f"Top 3 patterns:")
            for i, pattern in enumerate(multiple_patterns):
                print(f"  {i+1}. {pattern.pattern_type.value} (confidence: {pattern.confidence:.2f})")
            
            # Test export
            output_path = sample_dir / f"{test_image.stem}_pattern.json"
            analysis = detector.export_pattern_analysis(test_image, output_path)
            print(f"Analysis saved to: {output_path}")
            
            # Test batch detection
            batch_results = detector.batch_detect_patterns(sample_dir)
            print(f"Batch detected patterns for {len(batch_results)} images")
            
            # Get statistics
            stats = detector.get_pattern_statistics([analysis])
            print(f"Pattern distribution: {stats.get('pattern_distribution', {})}")
            
        else:
            print("No sample images found")
    else:
        print("Sample images directory not found")
    
    print("Pattern detector test completed!")


if __name__ == "__main__":
    test_pattern_detector()
