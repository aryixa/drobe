"""
Task 12 - Smart Tags System: Type Classification
Classify clothing items using labels and simple logic
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict, Counter
import re

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.image_processing import enhance_image_quality


class ClothingType(Enum):
    """Clothing type categories"""
    TOP = "top"
    BOTTOM = "bottom"
    SHOES = "shoes"
    DRESS = "dress"
    OUTERWEAR = "outerwear"
    ACCESSORY = "accessory"
    UNDERWEAR = "underwear"
    SWIMWEAR = "swimwear"
    SPORTSWEAR = "sportswear"


class SubType(Enum):
    """Clothing subtypes for more specific classification"""
    # Tops
    T_SHIRT = "t-shirt"
    SHIRT = "shirt"
    BLOUSE = "blouse"
    TANK_TOP = "tank_top"
    SWEATER = "sweater"
    HOODIE = "hoodie"
    CROP_TOP = "crop_top"
    POLO = "polo"
    
    # Bottoms
    PANTS = "pants"
    JEANS = "jeans"
    SHORTS = "shorts"
    SKIRT = "skirt"
    LEGGINGS = "leggings"
    TROUSERS = "trousers"
    
    # Shoes
    SNEAKERS = "sneakers"
    BOOTS = "boots"
    SANDALS = "sandals"
    HEELS = "heels"
    FLATS = "flats"
    LOAFERS = "loafers"
    
    # Dresses
    CASUAL_DRESS = "casual_dress"
    FORMAL_DRESS = "formal_dress"
    MAXI_DRESS = "maxi_dress"
    MINI_DRESS = "mini_dress"
    
    # Outerwear
    JACKET = "jacket"
    COAT = "coat"
    BLAZER = "blazer"
    CARDIGAN = "cardigan"
    
    # Accessories
    BAG = "bag"
    HAT = "hat"
    BELT = "belt"
    SCARF = "scarf"
    GLOVES = "gloves"


@dataclass
class TypeClassification:
    """Type classification result"""
    primary_type: ClothingType
    sub_type: Optional[SubType]
    confidence: float
    evidence: Dict[str, float]
    alternative_types: List[Tuple[ClothingType, float]]


class TypeClassifier:
    """
    Clothing type classifier using filename analysis and simple heuristics
    - Filename pattern matching
    - Aspect ratio analysis
    - Simple visual features
    """
    
    def __init__(self):
        # Type patterns for filename analysis
        self.type_patterns = self._init_type_patterns()
        
        # Aspect ratio ranges for different clothing types
        self.aspect_ratios = self._init_aspect_ratios()
        
        # Keywords for type detection
        self.type_keywords = self._init_type_keywords()
        
        print("Type Classifier initialized")
    
    def _init_type_patterns(self) -> Dict[ClothingType, List[str]]:
        """Initialize regex patterns for filename analysis"""
        return {
            ClothingType.TOP: [
                r'.*shirt.*', r'.*t.?shirt.*', r'.*blouse.*', r'.*top.*',
                r'.*tank.*', r'.*sweater.*', r'.*hoodie.*', r'.*crop.*',
                r'.*polo.*', r'.*tee.*'
            ],
            ClothingType.BOTTOM: [
                r'.*pant.*', r'.*jean.*', r'.*trouser.*', r'.*short.*',
                r'.*skirt.*', r'.*legging.*', r'.*bottom.*'
            ],
            ClothingType.SHOES: [
                r'.*shoe.*', r'.*sneaker.*', r'.*boot.*', r'.*sandal.*',
                r'.*heel.*', r'.*flat.*', r'.*loafer.*', r'.*footwear.*'
            ],
            ClothingType.DRESS: [
                r'.*dress.*', r'.*gown.*', r'.*robe.*'
            ],
            ClothingType.OUTERWEAR: [
                r'.*jacket.*', r'.*coat.*', r'.*blazer.*', r'.*cardigan.*',
                r'.*outer.*', r'.*windbreaker.*', r'.*parka.*'
            ],
            ClothingType.ACCESSORY: [
                r'.*bag.*', r'.*purse.*', r'.*hat.*', r'.*cap.*',
                r'.*belt.*', r'.*scarf.*', r'.*glove.*', r'.wallet.*'
            ],
            ClothingType.UNDERWEAR: [
                r'.*underwear.*', r'.*bra.*', r'.*panty.*', r'.*brief.*'
            ],
            ClothingType.SWIMWEAR: [
                r'.*swim.*', r'.*bikini.*', r'.*swimsuit.*', r'.*trunk.*'
            ],
            ClothingType.SPORTSWEAR: [
                r'.*sport.*', r'.*athletic.*', r'.*gym.*', r'.*workout.*'
            ]
        }
    
    def _init_aspect_ratios(self) -> Dict[ClothingType, Tuple[float, float]]:
        """Initialize typical aspect ratio ranges (width/height)"""
        return {
            ClothingType.TOP: (0.6, 1.2),      # Can be wide or tall
            ClothingType.BOTTOM: (0.4, 0.8),    # Typically wider than tall
            ClothingType.SHOES: (1.2, 3.0),     # Typically wider
            ClothingType.DRESS: (0.4, 0.7),     # Typically taller than wide
            ClothingType.OUTERWEAR: (0.6, 1.0),  # Similar to tops
            ClothingType.ACCESSORY: (0.8, 2.0),  # Variable
            ClothingType.UNDERWEAR: (0.5, 1.5),  # Variable
            ClothingType.SWIMWEAR: (0.5, 1.2),   # Variable
            ClothingType.SPORTSWEAR: (0.6, 1.5)  # Variable
        }
    
    def _init_type_keywords(self) -> Dict[str, ClothingType]:
        """Initialize keyword mappings"""
        return {
            # Tops
            'tshirt': ClothingType.TOP, 'tee': ClothingType.TOP, 'shirt': ClothingType.TOP,
            'blouse': ClothingType.TOP, 'top': ClothingType.TOP, 'tank': ClothingType.TOP,
            'sweater': ClothingType.TOP, 'hoodie': ClothingType.TOP, 'crop': ClothingType.TOP,
            'polo': ClothingType.TOP, 'pullover': ClothingType.TOP,
            
            # Bottoms
            'pant': ClothingType.BOTTOM, 'jean': ClothingType.BOTTOM, 'trouser': ClothingType.BOTTOM,
            'short': ClothingType.BOTTOM, 'skirt': ClothingType.BOTTOM, 'legging': ClothingType.BOTTOM,
            'bottom': ClothingType.BOTTOM, 'slack': ClothingType.BOTTOM,
            
            # Shoes
            'shoe': ClothingType.SHOES, 'sneaker': ClothingType.SHOES, 'boot': ClothingType.SHOES,
            'sandal': ClothingType.SHOES, 'heel': ClothingType.SHOES, 'flat': ClothingType.SHOES,
            'loafer': ClothingType.SHOES, 'footwear': ClothingType.SHOES,
            
            # Dresses
            'dress': ClothingType.DRESS, 'gown': ClothingType.DRESS, 'robe': ClothingType.DRESS,
            
            # Outerwear
            'jacket': ClothingType.OUTERWEAR, 'coat': ClothingType.OUTERWEAR, 'blazer': ClothingType.OUTERWEAR,
            'cardigan': ClothingType.OUTERWEAR, 'outer': ClothingType.OUTERWEAR, 'windbreaker': ClothingType.OUTERWEAR,
            
            # Accessories
            'bag': ClothingType.ACCESSORY, 'purse': ClothingType.ACCESSORY, 'hat': ClothingType.ACCESSORY,
            'cap': ClothingType.ACCESSORY, 'belt': ClothingType.ACCESSORY, 'scarf': ClothingType.ACCESSORY,
            'glove': ClothingType.ACCESSORY, 'wallet': ClothingType.ACCESSORY,
            
            # Other
            'underwear': ClothingType.UNDERWEAR, 'bra': ClothingType.UNDERWEAR,
            'swim': ClothingType.SWIMWEAR, 'bikini': ClothingType.SWIMWEAR,
            'sport': ClothingType.SPORTSWEAR, 'athletic': ClothingType.SPORTSWEAR
        }
    
    def classify_from_filename(self, image_path: Union[str, Path]) -> TypeClassification:
        """
        Classify clothing type from filename patterns
        
        Args:
            image_path: Path to image file
            
        Returns:
            Type classification result
        """
        image_path = Path(image_path)
        filename = image_path.stem.lower()
        
        # Score each type based on pattern matching
        type_scores = defaultdict(float)
        
        for clothing_type, patterns in self.type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    type_scores[clothing_type] += 1.0
        
        # Also check keywords
        for keyword, clothing_type in self.type_keywords.items():
            if keyword in filename:
                type_scores[clothing_type] += 0.5
        
        # Normalize scores
        total_score = sum(type_scores.values())
        if total_score > 0:
            for clothing_type in type_scores:
                type_scores[clothing_type] /= total_score
        
        # Get primary type and alternatives
        if type_scores:
            sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
            primary_type, primary_score = sorted_types[0]
            alternatives = sorted_types[1:3]  # Top 2 alternatives
        else:
            # Default classification if no patterns match
            primary_type = ClothingType.TOP
            primary_score = 0.3
            alternatives = []
        
        # Determine subtype
        sub_type = self._determine_subtype(filename, primary_type)
        
        # Calculate confidence
        confidence = min(1.0, primary_score + 0.2)  # Boost confidence slightly
        
        return TypeClassification(
            primary_type=primary_type,
            sub_type=sub_type,
            confidence=confidence,
            evidence=dict(type_scores),
            alternative_types=alternatives
        )
    
    def classify_from_image_features(self, image_path: Union[str, Path]) -> TypeClassification:
        """
        Classify clothing type from basic image features
        
        Args:
            image_path: Path to image file
            
        Returns:
            Type classification result
        """
        image_path = Path(image_path)
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Get basic features
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # Score types based on aspect ratio
            type_scores = defaultdict(float)
            
            for clothing_type, (min_ratio, max_ratio) in self.aspect_ratios.items():
                if min_ratio <= aspect_ratio <= max_ratio:
                    # Score based on how well the ratio fits
                    center_ratio = (min_ratio + max_ratio) / 2
                    distance = abs(aspect_ratio - center_ratio) / (max_ratio - min_ratio)
                    score = max(0.0, 1.0 - distance)
                    type_scores[clothing_type] = score
            
            # Add filename-based evidence
            filename_classification = self.classify_from_filename(image_path)
            
            # Combine evidence (70% image features, 30% filename)
            for clothing_type, score in filename_classification.evidence.items():
                type_scores[clothing_type] = type_scores.get(clothing_type, 0) * 0.7 + score * 0.3
            
            # Normalize and get result
            total_score = sum(type_scores.values())
            if total_score > 0:
                for clothing_type in type_scores:
                    type_scores[clothing_type] /= total_score
            
            if type_scores:
                sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
                primary_type, primary_score = sorted_types[0]
                alternatives = sorted_types[1:3]
            else:
                primary_type = ClothingType.TOP
                primary_score = 0.3
                alternatives = []
            
            sub_type = self._determine_subtype(image_path.stem.lower(), primary_type)
            confidence = min(1.0, primary_score)
            
            return TypeClassification(
                primary_type=primary_type,
                sub_type=sub_type,
                confidence=confidence,
                evidence=dict(type_scores),
                alternative_types=alternatives
            )
            
        except Exception as e:
            print(f"Error classifying {image_path}: {e}")
            # Fallback to filename classification
            return self.classify_from_filename(image_path)
    
    def _determine_subtype(self, filename: str, primary_type: ClothingType) -> Optional[SubType]:
        """Determine subtype based on filename and primary type"""
        subtype_patterns = {
            SubType.T_SHIRT: [r'.*t.?shirt.*', r'.*tee.*'],
            SubType.SHIRT: [r'.*shirt.*', r'.*button.*'],
            SubType.BLOUSE: [r'.*blouse.*'],
            SubType.TANK_TOP: [r'.*tank.*'],
            SubType.SWEATER: [r'.*sweater.*', r'.*pullover.*'],
            SubType.HOODIE: [r'.*hoodie.*'],
            SubType.CROP_TOP: [r'.*crop.*'],
            SubType.POLO: [r'.*polo.*'],
            
            SubType.PANTS: [r'.*pant.*', r'.*trouser.*'],
            SubType.JEANS: [r'.*jean.*'],
            SubType.SHORTS: [r'.*short.*'],
            SubType.SKIRT: [r'.*skirt.*'],
            SubType.LEGGINGS: [r'.*legging.*'],
            
            SubType.SNEAKERS: [r'.*sneaker.*'],
            SubType.BOOTS: [r'.*boot.*'],
            SubType.SANDALS: [r'.*sandal.*'],
            SubType.HEELS: [r'.*heel.*'],
            SubType.FLATS: [r'.*flat.*'],
            SubType.LOAFERS: [r'.*loafer.*'],
            
            SubType.CASUAL_DRESS: [r'.*casual.*dress.*', r'.*summer.*dress.*'],
            SubType.FORMAL_DRESS: [r'.*formal.*dress.*', r'.*evening.*dress.*'],
            SubType.MAXI_DRESS: [r'.*maxi.*dress.*'],
            SubType.MINI_DRESS: [r'.*mini.*dress.*'],
            
            SubType.JACKET: [r'.*jacket.*'],
            SubType.COAT: [r'.*coat.*'],
            SubType.BLAZER: [r'.*blazer.*'],
            SubType.CARDIGAN: [r'.*cardigan.*'],
        }
        
        # Filter patterns by primary type
        relevant_patterns = {}
        for subtype, patterns in subtype_patterns.items():
            if self._is_subtype_compatible(subtype, primary_type):
                relevant_patterns[subtype] = patterns
        
        # Check patterns
        for subtype, patterns in relevant_patterns.items():
            for pattern in patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    return subtype
        
        return None
    
    def _is_subtype_compatible(self, subtype: SubType, primary_type: ClothingType) -> bool:
        """Check if subtype is compatible with primary type"""
        compatibility = {
            # Tops
            SubType.T_SHIRT: [ClothingType.TOP],
            SubType.SHIRT: [ClothingType.TOP],
            SubType.BLOUSE: [ClothingType.TOP],
            SubType.TANK_TOP: [ClothingType.TOP],
            SubType.SWEATER: [ClothingType.TOP],
            SubType.HOODIE: [ClothingType.TOP],
            SubType.CROP_TOP: [ClothingType.TOP],
            SubType.POLO: [ClothingType.TOP],
            
            # Bottoms
            SubType.PANTS: [ClothingType.BOTTOM],
            SubType.JEANS: [ClothingType.BOTTOM],
            SubType.SHORTS: [ClothingType.BOTTOM],
            SubType.SKIRT: [ClothingType.BOTTOM],
            SubType.LEGGINGS: [ClothingType.BOTTOM],
            
            # Shoes
            SubType.SNEAKERS: [ClothingType.SHOES],
            SubType.BOOTS: [ClothingType.SHOES],
            SubType.SANDALS: [ClothingType.SHOES],
            SubType.HEELS: [ClothingType.SHOES],
            SubType.FLATS: [ClothingType.SHOES],
            SubType.LOAFERS: [ClothingType.SHOES],
            
            # Dresses
            SubType.CASUAL_DRESS: [ClothingType.DRESS],
            SubType.FORMAL_DRESS: [ClothingType.DRESS],
            SubType.MAXI_DRESS: [ClothingType.DRESS],
            SubType.MINI_DRESS: [ClothingType.DRESS],
            
            # Outerwear
            SubType.JACKET: [ClothingType.OUTERWEAR],
            SubType.COAT: [ClothingType.OUTERWEAR],
            SubType.BLAZER: [ClothingType.OUTERWEAR],
            SubType.CARDIGAN: [ClothingType.OUTERWEAR],
        }
        
        return primary_type in compatibility.get(subtype, [])
    
    def batch_classify(self, image_dir: Union[str, Path], use_image_features: bool = False) -> Dict[str, TypeClassification]:
        """
        Batch classify images in directory
        
        Args:
            image_dir: Directory containing images
            use_image_features: Whether to use image features (slower but more accurate)
            
        Returns:
            Dictionary mapping image paths to classifications
        """
        image_dir = Path(image_dir)
        results = {}
        
        # Get image files
        image_files = list(image_dir.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
        
        print(f"Classifying {len(image_files)} images...")
        
        for image_file in image_files:
            try:
                if use_image_features:
                    classification = self.classify_from_image_features(image_file)
                else:
                    classification = self.classify_from_filename(image_file)
                
                results[str(image_file)] = classification
                
            except Exception as e:
                print(f"Error classifying {image_file}: {e}")
                results[str(image_file)] = None
        
        print(f"Completed classification for {len(results)} images")
        return results
    
    def export_classifications(self, classifications: Dict[str, TypeClassification], output_path: Union[str, Path]):
        """Export classifications to JSON"""
        output_data = {}
        
        for image_path, classification in classifications.items():
            if classification:
                output_data[image_path] = {
                    'primary_type': classification.primary_type.value,
                    'sub_type': classification.sub_type.value if classification.sub_type else None,
                    'confidence': classification.confidence,
                    'evidence': {k.value: v for k, v in classification.evidence.items()},
                    'alternatives': [(k.value, v) for k, v in classification.alternative_types]
                }
            else:
                output_data[image_path] = {'error': 'Classification failed'}
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Classifications exported to: {output_path}")
    
    def get_classification_statistics(self, classifications: Dict[str, TypeClassification]) -> Dict:
        """Get statistics from classifications"""
        if not classifications:
            return {}
        
        # Count primary types
        type_counts = Counter()
        subtype_counts = Counter()
        confidence_scores = []
        
        for classification in classifications.values():
            if classification:
                type_counts[classification.primary_type.value] += 1
                if classification.sub_type:
                    subtype_counts[classification.sub_type.value] += 1
                confidence_scores.append(classification.confidence)
        
        return {
            'total_classified': len(classifications),
            'type_distribution': dict(type_counts),
            'subtype_distribution': dict(subtype_counts),
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'confidence_std': np.std(confidence_scores) if confidence_scores else 0,
            'most_common_type': type_counts.most_common(1)[0] if type_counts else None
        }


# Test function
def test_type_classifier():
    """Test the type classifier"""
    print("Testing Type Classifier...")
    
    classifier = TypeClassifier()
    
    # Test with sample images if available
    sample_dir = Path(__file__).parent.parent.parent / "data" / "sample_images"
    
    if sample_dir.exists():
        image_files = list(sample_dir.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
        
        if image_files:
            test_image = image_files[0]
            print(f"Testing with: {test_image}")
            
            # Test filename classification
            filename_result = classifier.classify_from_filename(test_image)
            print(f"Filename classification: {filename_result.primary_type.value} (confidence: {filename_result.confidence:.2f})")
            
            # Test image feature classification
            image_result = classifier.classify_from_image_features(test_image)
            print(f"Image feature classification: {image_result.primary_type.value} (confidence: {image_result.confidence:.2f})")
            
            if image_result.sub_type:
                print(f"Subtype: {image_result.sub_type.value}")
            
            # Test batch classification
            batch_results = classifier.batch_classify(sample_dir, use_image_features=False)
            print(f"Batch classified {len(batch_results)} images")
            
            # Get statistics
            stats = classifier.get_classification_statistics(batch_results)
            print(f"Type distribution: {stats.get('type_distribution', {})}")
            print(f"Average confidence: {stats.get('avg_confidence', 0):.2f}")
            
            # Export results
            output_path = sample_dir / "type_classifications.json"
            classifier.export_classifications(batch_results, output_path)
            
        else:
            print("No sample images found")
    else:
        print("Sample images directory not found")
    
    print("Type classifier test completed!")


if __name__ == "__main__":
    test_type_classifier()
