"""
Test suite for Task 12 - Smart Tags System
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.smart_tags.color_extractor import ColorExtractor, ColorCategory, ColorInfo
from src.smart_tags.type_classifier import TypeClassifier, ClothingType, SubType, TypeClassification
from src.smart_tags.pattern_detector import PatternDetector, PatternType, PatternInfo


class TestColorExtractor(unittest.TestCase):
    """Test color extraction system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = ColorExtractor()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test image
        self.test_image_path = self.temp_dir / "test_image.jpg"
        self._create_test_image()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_image(self):
        """Create a simple test image"""
        # Create a 200x200 image with distinct colors
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Red square
        image[0:100, 0:100] = [255, 0, 0]
        # Blue square
        image[0:100, 100:200] = [0, 0, 255]
        # Green square
        image[100:200, 0:100] = [0, 255, 0]
        # White square
        image[100:200, 100:200] = [255, 255, 255]
        
        cv2.imwrite(str(self.test_image_path), image)
    
    def test_extractor_initialization(self):
        """Test extractor initialization"""
        self.assertIsNotNone(self.extractor.color_ranges)
        self.assertIsNotNone(self.extractor.color_names)
        self.assertEqual(self.extractor.n_clusters, 5)
    
    def test_color_extraction(self):
        """Test color extraction from image"""
        colors = self.extractor.extract_colors(self.test_image_path, top_k=3)
        
        self.assertLessEqual(len(colors), 3)
        self.assertGreater(len(colors), 0)
        
        # Check color info structure
        for color in colors:
            self.assertIsInstance(color, ColorInfo)
            self.assertIsInstance(color.rgb, tuple)
            self.assertIsInstance(color.category, ColorCategory)
            self.assertGreaterEqual(color.percentage, 0.0)
            self.assertLessEqual(color.percentage, 1.0)
            self.assertGreaterEqual(color.confidence, 0.0)
            self.assertLessEqual(color.confidence, 1.0)
    
    def test_primary_color(self):
        """Test primary color detection"""
        primary = self.extractor.get_primary_color(self.test_image_path)
        
        if primary:
            self.assertIsInstance(primary, ColorInfo)
            self.assertGreaterEqual(primary.percentage, 0.0)
    
    def test_color_harmony(self):
        """Test color harmony analysis"""
        harmony = self.extractor.analyze_color_harmony(self.test_image_path)
        
        self.assertIn('harmony_score', harmony)
        self.assertIn('harmony_type', harmony)
        self.assertIn('color_count', harmony)
        
        self.assertGreaterEqual(harmony['harmony_score'], 0.0)
        self.assertLessEqual(harmony['harmony_score'], 1.0)
    
    def test_color_temperature(self):
        """Test color temperature detection"""
        temperature = self.extractor.detect_color_temperature(self.test_image_path)
        
        self.assertIn(temperature, ['warm', 'cool', 'neutral'])
    
    def test_color_categorization(self):
        """Test color categorization"""
        # Test known colors
        test_cases = [
            ([0, 0, 255], ColorCategory.BLUE),      # Blue
            ([255, 0, 0], ColorCategory.RED),       # Red
            ([0, 255, 0], ColorCategory.GREEN),     # Green
            ([0, 0, 0], ColorCategory.BLACK),       # Black
            ([255, 255, 255], ColorCategory.WHITE), # White
        ]
        
        for rgb, expected_category in test_cases:
            hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
            category = self.extractor._categorize_color(hsv)
            self.assertEqual(category, expected_category)


class TestTypeClassifier(unittest.TestCase):
    """Test clothing type classification"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.classifier = TypeClassifier()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test images with different names
        self._create_test_images()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_images(self):
        """Create test images with different names"""
        # Create simple test images
        for name in ["tshirt_blue.jpg", "jeans_black.jpg", "sneakers_white.jpg"]:
            image_path = self.temp_dir / name
            image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), image)
    
    def test_classifier_initialization(self):
        """Test classifier initialization"""
        self.assertIsNotNone(self.classifier.type_patterns)
        self.assertIsNotNone(self.classifier.aspect_ratios)
        self.assertIsNotNone(self.classifier.type_keywords)
    
    def test_filename_classification(self):
        """Test classification from filename"""
        test_cases = [
            ("tshirt_blue.jpg", ClothingType.TOP),
            ("jeans_black.jpg", ClothingType.BOTTOM),
            ("sneakers_white.jpg", ClothingType.SHOES),
            ("dress_red.jpg", ClothingType.DRESS),
            ("jacket_brown.jpg", ClothingType.OUTERWEAR),
        ]
        
        for filename, expected_type in test_cases:
            image_path = self.temp_dir / filename
            result = self.classifier.classify_from_filename(image_path)
            
            self.assertIsInstance(result, TypeClassification)
            self.assertIsInstance(result.primary_type, ClothingType)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
            
            # Check if expected type is in top alternatives
            all_types = [result.primary_type] + [alt[0] for alt in result.alternative_types]
            self.assertIn(expected_type, all_types)
    
    def test_subtype_detection(self):
        """Test subtype detection"""
        test_cases = [
            ("tshirt_blue.jpg", ClothingType.TOP, SubType.T_SHIRT),
            ("sneakers_white.jpg", ClothingType.SHOES, SubType.SNEAKERS),
            ("jeans_black.jpg", ClothingType.BOTTOM, SubType.JEANS),
        ]
        
        for filename, primary_type, expected_subtype in test_cases:
            subtype = self.classifier._determine_subtype(filename, primary_type)
            self.assertEqual(subtype, expected_subtype)
    
    def test_batch_classification(self):
        """Test batch classification"""
        results = self.classifier.batch_classify(self.temp_dir)
        
        self.assertEqual(len(results), 3)  # We created 3 test images
        
        for image_path, classification in results.items():
            self.assertIsNotNone(classification)
            self.assertIsInstance(classification, TypeClassification)
    
    def test_classification_statistics(self):
        """Test classification statistics"""
        results = self.classifier.batch_classify(self.temp_dir)
        stats = self.classifier.get_classification_statistics(results)
        
        self.assertIn('total_classified', stats)
        self.assertIn('type_distribution', stats)
        self.assertIn('avg_confidence', stats)
        
        self.assertEqual(stats['total_classified'], 3)


class TestPatternDetector(unittest.TestCase):
    """Test pattern detection system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = PatternDetector()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test images with different patterns
        self._create_pattern_images()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def _create_pattern_images(self):
        """Create test images with different patterns"""
        # Solid color image
        solid_image = np.full((200, 200, 3), [128, 128, 128], dtype=np.uint8)
        cv2.imwrite(str(self.temp_dir / "solid_gray.jpg"), solid_image)
        
        # Striped image
        striped_image = np.zeros((200, 200, 3), dtype=np.uint8)
        striped_image[:, ::20] = [255, 255, 255]  # White stripes
        cv2.imwrite(str(self.temp_dir / "striped.jpg"), striped_image)
        
        # Dots image (polka dots)
        dots_image = np.full((200, 200, 3), [255, 0, 0], dtype=np.uint8)  # Red background
        for i in range(0, 200, 40):
            for j in range(0, 200, 40):
                cv2.circle(dots_image, (i, j), 8, [255, 255, 255], -1)  # White dots
        cv2.imwrite(str(self.temp_dir / "polka_dot.jpg"), dots_image)
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertEqual(self.detector.edge_threshold, 50)
        self.assertEqual(len(self.detector.gabor_frequencies), 3)
        self.assertEqual(len(self.detector.gabor_angles), 4)
    
    def test_pattern_detection(self):
        """Test pattern detection"""
        test_images = [
            ("solid_gray.jpg", PatternType.SOLID),
            ("striped.jpg", PatternType.STRIPED),
            ("polka_dot.jpg", PatternType.POLKA_DOT),
        ]
        
        for filename, expected_pattern in test_images:
            image_path = self.temp_dir / filename
            result = self.detector.detect_pattern(image_path)
            
            self.assertIsInstance(result, PatternInfo)
            self.assertIsInstance(result.pattern_type, PatternType)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)
            self.assertIsInstance(result.characteristics, dict)
            self.assertIsInstance(result.evidence, dict)
            
            # For solid image, should detect solid pattern
            if filename == "solid_gray.jpg":
                self.assertEqual(result.pattern_type, PatternType.SOLID)
    
    def test_multiple_patterns(self):
        """Test detection of multiple patterns"""
        image_path = self.temp_dir / "striped.jpg"
        patterns = self.detector.detect_multiple_patterns(image_path, top_k=3)
        
        self.assertLessEqual(len(patterns), 3)
        self.assertGreater(len(patterns), 0)
        
        # Check that patterns are sorted by confidence
        for i in range(len(patterns) - 1):
            self.assertGreaterEqual(patterns[i].confidence, patterns[i + 1].confidence)
    
    def test_edge_analysis(self):
        """Test edge analysis"""
        image_path = self.temp_dir / "striped.jpg"
        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        edge_score = self.detector._analyze_edges(gray)
        
        self.assertGreaterEqual(edge_score, 0.0)
        self.assertLessEqual(edge_score, 1.0)
        
        # Striped image should have higher edge score than solid
        solid_path = self.temp_dir / "solid_gray.jpg"
        solid_image = cv2.imread(str(solid_path))
        solid_gray = cv2.cvtColor(solid_image, cv2.COLOR_BGR2GRAY)
        
        solid_edge_score = self.detector._analyze_edges(solid_gray)
        
        self.assertGreater(edge_score, solid_edge_score)
    
    def test_blob_analysis(self):
        """Test blob analysis"""
        image_path = self.temp_dir / "polka_dot.jpg"
        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blob_score = self.detector._analyze_blobs(gray)
        
        self.assertGreaterEqual(blob_score, 0.0)
        self.assertLessEqual(blob_score, 1.0)
    
    def test_batch_detection(self):
        """Test batch pattern detection"""
        results = self.detector.batch_detect_patterns(self.temp_dir)
        
        self.assertEqual(len(results), 3)  # We created 3 test images
        
        for image_path, pattern_info in results.items():
            self.assertIsNotNone(pattern_info)
            self.assertIsInstance(pattern_info, PatternInfo)
    
    def test_pattern_statistics(self):
        """Test pattern statistics"""
        # Create dummy analyses
        analyses = [
            {'primary_pattern': {'type': 'solid', 'confidence': 0.9}},
            {'primary_pattern': {'type': 'striped', 'confidence': 0.8}},
            {'primary_pattern': {'type': 'solid', 'confidence': 0.7}},
        ]
        
        stats = self.detector.get_pattern_statistics(analyses)
        
        self.assertIn('total_analyzed', stats)
        self.assertIn('pattern_distribution', stats)
        self.assertIn('avg_confidence', stats)
        
        self.assertEqual(stats['total_analyzed'], 3)
        self.assertEqual(stats['pattern_distribution']['solid'], 2)
        self.assertEqual(stats['pattern_distribution']['striped'], 1)


def run_integration_test():
    """Integration test for smart tags system"""
    print("Running smart tags integration test...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test images
        print("Creating test images...")
        
        # Create a complex test image
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Add different regions
        test_image[0:150, 0:150] = [255, 0, 0]  # Red region
        test_image[0:150, 150:300] = [0, 255, 0]  # Green region
        test_image[150:300, 0:150] = [0, 0, 255]  # Blue region
        test_image[150:300, 150:300] = [255, 255, 255]  # White region
        
        # Add some stripes to blue region
        test_image[150:300, 150:300:10] = [0, 0, 200]  # Darker blue stripes
        
        # Add some dots to red region
        for i in range(20, 130, 20):
            for j in range(20, 130, 20):
                cv2.circle(test_image, (i, j), 5, [255, 255, 255], -1)
        
        test_image_path = temp_dir / "complex_test.jpg"
        cv2.imwrite(str(test_image_path), test_image)
        
        # Test color extraction
        print("Testing color extraction...")
        color_extractor = ColorExtractor()
        colors = color_extractor.extract_colors(test_image_path, top_k=5)
        print(f"Extracted {len(colors)} colors")
        
        for i, color in enumerate(colors[:3]):
            print(f"  {i+1}. {color.category.value} ({color.percentage:.1%})")
        
        # Test pattern detection
        print("Testing pattern detection...")
        pattern_detector = PatternDetector()
        pattern_info = pattern_detector.detect_pattern(test_image_path)
        print(f"Detected pattern: {pattern_info.pattern_type.value} (confidence: {pattern_info.confidence:.2f})")
        
        # Test type classification
        print("Testing type classification...")
        type_classifier = TypeClassifier()
        
        # Test with different filenames
        test_filenames = [
            "tshirt_red_striped.jpg",
            "dress_blue_floral.jpg",
            "jacket_black_solid.jpg"
        ]
        
        for filename in test_filenames:
            # Copy test image with new name
            new_path = temp_dir / filename
            cv2.imwrite(str(new_path), test_image)
            
            classification = type_classifier.classify_from_filename(new_path)
            print(f"  {filename}: {classification.primary_type.value} (confidence: {classification.confidence:.2f})")
        
        # Test batch processing
        print("Testing batch processing...")
        
        # Color batch
        color_results = color_extractor.batch_extract_colors(temp_dir, temp_dir / "color_results")
        print(f"Color batch processed: {len(color_results)} images")
        
        # Pattern batch
        pattern_results = pattern_detector.batch_detect_patterns(temp_dir)
        print(f"Pattern batch processed: {len(pattern_results)} images")
        
        # Type batch
        type_results = type_classifier.batch_classify(temp_dir)
        print(f"Type batch processed: {len(type_results)} images")
        
        # Get statistics
        print("Getting statistics...")
        
        color_stats = color_extractor.get_color_statistics([color_results.get(str(p), {}) for p in temp_dir.glob("*_colors.json")])
        pattern_stats = pattern_detector.get_pattern_statistics([{'primary_pattern': {'type': p.pattern_info.pattern_type.value, 'confidence': p.pattern_info.confidence}} for p in pattern_results.values() if p])
        type_stats = type_classifier.get_classification_statistics(type_results)
        
        print(f"Color stats: {color_stats}")
        print(f"Pattern stats: {pattern_stats}")
        print(f"Type stats: {type_stats}")
        
        print("Smart tags integration test completed successfully!")
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        raise
    
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run integration test
    run_integration_test()
