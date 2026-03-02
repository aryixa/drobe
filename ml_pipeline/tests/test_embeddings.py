"""
Test suite for Task 9 - Image Embeddings
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.embeddings.resnet_extractor import ResNetEmbeddingExtractor
from src.embeddings.embedding_storage import EmbeddingStorage
from src.utils.image_processing import get_image_metadata, validate_image


class TestResNetExtractor(unittest.TestCase):
    """Test ResNet embedding extractor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = ResNetEmbeddingExtractor()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create dummy image file
        from PIL import Image
        dummy_image = Image.new('RGB', (224, 224), color='red')
        self.test_image_path = self.temp_dir / "test_image.jpg"
        dummy_image.save(self.test_image_path)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.extractor.model)
        self.assertEqual(self.extractor.embedding_dim, 512)
        self.assertIsNotNone(self.extractor.transforms)
    
    def test_single_embedding_extraction(self):
        """Test single image embedding extraction"""
        embedding = self.extractor.extract_embedding(self.test_image_path)
        
        self.assertEqual(embedding.shape[0], 512)
        self.assertEqual(embedding.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(embedding)))
    
    def test_embedding_stats(self):
        """Test embedding statistics"""
        embedding = self.extractor.extract_embedding(self.test_image_path)
        embeddings = {str(self.test_image_path): embedding}
        
        stats = self.extractor.get_embedding_stats(embeddings)
        
        self.assertEqual(stats['num_embeddings'], 1)
        self.assertEqual(stats['embedding_dim'], 512)
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
    
    def test_invalid_image_path(self):
        """Test handling of invalid image path"""
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_embedding("nonexistent.jpg")
    
    def test_unsupported_format(self):
        """Test handling of unsupported file format"""
        # Create a text file
        text_file = self.temp_dir / "test.txt"
        text_file.write_text("not an image")
        
        with self.assertRaises(ValueError):
            self.extractor.extract_embedding(text_file)


class TestEmbeddingStorage(unittest.TestCase):
    """Test embedding storage system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test.db"
        self.storage = EmbeddingStorage(self.db_path)
        
        # Create dummy embedding
        self.test_embedding = np.random.rand(512).astype(np.float32)
        self.test_metadata = {
            'file_size': 1024,
            'width': 224,
            'height': 224
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """Test database table creation"""
        self.assertTrue(self.db_path.exists())
        
        # Check if tables exist
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            self.assertIn('embeddings', tables)
            self.assertIn('image_tags', tables)
    
    def test_store_and_load_embedding(self):
        """Test storing and loading embeddings"""
        image_path = "test_image.jpg"
        
        # Store embedding
        success = self.storage.store_embedding(image_path, self.test_embedding, self.test_metadata)
        self.assertTrue(success)
        
        # Load embedding
        loaded_embedding = self.storage.load_embedding(image_path)
        self.assertIsNotNone(loaded_embedding)
        np.testing.assert_array_equal(loaded_embedding, self.test_embedding)
    
    def test_duplicate_prevention(self):
        """Test prevention of duplicate embeddings"""
        image_path = "test_image.jpg"
        
        # Store first embedding
        success1 = self.storage.store_embedding(image_path, self.test_embedding, self.test_metadata)
        self.assertTrue(success1)
        
        # Try to store duplicate
        success2 = self.storage.store_embedding(image_path, self.test_embedding, self.test_metadata)
        self.assertFalse(success2)
    
    def test_tags_system(self):
        """Test tags storage and retrieval"""
        image_path = "test_image.jpg"
        
        # Store embedding first
        self.storage.store_embedding(image_path, self.test_embedding, self.test_metadata)
        
        # Add tags
        tags = {
            'color': {'blue': 0.8, 'dark': 0.6},
            'type': {'shirt': 0.9}
        }
        self.storage.add_tags(image_path, tags)
        
        # Retrieve tags
        retrieved_tags = self.storage.get_tags(image_path)
        
        self.assertEqual(retrieved_tags['color']['blue'], 0.8)
        self.assertEqual(retrieved_tags['type']['shirt'], 0.9)
    
    def test_search_by_tags(self):
        """Test searching images by tags"""
        image_path1 = "test1.jpg"
        image_path2 = "test2.jpg"
        
        # Store embeddings
        self.storage.store_embedding(image_path1, self.test_embedding, self.test_metadata)
        self.storage.store_embedding(image_path2, self.test_embedding, self.test_metadata)
        
        # Add tags
        self.storage.add_tags(image_path1, {'color': {'blue': 0.8}})
        self.storage.add_tags(image_path2, {'color': {'red': 0.8}})
        
        # Search by color
        results = self.storage.search_by_tags({'color': ['blue']})
        self.assertIn(image_path1, results)
        self.assertNotIn(image_path2, results)
    
    def test_statistics(self):
        """Test storage statistics"""
        image_path = "test_image.jpg"
        
        # Store embedding
        self.storage.store_embedding(image_path, self.test_embedding, self.test_metadata)
        
        # Get stats
        stats = self.storage.get_statistics()
        
        self.assertEqual(stats['total_embeddings'], 1)
        self.assertEqual(stats['average_embedding_dim'], 512)
        self.assertGreater(stats['database_size'], 0)


class TestImageProcessing(unittest.TestCase):
    """Test image processing utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test image
        from PIL import Image
        self.test_image = Image.new('RGB', (224, 224), color='blue')
        self.test_image_path = self.temp_dir / "test_image.jpg"
        self.test_image.save(self.test_image_path)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_metadata_extraction(self):
        """Test metadata extraction"""
        metadata = get_image_metadata(self.test_image_path)
        
        self.assertIn('file_size', metadata)
        self.assertIn('width', metadata)
        self.assertIn('height', metadata)
        self.assertEqual(metadata['width'], 224)
        self.assertEqual(metadata['height'], 224)
    
    def test_image_validation(self):
        """Test image validation"""
        is_valid, error = validate_image(self.test_image_path)
        
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
    
    def test_invalid_image_validation(self):
        """Test validation of invalid image"""
        # Create invalid file
        invalid_file = self.temp_dir / "invalid.txt"
        invalid_file.write_text("not an image")
        
        is_valid, error = validate_image(invalid_file)
        
        self.assertFalse(is_valid)
        self.assertIn("Invalid image file", error)


def run_integration_test():
    """Integration test for the complete embedding pipeline"""
    print("Running integration test...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test image
        from PIL import Image
        test_image = Image.new('RGB', (224, 224), color='green')
        image_path = temp_dir / "integration_test.jpg"
        test_image.save(image_path)
        
        # Initialize components
        extractor = ResNetEmbeddingExtractor()
        storage = EmbeddingStorage(temp_dir / "integration.db")
        
        # Extract embedding
        embedding = extractor.extract_embedding(image_path)
        print(f"Extracted embedding shape: {embedding.shape}")
        
        # Store embedding
        metadata = get_image_metadata(image_path)
        success = storage.store_embedding(str(image_path), embedding, metadata)
        print(f"Storage success: {success}")
        
        # Load embedding
        loaded_embedding = storage.load_embedding(str(image_path))
        print(f"Loaded embedding shape: {loaded_embedding.shape}")
        
        # Verify embeddings match
        np.testing.assert_array_equal(embedding, loaded_embedding)
        print("Embeddings match!")
        
        # Add tags
        tags = {'color': {'green': 0.9}, 'type': {'test': 1.0}}
        storage.add_tags(str(image_path), tags)
        
        # Retrieve tags
        retrieved_tags = storage.get_tags(str(image_path))
        print(f"Retrieved tags: {retrieved_tags}")
        
        # Get statistics
        stats = storage.get_statistics()
        print(f"Storage stats: {stats}")
        
        print("Integration test completed successfully!")
        
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
