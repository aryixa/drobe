"""
Test suite for Task 10 - Similarity System
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.similarity.cosine_sim import CosineSimilarityEngine
from src.similarity.nearest_neighbors import NearestNeighborSearch, SearchMethod
from src.embeddings.embedding_storage import EmbeddingStorage
from sklearn.preprocessing import normalize


class TestCosineSimilarityEngine(unittest.TestCase):
    """Test cosine similarity engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = EmbeddingStorage(self.temp_dir / "test.db")
        
        # Create test embeddings
        np.random.seed(42)
        self.test_embeddings = np.random.rand(10, 512)
        self.test_embeddings = normalize(self.test_embeddings, norm='l2')
        
        # Store test embeddings
        for i, embedding in enumerate(self.test_embeddings):
            self.storage.store_embedding(f"test_{i}.jpg", embedding)
        
        self.engine = CosineSimilarityEngine(self.storage)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        self.assertEqual(len(self.engine.image_paths), 10)
        self.assertEqual(self.engine.embedding_dim, 512)
        self.assertIsNotNone(self.engine.embeddings_matrix)
    
    def test_cosine_similarity_computation(self):
        """Test cosine similarity computation"""
        # Test similar vectors
        similarity = self.engine.compute_cosine_similarity(
            self.test_embeddings[0], 
            self.test_embeddings[0]
        )
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        # Test different vectors
        similarity = self.engine.compute_cosine_similarity(
            self.test_embeddings[0], 
            self.test_embeddings[1]
        )
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_find_similar_items(self):
        """Test finding similar items"""
        query_embedding = self.test_embeddings[0]
        similar_items = self.engine.find_similar_items(query_embedding, top_k=5)
        
        self.assertLessEqual(len(similar_items), 5)
        
        # Check that results are sorted by similarity
        if len(similar_items) > 1:
            for i in range(len(similar_items) - 1):
                self.assertGreaterEqual(
                    similar_items[i]['similarity'], 
                    similar_items[i + 1]['similarity']
                )
        
        # Check result structure
        for item in similar_items:
            self.assertIn('image_path', item)
            self.assertIn('similarity', item)
            self.assertIn('index', item)
            self.assertGreaterEqual(item['similarity'], 0.0)
            self.assertLessEqual(item['similarity'], 1.0)
    
    def test_find_similar_by_path(self):
        """Test finding similar items by path"""
        similar_items = self.engine.find_similar_by_path("test_0.jpg", top_k=3)
        
        self.assertLessEqual(len(similar_items), 3)
        
        # Should not include the query item itself
        paths = [item['image_path'] for item in similar_items]
        self.assertNotIn("test_0.jpg", paths)
    
    def test_similarity_threshold(self):
        """Test similarity threshold filtering"""
        query_embedding = self.test_embeddings[0]
        
        # Test with high threshold
        high_threshold = 0.99
        results_high = self.engine.find_similar_items(
            query_embedding, 
            threshold=high_threshold
        )
        
        # Test with low threshold
        low_threshold = 0.0
        results_low = self.engine.find_similar_items(
            query_embedding, 
            threshold=low_threshold
        )
        
        # High threshold should return fewer or equal results
        self.assertLessEqual(len(results_high), len(results_low))
    
    def test_similarity_matrix(self):
        """Test similarity matrix computation"""
        sim_matrix, paths = self.engine.compute_similarity_matrix()
        
        self.assertEqual(sim_matrix.shape, (10, 10))
        self.assertEqual(len(paths), 10)
        
        # Check diagonal is 1.0 (self-similarity)
        np.testing.assert_array_almost_equal(np.diag(sim_matrix), 1.0)
        
        # Check symmetry
        np.testing.assert_array_almost_equal(sim_matrix, sim_matrix.T)
    
    def test_outfit_suggestions(self):
        """Test outfit suggestions"""
        item_paths = ["test_0.jpg", "test_1.jpg"]
        suggestions = self.engine.get_outfit_suggestions(item_paths, top_k=3)
        
        self.assertIn("test_0.jpg", suggestions)
        self.assertIn("test_1.jpg", suggestions)
        
        # Check that suggestions don't include original items
        for item_path, suggestion_list in suggestions.items():
            suggested_paths = [s['image_path'] for s in suggestion_list]
            self.assertNotIn(item_path, suggested_paths)
    
    def test_rank_by_similarity(self):
        """Test ranking by similarity"""
        candidates = ["test_1.jpg", "test_2.jpg", "test_3.jpg"]
        reference = "test_0.jpg"
        
        ranked = self.engine.rank_by_similarity(candidates, reference)
        
        self.assertEqual(len(ranked), 3)
        
        # Check that results are sorted by similarity
        if len(ranked) > 1:
            for i in range(len(ranked) - 1):
                self.assertGreaterEqual(
                    ranked[i]['similarity'], 
                    ranked[i + 1]['similarity']
                )
    
    def test_statistics(self):
        """Test engine statistics"""
        stats = self.engine.get_statistics()
        
        self.assertIn('total_embeddings', stats)
        self.assertIn('embedding_dim', stats)
        self.assertIn('similarity_threshold', stats)
        self.assertEqual(stats['total_embeddings'], 10)
        self.assertEqual(stats['embedding_dim'], 512)


class TestNearestNeighborSearch(unittest.TestCase):
    """Test nearest neighbor search"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = EmbeddingStorage(self.temp_dir / "test.db")
        
        # Create test embeddings
        np.random.seed(42)
        self.test_embeddings = np.random.rand(20, 512)
        self.test_embeddings = normalize(self.test_embeddings, norm='l2')
        
        # Store test embeddings
        for i, embedding in enumerate(self.test_embeddings):
            self.storage.store_embedding(f"test_{i}.jpg", embedding)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_brute_force_search(self):
        """Test brute-force search"""
        nn_search = NearestNeighborSearch(self.storage, SearchMethod.BRUTE_FORCE)
        
        query = self.test_embeddings[0]
        results = nn_search.search(query, k=5)
        
        self.assertLessEqual(len(results), 5)
        
        # Check result structure
        for result in results:
            self.assertIsInstance(result.image_path, str)
            self.assertIsInstance(result.similarity, float)
            self.assertIsInstance(result.index, int)
            self.assertGreaterEqual(result.similarity, 0.0)
            self.assertLessEqual(result.similarity, 1.0)
    
    def test_faiss_search(self):
        """Test FAISS search"""
        nn_search = NearestNeighborSearch(self.storage, SearchMethod.FAISS)
        
        query = self.test_embeddings[0]
        results = nn_search.search(query, k=5)
        
        self.assertLessEqual(len(results), 5)
        
        # Check that results are sorted by similarity
        if len(results) > 1:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(
                    results[i].similarity, 
                    results[i + 1].similarity
                )
    
    def test_batch_search(self):
        """Test batch search"""
        nn_search = NearestNeighborSearch(self.storage, SearchMethod.FAISS)
        
        queries = self.test_embeddings[:3]
        batch_results = nn_search.batch_search(queries, k=3)
        
        self.assertEqual(len(batch_results), 3)
        
        for results in batch_results:
            self.assertLessEqual(len(results), 3)
    
    def test_range_search(self):
        """Test range search"""
        nn_search = NearestNeighborSearch(self.storage, SearchMethod.FAISS)
        
        query = self.test_embeddings[0]
        results = nn_search.range_search(query, radius=0.5)
        
        # All results should have similarity >= radius
        for result in results:
            self.assertGreaterEqual(result.similarity, 0.5)
    
    def test_performance_stats(self):
        """Test performance statistics"""
        nn_search = NearestNeighborSearch(self.storage, SearchMethod.FAISS)
        
        # Perform some searches to generate stats
        for i in range(3):
            query = self.test_embeddings[i]
            nn_search.search(query, k=5)
        
        stats = nn_search.get_performance_stats()
        
        self.assertIn('total_searches', stats)
        self.assertIn('avg_search_time', stats)
        self.assertIn('method', stats)
        self.assertEqual(stats['total_searches'], 3)
        self.assertEqual(stats['method'], 'faiss')
    
    def test_method_comparison(self):
        """Test comparison of different methods"""
        nn_search = NearestNeighborSearch(self.storage, SearchMethod.BRUTE_FORCE)
        
        query = self.test_embeddings[0]
        comparison = nn_search.compare_methods(query, k=5)
        
        self.assertIn('brute_force', comparison)
        self.assertIn('faiss', comparison)
        
        for method, stats in comparison.items():
            self.assertIn('search_time', stats)
            self.assertIn('results_count', stats)
            self.assertIn('top_similarity', stats)


class TestSearchResult(unittest.TestCase):
    """Test search result data structure"""
    
    def test_search_result_creation(self):
        """Test search result creation"""
        from src.similarity.nearest_neighbors import SearchResult
        
        result = SearchResult(
            image_path="test.jpg",
            similarity=0.85,
            index=5,
            distance=0.15
        )
        
        self.assertEqual(result.image_path, "test.jpg")
        self.assertEqual(result.similarity, 0.85)
        self.assertEqual(result.index, 5)
        self.assertEqual(result.distance, 0.15)


def run_integration_test():
    """Integration test for similarity system"""
    print("Running similarity system integration test...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test data
        np.random.seed(42)
        test_embeddings = np.random.rand(15, 512)
        test_embeddings = normalize(test_embeddings, norm='l2')
        
        # Set up storage
        storage = EmbeddingStorage(temp_dir / "integration.db")
        
        # Store embeddings
        for i, embedding in enumerate(test_embeddings):
            storage.store_embedding(f"test_{i}.jpg", embedding)
        
        # Test cosine similarity engine
        print("Testing cosine similarity engine...")
        engine = CosineSimilarityEngine(storage)
        
        # Test similarity computation
        similarity = engine.compute_cosine_similarity(test_embeddings[0], test_embeddings[1])
        print(f"Similarity between test_0 and test_1: {similarity:.4f}")
        
        # Test finding similar items
        similar = engine.find_similar_items(test_embeddings[0], top_k=5)
        print(f"Found {len(similar)} similar items")
        
        # Test nearest neighbor search
        print("Testing nearest neighbor search...")
        nn_search = NearestNeighborSearch(storage, SearchMethod.FAISS)
        
        results = nn_search.search(test_embeddings[0], k=5)
        print(f"NN search found {len(results)} results")
        
        # Test batch operations
        batch_results = engine.batch_similarity(test_embeddings[:3], top_k=3)
        print(f"Batch similarity completed for {len(batch_results)} queries")
        
        # Test outfit suggestions
        suggestions = engine.get_outfit_suggestions(["test_0.jpg", "test_1.jpg"])
        print(f"Generated outfit suggestions: {list(suggestions.keys())}")
        
        # Get statistics
        engine_stats = engine.get_statistics()
        nn_stats = nn_search.get_performance_stats()
        
        print(f"Engine stats: {engine_stats}")
        print(f"NN performance: {nn_stats}")
        
        print("Similarity system integration test completed successfully!")
        
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
