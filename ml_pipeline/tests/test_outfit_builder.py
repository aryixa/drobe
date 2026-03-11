"""
Test suite for Task 11 - Outfit Builder (CORE FEATURE)
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.outfit_builder.core_engine import (
    OutfitBuilder, Outfit, ClothingItem, ClothingType, OutfitStyle
)
from src.outfit_builder.ranking import (
    OutfitRanker, RankingMethod, RankingWeights, RankingResult
)
from src.embeddings.embedding_storage import EmbeddingStorage
from sklearn.preprocessing import normalize


class TestClothingItem(unittest.TestCase):
    """Test clothing item data structure"""
    
    def test_clothing_item_creation(self):
        """Test clothing item creation and validation"""
        embedding = np.random.rand(512)
        embedding = embedding / np.linalg.norm(embedding)
        
        item = ClothingItem(
            path="test.jpg",
            embedding=embedding,
            clothing_type=ClothingType.TOP,
            style=OutfitStyle.CASUAL,
            color="blue",
            pattern="solid",
            formality=0.5
        )
        
        self.assertEqual(item.path, "test.jpg")
        self.assertEqual(item.clothing_type, ClothingType.TOP)
        self.assertEqual(item.style, OutfitStyle.CASUAL)
        self.assertEqual(item.color, "blue")
        self.assertEqual(item.pattern, "solid")
        self.assertEqual(item.formality, 0.5)
        
        # Check embedding normalization
        norm = np.linalg.norm(item.embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_clothing_item_with_list_embedding(self):
        """Test clothing item with list embedding"""
        embedding_list = np.random.rand(512).tolist()
        
        item = ClothingItem(
            path="test.jpg",
            embedding=embedding_list,
            clothing_type=ClothingType.TOP
        )
        
        self.assertIsInstance(item.embedding, np.ndarray)
        self.assertEqual(item.embedding.shape, (512,))


class TestOutfit(unittest.TestCase):
    """Test outfit data structure"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        
        # Create test items
        self.items = [
            ClothingItem("top.jpg", np.random.rand(512), ClothingType.TOP, color="blue"),
            ClothingItem("bottom.jpg", np.random.rand(512), ClothingType.BOTTOM, color="blue"),
            ClothingItem("shoes.jpg", np.random.rand(512), ClothingType.SHOES, color="white")
        ]
        
        # Normalize embeddings
        for item in self.items:
            item.embedding = item.embedding / np.linalg.norm(item.embedding)
    
    def test_outfit_creation(self):
        """Test outfit creation"""
        outfit = Outfit(items=self.items)
        
        self.assertEqual(len(outfit.items), 3)
        self.assertIsNotNone(outfit.embedding)
        self.assertEqual(outfit.embedding.shape, (512,))
        
        # Check embedding normalization
        norm = np.linalg.norm(outfit.embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_outfit_compatibility_scores(self):
        """Test outfit compatibility scoring"""
        outfit = Outfit(items=self.items)
        
        # Scores should be between 0 and 1
        self.assertGreaterEqual(outfit.style_score, 0.0)
        self.assertLessEqual(outfit.style_score, 1.0)
        self.assertGreaterEqual(outfit.color_score, 0.0)
        self.assertLessEqual(outfit.color_score, 1.0)
        self.assertGreaterEqual(outfit.pattern_score, 0.0)
        self.assertLessEqual(outfit.pattern_score, 1.0)
        self.assertGreaterEqual(outfit.formality_score, 0.0)
        self.assertLessEqual(outfit.formality_score, 1.0)
    
    def test_get_items_by_type(self):
        """Test getting items by type"""
        outfit = Outfit(items=self.items)
        
        tops = outfit.get_items_by_type(ClothingType.TOP)
        bottoms = outfit.get_items_by_type(ClothingType.BOTTOM)
        shoes = outfit.get_items_by_type(ClothingType.SHOES)
        
        self.assertEqual(len(tops), 1)
        self.assertEqual(len(bottoms), 1)
        self.assertEqual(len(shoes), 1)
        
        self.assertEqual(tops[0].clothing_type, ClothingType.TOP)
        self.assertEqual(bottoms[0].clothing_type, ClothingType.BOTTOM)
        self.assertEqual(shoes[0].clothing_type, ClothingType.SHOES)
    
    def test_has_complete_set(self):
        """Test complete set detection"""
        complete_outfit = Outfit(items=self.items)
        incomplete_outfit = Outfit(items=self.items[:2])  # Missing shoes
        
        self.assertTrue(complete_outfit.has_complete_set())
        self.assertFalse(incomplete_outfit.has_complete_set())


class TestOutfitBuilder(unittest.TestCase):
    """Test outfit builder core engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = EmbeddingStorage(self.temp_dir / "test.db")
        
        # Create test data
        self._create_test_data()
        
        self.builder = OutfitBuilder(self.storage)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """Create test clothing items"""
        np.random.seed(42)
        
        test_items = [
            ("top1.jpg", ClothingType.TOP, "blue", "solid", OutfitStyle.CASUAL, 0.3),
            ("top2.jpg", ClothingType.TOP, "white", "striped", OutfitStyle.FORMAL, 0.8),
            ("bottom1.jpg", ClothingType.BOTTOM, "blue", "solid", OutfitStyle.CASUAL, 0.4),
            ("bottom2.jpg", ClothingType.BOTTOM, "black", "solid", OutfitStyle.FORMAL, 0.9),
            ("shoes1.jpg", ClothingType.SHOES, "white", "solid", OutfitStyle.CASUAL, 0.2),
            ("shoes2.jpg", ClothingType.SHOES, "black", "solid", OutfitStyle.FORMAL, 0.7)
        ]
        
        for path, clothing_type, color, pattern, style, formality in test_items:
            embedding = np.random.rand(512)
            embedding = embedding / np.linalg.norm(embedding)
            
            tags = {
                'type': {clothing_type.value: 1.0},
                'color': {color: 1.0},
                'pattern': {pattern: 1.0},
                'style': {style.value: 1.0},
                'formality': {str(formality): 1.0}
            }
            
            self.storage.store_embedding(path, embedding)
            self.storage.add_tags(path, tags)
    
    def test_builder_initialization(self):
        """Test builder initialization"""
        self.assertIsNotNone(self.builder.storage)
        self.assertIsNotNone(self.builder.similarity_engine)
        self.assertGreater(len(self.builder.items_cache), 0)
        
        # Check that items were loaded correctly
        self.assertIn(ClothingType.TOP, self.builder.items_cache)
        self.assertIn(ClothingType.BOTTOM, self.builder.items_cache)
        self.assertIn(ClothingType.SHOES, self.builder.items_cache)
        
        self.assertEqual(len(self.builder.items_cache[ClothingType.TOP]), 2)
        self.assertEqual(len(self.builder.items_cache[ClothingType.BOTTOM]), 2)
        self.assertEqual(len(self.builder.items_cache[ClothingType.SHOES]), 2)
    
    def test_generate_outfit_combinations(self):
        """Test outfit combination generation"""
        combinations = self.builder.generate_outfit_combinations(max_combinations=10)
        
        self.assertGreater(len(combinations), 0)
        self.assertLessEqual(len(combinations), 10)
        
        # Check that each combination has the right items
        for outfit in combinations:
            self.assertTrue(outfit.has_complete_set())
            self.assertEqual(len(outfit.items), 3)
    
    def test_build_outfit_from_items(self):
        """Test building outfit from specific items"""
        outfit = self.builder.build_outfit_from_items(
            "top1.jpg", "bottom1.jpg", "shoes1.jpg"
        )
        
        self.assertIsNotNone(outfit)
        self.assertEqual(len(outfit.items), 3)
        self.assertTrue(outfit.has_complete_set())
        
        # Check item types
        item_types = [item.clothing_type for item in outfit.items]
        self.assertIn(ClothingType.TOP, item_types)
        self.assertIn(ClothingType.BOTTOM, item_types)
        self.assertIn(ClothingType.SHOES, item_types)
    
    def test_rank_outfits(self):
        """Test outfit ranking"""
        combinations = self.builder.generate_outfit_combinations(max_combinations=5)
        
        ranked = self.builder.rank_outfits(combinations)
        
        self.assertEqual(len(ranked), len(combinations))
        
        # Check that results are sorted by score
        if len(ranked) > 1:
            for i in range(len(ranked) - 1):
                self.assertGreaterEqual(ranked[i].score, ranked[i + 1].score)
    
    def test_find_similar_outfits(self):
        """Test finding similar outfits"""
        # Create reference outfit
        reference = self.builder.build_outfit_from_items(
            "top1.jpg", "bottom1.jpg", "shoes1.jpg"
        )
        
        similar = self.builder.find_similar_outfits(reference, top_k=3)
        
        self.assertLessEqual(len(similar), 3)
        
        # Check that results are sorted by similarity
        if len(similar) > 1:
            for i in range(len(similar) - 1):
                self.assertGreaterEqual(similar[i].score, similar[i + 1].score)
    
    def test_get_outfit_suggestions(self):
        """Test getting outfit suggestions"""
        suggestions = self.builder.get_outfit_suggestions(["top1.jpg"])
        
        self.assertIn(ClothingType.BOTTOM, suggestions)
        self.assertIn(ClothingType.SHOES, suggestions)
        
        # Check that suggestions are valid items
        for clothing_type, items in suggestions.items():
            for item in items:
                self.assertEqual(item.clothing_type, clothing_type)
    
    def test_optimize_outfit(self):
        """Test outfit optimization"""
        initial = self.builder.build_outfit_from_items(
            "top1.jpg", "bottom1.jpg", "shoes1.jpg"
        )
        
        optimized = self.builder.optimize_outfit(initial)
        
        self.assertIsNotNone(optimized)
        self.assertEqual(len(optimized.items), 3)
        self.assertTrue(optimized.has_complete_set())
    
    def test_get_statistics(self):
        """Test builder statistics"""
        stats = self.builder.get_statistics()
        
        self.assertIn('total_items', stats)
        self.assertIn('items_by_type', stats)
        self.assertIn('available_types', stats)
        
        self.assertEqual(stats['total_items'], 6)
        self.assertEqual(stats['items_by_type']['top'], 2)
        self.assertEqual(stats['items_by_type']['bottom'], 2)
        self.assertEqual(stats['items_by_type']['shoes'], 2)


class TestOutfitRanker(unittest.TestCase):
    """Test outfit ranking system"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        
        # Create test outfits
        self.items = [
            ClothingItem("top1.jpg", np.random.rand(512), ClothingType.TOP, style=OutfitStyle.CASUAL, color="blue", pattern="solid", formality=0.3),
            ClothingItem("bottom1.jpg", np.random.rand(512), ClothingType.BOTTOM, style=OutfitStyle.CASUAL, color="blue", pattern="solid", formality=0.4),
            ClothingItem("shoes1.jpg", np.random.rand(512), ClothingType.SHOES, style=OutfitStyle.CASUAL, color="white", pattern="solid", formality=0.2),
            ClothingItem("top2.jpg", np.random.rand(512), ClothingType.TOP, style=OutfitStyle.FORMAL, color="white", pattern="solid", formality=0.8),
            ClothingItem("bottom2.jpg", np.random.rand(512), ClothingType.BOTTOM, style=OutfitStyle.FORMAL, color="black", pattern="solid", formality=0.9),
            ClothingItem("shoes2.jpg", np.random.rand(512), ClothingType.SHOES, style=OutfitStyle.FORMAL, color="black", pattern="solid", formality=0.7),
        ]
        
        # Normalize embeddings
        for item in self.items:
            item.embedding = item.embedding / np.linalg.norm(item.embedding)
        
        self.outfits = [
            Outfit(items=[self.items[0], self.items[1], self.items[2]]),  # Casual
            Outfit(items=[self.items[3], self.items[4], self.items[5]]),  # Formal
            Outfit(items=[self.items[0], self.items[4], self.items[2]]),  # Mixed
        ]
    
    def test_ranker_initialization(self):
        """Test ranker initialization"""
        ranker = OutfitRanker()
        
        self.assertIsNotNone(ranker.weights)
        self.assertIsNotNone(ranker.scoring_functions)
        self.assertEqual(ranker.method, RankingMethod.WEIGHTED_SUM)
        
        # Check weight normalization
        total_weight = (ranker.weights.style_compatibility + 
                       ranker.weights.color_harmony + 
                       ranker.weights.pattern_balance + 
                       ranker.weights.formality_consistency + 
                       ranker.weights.visual_appeal)
        self.assertAlmostEqual(total_weight, 1.0, places=5)
    
    def test_weighted_sum_ranking(self):
        """Test weighted sum ranking"""
        ranker = OutfitRanker(method=RankingMethod.WEIGHTED_SUM)
        results = ranker.rank_outfits(self.outfits)
        
        self.assertEqual(len(results), len(self.outfits))
        
        # Check that results are sorted
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].score, results[i + 1].score)
        
        # Check result structure
        for result in results:
            self.assertIsInstance(result, RankingResult)
            self.assertIsInstance(result.outfit, Outfit)
            self.assertIsInstance(result.score, float)
            self.assertIsInstance(result.rank, int)
            self.assertIsInstance(result.sub_scores, dict)
            self.assertIsInstance(result.explanation, str)
    
    def test_rule_based_ranking(self):
        """Test rule-based ranking"""
        ranker = OutfitRanker(method=RankingMethod.RULE_BASED)
        results = ranker.rank_outfits(self.outfits)
        
        self.assertEqual(len(results), len(self.outfits))
        
        # Check that fashion_rules score is included
        for result in results:
            self.assertIn('fashion_rules', result.sub_scores)
    
    def test_hybrid_ranking(self):
        """Test hybrid ranking"""
        ranker = OutfitRanker(method=RankingMethod.HYBRID)
        results = ranker.rank_outfits(self.outfits)
        
        self.assertEqual(len(results), len(self.outfits))
        
        # Check that multiple scoring methods are included
        for result in results:
            self.assertIn('similarity', result.sub_scores)
            self.assertIn('rule_based', result.sub_scores)
    
    def test_user_preferences(self):
        """Test user preference application"""
        user_prefs = {
            'style_compatibility': 0.5,
            'color_harmony': 0.3,
            'pattern_balance': 0.2
        }
        
        ranker = OutfitRanker()
        results = ranker.rank_outfits(self.outfits, user_preferences=user_prefs)
        
        self.assertEqual(len(results), len(self.outfits))
    
    def test_scoring_functions(self):
        """Test individual scoring functions"""
        ranker = OutfitRanker()
        
        for outfit in self.outfits:
            # Test each scoring function
            style_score = ranker._score_style_compatibility(outfit)
            color_score = ranker._score_color_harmony(outfit)
            pattern_score = ranker._score_pattern_balance(outfit)
            formality_score = ranker._score_formality_consistency(outfit)
            appeal_score = ranker._score_visual_appeal(outfit)
            
            # All scores should be between 0 and 1
            for score in [style_score, color_score, pattern_score, formality_score, appeal_score]:
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
    
    def test_ranking_statistics(self):
        """Test ranking statistics"""
        ranker = OutfitRanker()
        results = ranker.rank_outfits(self.outfits)
        
        stats = ranker.get_ranking_statistics(results)
        
        self.assertIn('total_outfits', stats)
        self.assertIn('avg_score', stats)
        self.assertIn('std_score', stats)
        self.assertIn('min_score', stats)
        self.assertIn('max_score', stats)
        self.assertIn('score_distribution', stats)
        
        self.assertEqual(stats['total_outfits'], len(self.outfits))
    
    def test_compare_rankings(self):
        """Test comparison of ranking methods"""
        ranker = OutfitRanker()
        methods = [RankingMethod.WEIGHTED_SUM, RankingMethod.RULE_BASED]
        
        comparison = ranker.compare_rankings(self.outfits, methods)
        
        self.assertIn('weighted_sum', comparison)
        self.assertIn('rule_based', comparison)
        
        for method, results in comparison.items():
            self.assertEqual(len(results), len(self.outfits))


class TestRankingWeights(unittest.TestCase):
    """Test ranking weights"""
    
    def test_weights_normalization(self):
        """Test weight normalization"""
        weights = RankingWeights(
            style_compatibility=2.0,
            color_harmony=1.0,
            pattern_balance=1.0,
            formality_consistency=1.0,
            visual_appeal=1.0
        )
        
        weights.normalize()
        
        total = (weights.style_compatibility + 
                weights.color_harmony + 
                weights.pattern_balance + 
                weights.formality_consistency + 
                weights.visual_appeal)
        
        self.assertAlmostEqual(total, 1.0, places=5)


def run_integration_test():
    """Integration test for outfit builder"""
    print("Running outfit builder integration test...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test data
        storage = EmbeddingStorage(temp_dir / "integration.db")
        
        np.random.seed(42)
        
        # Create comprehensive test dataset
        test_items = [
            ("casual_top1.jpg", ClothingType.TOP, "blue", "solid", OutfitStyle.CASUAL, 0.3),
            ("casual_top2.jpg", ClothingType.TOP, "white", "striped", OutfitStyle.CASUAL, 0.4),
            ("formal_top1.jpg", ClothingType.TOP, "white", "solid", OutfitStyle.FORMAL, 0.8),
            ("casual_bottom1.jpg", ClothingType.BOTTOM, "blue", "solid", OutfitStyle.CASUAL, 0.3),
            ("casual_bottom2.jpg", ClothingType.BOTTOM, "khaki", "solid", OutfitStyle.CASUAL, 0.4),
            ("formal_bottom1.jpg", ClothingType.BOTTOM, "black", "solid", OutfitStyle.FORMAL, 0.9),
            ("casual_shoes1.jpg", ClothingType.SHOES, "white", "solid", OutfitStyle.CASUAL, 0.2),
            ("formal_shoes1.jpg", ClothingType.SHOES, "black", "solid", OutfitStyle.FORMAL, 0.7),
        ]
        
        for path, clothing_type, color, pattern, style, formality in test_items:
            embedding = np.random.rand(512)
            embedding = embedding / np.linalg.norm(embedding)
            
            tags = {
                'type': {clothing_type.value: 1.0},
                'color': {color: 1.0},
                'pattern': {pattern: 1.0},
                'style': {style.value: 1.0},
                'formality': {str(formality): 1.0}
            }
            
            storage.store_embedding(path, embedding)
            storage.add_tags(path, tags)
        
        # Test outfit builder
        print("Testing outfit builder...")
        builder = OutfitBuilder(storage)
        
        # Generate combinations
        combinations = builder.generate_outfit_combinations(max_combinations=20)
        print(f"Generated {len(combinations)} combinations")
        
        # Test ranking
        ranker = OutfitRanker(method=RankingMethod.HYBRID)
        results = ranker.rank_outfits(combinations)
        
        print(f"Top 3 outfits:")
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. Score: {result.score:.3f} - {result.explanation}")
        
        # Test specific outfit building
        outfit = builder.build_outfit_from_items(
            "casual_top1.jpg", "casual_bottom1.jpg", "casual_shoes1.jpg"
        )
        print(f"Built outfit score: {outfit.score:.3f}")
        
        # Test suggestions
        suggestions = builder.get_outfit_suggestions(["casual_top1.jpg"])
        print(f"Suggestions available for: {list(suggestions.keys())}")
        
        # Test optimization
        optimized = builder.optimize_outfit(outfit)
        print(f"Optimized outfit score: {optimized.score:.3f} (was {outfit.score:.3f})")
        
        # Get statistics
        builder_stats = builder.get_statistics()
        ranker_stats = ranker.get_ranking_statistics(results)
        
        print(f"Builder stats: {builder_stats}")
        print(f"Ranker stats: {ranker_stats}")
        
        print("Outfit builder integration test completed successfully!")
        
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
