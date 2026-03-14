"""
Test suite for Task 13 - RAG + Explainability System
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.rag_system.rule_base import RuleBase, FashionRule, RuleCategory, RulePriority, OccasionType, SeasonType
from src.rag_system.embedding_index import EmbeddingIndex
from src.rag_system.explainability import ExplainabilityEngine, ExplanationType, Explanation


class TestRuleBase(unittest.TestCase):
    """Test fashion rule base"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rule_base = RuleBase()
    
    def test_rule_base_initialization(self):
        """Test rule base initialization"""
        self.assertGreater(len(self.rule_base.rules), 0)
        
        # Check that default rules were loaded
        categories = set(rule.category for rule in self.rule_base.rules)
        self.assertIn(RuleCategory.COLOR_HARMONY, categories)
        self.assertIn(RuleCategory.PATTERN_MIXING, categories)
        self.assertIn(RuleCategory.STYLE_COORDINATION, categories)
    
    def test_add_rule(self):
        """Test adding new rule"""
        new_rule = FashionRule(
            id="test_rule",
            title="Test Rule",
            description="A test rule for testing",
            category=RuleCategory.STYLE_COORDINATION,
            priority=RulePriority.RECOMMENDED,
            occasions=[OccasionType.CASUAL],
            seasons=[SeasonType.ALL_SEASON],
            recommendations=["Test recommendation"],
            confidence=0.8
        )
        
        initial_count = len(self.rule_base.rules)
        self.rule_base.add_rule(new_rule)
        
        self.assertEqual(len(self.rule_base.rules), initial_count + 1)
        
        # Test retrieval
        retrieved_rule = self.rule_base.get_rule("test_rule")
        self.assertIsNotNone(retrieved_rule)
        self.assertEqual(retrieved_rule.title, "Test Rule")
    
    def test_filter_rules(self):
        """Test rule filtering"""
        # Filter by category
        color_rules = self.rule_base.filter_rules(category=RuleCategory.COLOR_HARMONY)
        self.assertGreater(len(color_rules), 0)
        
        for rule in color_rules:
            self.assertEqual(rule.category, RuleCategory.COLOR_HARMONY)
        
        # Filter by priority
        critical_rules = self.rule_base.filter_rules(priority=RulePriority.CRITICAL)
        self.assertGreater(len(critical_rules), 0)
        
        for rule in critical_rules:
            self.assertEqual(rule.priority, RulePriority.CRITICAL)
        
        # Filter by occasion
        business_rules = self.rule_base.filter_rules(occasion=OccasionType.BUSINESS)
        self.assertGreater(len(business_rules), 0)
        
        for rule in business_rules:
            self.assertIn(OccasionType.BUSINESS, rule.occasions)
    
    def test_search_rules(self):
        """Test rule search"""
        # Search for color-related rules
        color_results = self.rule_base.search_rules("color", top_k=5)
        self.assertGreater(len(color_results), 0)
        
        for rule, score in color_results:
            self.assertGreater(score, 0)
        
        # Search for pattern-related rules
        pattern_results = self.rule_base.search_rules("pattern", top_k=3)
        self.assertGreater(len(pattern_results), 0)
    
    def test_get_rules_for_outfit(self):
        """Test getting rules for specific outfit context"""
        context = {
            'occasion': 'business',
            'season': 'winter',
            'colors': ['blue', 'gray'],
            'patterns': ['solid'],
            'style': 'professional'
        }
        
        relevant_rules = self.rule_base.get_rules_for_outfit(context)
        self.assertGreater(len(relevant_rules), 0)
        
        # Should include business-related rules
        business_rules = [r for r in relevant_rules if OccasionType.BUSINESS in r.occasions]
        self.assertGreater(len(business_rules), 0)
    
    def test_rule_statistics(self):
        """Test rule statistics"""
        stats = self.rule_base.get_rule_statistics()
        
        self.assertIn('total_rules', stats)
        self.assertIn('category_distribution', stats)
        self.assertIn('priority_distribution', stats)
        self.assertIn('occasion_distribution', stats)
        self.assertIn('season_distribution', stats)
        
        self.assertGreater(stats['total_rules'], 0)
        self.assertGreater(stats['avg_confidence'], 0)
    
    def test_rule_serialization(self):
        """Test rule serialization to/from dict"""
        rule = self.rule_base.rules[0]  # Get first rule
        
        # Convert to dict
        rule_dict = rule.to_dict()
        
        # Check required fields
        required_fields = ['id', 'title', 'description', 'category', 'priority', 'occasions', 'seasons']
        for field in required_fields:
            self.assertIn(field, rule_dict)
        
        # Convert back from dict
        restored_rule = FashionRule.from_dict(rule_dict)
        
        # Check equality
        self.assertEqual(restored_rule.id, rule.id)
        self.assertEqual(restored_rule.title, rule.title)
        self.assertEqual(restored_rule.category, rule.category)
        self.assertEqual(restored_rule.priority, rule.priority)


class TestEmbeddingIndex(unittest.TestCase):
    """Test embedding index system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.rule_base = RuleBase()
        self.embedding_index = EmbeddingIndex()
        
        # Build index with test rules
        self.embedding_index.build_index(self.rule_base.rules[:5])  # Use subset for testing
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_index_initialization(self):
        """Test embedding index initialization"""
        self.assertIsNotNone(self.embedding_index.model)
        self.assertGreater(self.embedding_index.embedding_dim, 0)
        self.assertIsNotNone(self.embedding_index.index)
        self.assertGreater(len(self.embedding_index.rule_ids), 0)
    
    def test_create_embeddings(self):
        """Test embedding creation"""
        test_rules = self.rule_base.rules[:3]
        embeddings = self.embedding_index.create_embeddings(test_rules)
        
        self.assertEqual(embeddings.shape[0], len(test_rules))
        self.assertEqual(embeddings.shape[1], self.embedding_index.embedding_dim)
    
    def test_search(self):
        """Test semantic search"""
        # Search for color-related rules
        results = self.embedding_index.search("color coordination", top_k=3)
        
        self.assertLessEqual(len(results), 3)
        
        for rule_id, similarity in results:
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)
            
            # Verify rule exists
            rule = self.rule_base.get_rule(rule_id)
            self.assertIsNotNone(rule)
    
    def test_search_by_context(self):
        """Test context-based search"""
        context = {
            'occasion': 'business',
            'season': 'winter',
            'colors': ['blue', 'gray']
        }
        
        results = self.embedding_index.search_by_context(context, top_k=3)
        
        self.assertLessEqual(len(results), 3)
        
        for rule_id, similarity in results:
            self.assertGreaterEqual(similarity, 0.0)
    
    def test_get_rule_recommendations(self):
        """Test rule recommendations"""
        context = {
            'occasion': 'business',
            'season': 'winter',
            'colors': ['blue', 'gray']
        }
        
        recommendations = self.embedding_index.get_rule_recommendations(
            context, self.rule_base, top_k=3
        )
        
        self.assertLessEqual(len(recommendations), 3)
        
        for rec in recommendations:
            self.assertIn('rule_id', rec)
            self.assertIn('title', rec)
            self.assertIn('similarity', rec)
            self.assertIn('relevance_score', rec)
            
            # Verify rule exists
            rule = self.rule_base.get_rule(rec['rule_id'])
            self.assertIsNotNone(rule)
    
    def test_save_and_load_index(self):
        """Test saving and loading index"""
        index_path = self.temp_dir / "test_index"
        
        # Save index
        self.embedding_index.save_index(index_path)
        
        # Check files exist
        self.assertTrue(index_path.exists())
        self.assertTrue(index_path.with_suffix('.meta').exists())
        
        # Create new index and load
        new_index = EmbeddingIndex()
        loaded = new_index.load_index(index_path)
        
        self.assertTrue(loaded)
        self.assertEqual(new_index.index.ntotal, self.embedding_index.index.ntotal)
        self.assertEqual(len(new_index.rule_ids), len(self.embedding_index.rule_ids))
    
    def test_index_statistics(self):
        """Test index statistics"""
        stats = self.embedding_index.get_index_statistics()
        
        self.assertIn('status', stats)
        self.assertIn('total_rules', stats)
        self.assertIn('embedding_dim', stats)
        self.assertIn('model_name', stats)
        
        self.assertEqual(stats['status'], 'built')
        self.assertGreater(stats['total_rules'], 0)


class TestExplainabilityEngine(unittest.TestCase):
    """Test explainability engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rule_base = RuleBase()
        self.embedding_index = EmbeddingIndex()
        self.embedding_index.build_index(self.rule_base.rules[:5])  # Use subset for testing
        self.engine = ExplainabilityEngine(self.rule_base, self.embedding_index)
        
        # Test recommendation and context
        self.recommendation = {
            'id': 'test_rec',
            'items': ['blue_shirt', 'gray_pants', 'black_shoes'],
            'score': 0.85
        }
        
        self.context = {
            'occasion': 'business',
            'season': 'winter',
            'colors': ['blue', 'gray', 'black'],
            'patterns': ['solid'],
            'style': 'professional'
        }
    
    def test_engine_initialization(self):
        """Test explainability engine initialization"""
        self.assertIsNotNone(self.engine.rule_base)
        self.assertIsNotNone(self.engine.embedding_index)
        self.assertIsNotNone(self.engine.templates)
    
    def test_generate_explanation(self):
        """Test explanation generation"""
        explanation = self.engine.generate_explanation(
            self.recommendation, 
            self.context, 
            ExplanationType.COMPREHENSIVE
        )
        
        # Check explanation structure
        self.assertIsInstance(explanation, Explanation)
        self.assertEqual(explanation.recommendation_id, 'test_rec')
        self.assertEqual(explanation.explanation_type, ExplanationType.COMPREHENSIVE)
        
        self.assertIsNotNone(explanation.primary_reason)
        self.assertIsInstance(explanation.supporting_rules, list)
        self.assertGreaterEqual(explanation.confidence_score, 0.0)
        self.assertLessEqual(explanation.confidence_score, 1.0)
        self.assertIsInstance(explanation.context_factors, dict)
        self.assertIsInstance(explanation.alternatives, list)
        self.assertIsInstance(explanation.additional_tips, list)
        self.assertIsNotNone(explanation.timestamp)
    
    def test_explanation_serialization(self):
        """Test explanation serialization"""
        explanation = self.engine.generate_explanation(
            self.recommendation, 
            self.context
        )
        
        # Convert to dict
        explanation_dict = explanation.to_dict()
        
        # Check required fields
        required_fields = [
            'recommendation_id', 'explanation_type', 'primary_reason',
            'supporting_rules', 'confidence_score', 'context_factors',
            'alternatives', 'additional_tips', 'timestamp'
        ]
        
        for field in required_fields:
            self.assertIn(field, explanation_dict)
    
    def test_explain_outfit_score(self):
        """Test outfit score explanation"""
        outfit = {
            'items': ['blue_shirt', 'gray_pants', 'black_shoes'],
            'score': 0.85
        }
        
        score_explanation = self.engine.explain_outfit_score(outfit, self.context)
        
        # Check structure
        self.assertIn('overall_score', score_explanation)
        self.assertIn('components', score_explanation)
        self.assertIn('strengths', score_explanation)
        self.assertIn('weaknesses', score_explanation)
        self.assertIn('improvement_suggestions', score_explanation)
        
        self.assertEqual(score_explanation['overall_score'], 0.85)
        self.assertIsInstance(score_explanation['components'], dict)
        self.assertIsInstance(score_explanation['strengths'], list)
        self.assertIsInstance(score_explanation['weaknesses'], list)
        self.assertIsInstance(score_explanation['improvement_suggestions'], list)
    
    def test_generate_comparison_explanation(self):
        """Test outfit comparison explanation"""
        outfit1 = {
            'items': ['blue_shirt', 'gray_pants', 'black_shoes'],
            'score': 0.85
        }
        
        outfit2 = {
            'items': ['red_shirt', 'blue_jeans', 'brown_shoes'],
            'score': 0.72
        }
        
        comparison = self.engine.generate_comparison_explanation(outfit1, outfit2, self.context)
        
        # Check structure
        self.assertIn('outfit1', comparison)
        self.assertIn('outfit2', comparison)
        self.assertIn('winner', comparison)
        self.assertIn('score_difference', comparison)
        self.assertIn('key_differences', comparison)
        
        self.assertEqual(comparison['winner'], 'outfit1')  # Higher score
        self.assertGreater(comparison['score_difference'], 0)
    
    def test_export_explanation(self):
        """Test explanation export"""
        explanation = self.engine.generate_explanation(
            self.recommendation, 
            self.context
        )
        
        output_path = Path(tempfile.mkdtemp()) / "test_explanation.json"
        
        self.engine.export_explanation(explanation, output_path)
        
        # Check file exists
        self.assertTrue(output_path.exists())
        
        # Clean up
        output_path.unlink()
        output_path.parent.rmdir()
    
    def test_batch_explain(self):
        """Test batch explanation generation"""
        recommendations = [
            {'id': 'rec1', 'items': ['item1', 'item2'], 'score': 0.8},
            {'id': 'rec2', 'items': ['item3', 'item4'], 'score': 0.7},
            {'id': 'rec3', 'items': ['item5', 'item6'], 'score': 0.9}
        ]
        
        explanations = self.engine.batch_explain(recommendations, self.context)
        
        self.assertEqual(len(explanations), len(recommendations))
        
        for explanation in explanations:
            self.assertIsInstance(explanation, Explanation)
            self.assertIn(explanation.recommendation_id, ['rec1', 'rec2', 'rec3'])


def run_integration_test():
    """Integration test for RAG system"""
    print("Running RAG system integration test...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create rule base
        print("Creating rule base...")
        rule_base = RuleBase()
        
        # Create embedding index
        print("Creating embedding index...")
        embedding_index = EmbeddingIndex()
        embedding_index.build_index(rule_base.rules)
        
        # Test semantic search
        print("Testing semantic search...")
        search_results = embedding_index.search("business professional outfit", top_k=3)
        print(f"Found {len(search_results)} relevant rules")
        
        for rule_id, similarity in search_results:
            rule = rule_base.get_rule(rule_id)
            print(f"  {rule.title} (similarity: {similarity:.3f})")
        
        # Test context search
        print("\nTesting context search...")
        context = {
            'occasion': 'business',
            'season': 'winter',
            'colors': ['navy', 'white', 'gray'],
            'patterns': ['solid', 'pinstripe'],
            'style': 'professional'
        }
        
        context_results = embedding_index.search_by_context(context, top_k=3)
        print(f"Found {len(context_results)} context-relevant rules")
        
        # Test recommendations
        print("\nTesting rule recommendations...")
        recommendations = embedding_index.get_rule_recommendations(
            context, rule_base, top_k=3
        )
        
        for rec in recommendations:
            print(f"  {rec['title']} (relevance: {rec['relevance_score']:.3f})")
        
        # Create explainability engine
        print("\nCreating explainability engine...")
        engine = ExplainabilityEngine(rule_base, embedding_index)
        
        # Test outfit recommendation
        outfit_recommendation = {
            'id': 'business_outfit_1',
            'items': ['navy_blazer', 'white_shirt', 'gray_trousers', 'black_oxfords'],
            'score': 0.92
        }
        
        # Generate explanation
        explanation = engine.generate_explanation(outfit_recommendation, context)
        
        print(f"Generated explanation:")
        print(f"  Primary reason: {explanation.primary_reason}")
        print(f"  Confidence: {explanation.confidence_score:.3f}")
        print(f"  Supporting rules: {len(explanation.supporting_rules)}")
        print(f"  Context factors: {len(explanation.context_factors)}")
        print(f"  Alternatives: {len(explanation.alternatives)}")
        print(f"  Additional tips: {len(explanation.additional_tips)}")
        
        # Test score explanation
        score_explanation = engine.explain_outfit_score(outfit_recommendation, context)
        print(f"\nScore breakdown:")
        print(f"  Overall: {score_explanation['overall_score']:.3f}")
        print(f"  Components: {list(score_explanation['components'].keys())}")
        print(f"  Strengths: {score_explanation['strengths']}")
        
        # Test comparison
        casual_outfit = {
            'items': ['tshirt', 'jeans', 'sneakers'],
            'score': 0.75
        }
        
        comparison = engine.generate_comparison_explanation(
            outfit_recommendation, casual_outfit, context
        )
        
        print(f"\nComparison:")
        print(f"  Winner: {comparison['winner']}")
        print(f"  Score difference: {comparison['score_difference']:.3f}")
        
        # Test saving and loading
        print("\nTesting persistence...")
        index_path = temp_dir / "test_index"
        embedding_index.save_index(index_path)
        
        new_index = EmbeddingIndex()
        loaded = new_index.load_index(index_path)
        
        if loaded:
            print("Index loaded successfully")
            
            # Test search with loaded index
            loaded_results = new_index.search("professional attire", top_k=2)
            print(f"Loaded index search: {len(loaded_results)} results")
        
        # Export explanation
        explanation_path = temp_dir / "test_explanation.json"
        engine.export_explanation(explanation, explanation_path)
        print(f"Explanation exported to: {explanation_path}")
        
        # Get statistics
        rule_stats = rule_base.get_rule_statistics()
        index_stats = embedding_index.get_index_statistics()
        
        print(f"\nSystem statistics:")
        print(f"  Rules: {rule_stats['total_rules']}")
        print(f"  Categories: {len(rule_stats['category_distribution'])}")
        print(f"  Index size: {index_stats['total_rules']} rules")
        print(f"  Embedding dim: {index_stats['embedding_dim']}")
        
        print("RAG system integration test completed successfully!")
        
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
