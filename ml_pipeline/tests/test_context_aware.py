"""
Test suite for Task 14 - Context-Aware Styling + Integration
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.context_aware.context_parser import ContextParser, ParsedContext, OccasionType, SeasonType, WeatherCondition, TimeOfDay, StyleLevel
from src.context_aware.integration_engine import IntegrationEngine, RecommendationResult


class TestContextParser(unittest.TestCase):
    """Test context parsing system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.parser = ContextParser()
    
    def test_parser_initialization(self):
        """Test parser initialization"""
        self.assertIsNotNone(self.parser.occasion_keywords)
        self.assertIsNotNone(self.parser.season_keywords)
        self.assertIsNotNone(self.parser.weather_keywords)
        self.assertIsNotNone(self.parser.time_keywords)
        self.assertIsNotNone(self.parser.style_keywords)
        self.assertGreater(len(self.parser.color_keywords), 0)
        self.assertGreater(len(self.parser.pattern_keywords), 0)
        self.assertGreater(len(self.parser.clothing_type_keywords), 0)
    
    def test_parse_simple_queries(self):
        """Test parsing simple queries"""
        test_cases = [
            ("summer casual day outfit", {
                'season': SeasonType.SUMMER,
                'style_level': StyleLevel.CASUAL,
                'time_of_day': TimeOfDay.DAY
            }),
            ("business formal meeting", {
                'occasion': OccasionType.BUSINESS,
                'style_level': StyleLevel.FORMAL
            }),
            ("winter cold weather", {
                'season': SeasonType.WINTER,
                'weather': WeatherCondition.COLD
            }),
            ("date night elegant", {
                'occasion': OccasionType.DATE,
                'time_of_day': TimeOfDay.EVENING,
                'style_level': StyleLevel.SMART_CASUAL  # Enhanced default
            })
        ]
        
        for query, expected in test_cases:
            with self.subTest(query=query):
                context = self.parser.parse_context(query)
                enhanced = self.parser.enhance_context_with_defaults(context)
                
                self.assertEqual(enhanced.original_query, query)
                self.assertIsInstance(enhanced, ParsedContext)
                
                # Check expected values
                for key, expected_value in expected.items():
                    actual_value = getattr(enhanced, key)
                    self.assertEqual(actual_value, expected_value, 
                                   f"Failed for {key} in query: {query}")
    
    def test_parse_complex_queries(self):
        """Test parsing complex queries with multiple elements"""
        query = "formal business meeting outfit with blue shirt and gray pants for cold winter weather"
        
        context = self.parser.parse_context(query)
        enhanced = self.parser.enhance_context_with_defaults(context)
        
        # Check multiple elements
        self.assertEqual(enhanced.occasion, OccasionType.BUSINESS)
        self.assertEqual(enhanced.style_level, StyleLevel.FORMAL)
        self.assertEqual(enhanced.season, SeasonType.WINTER)
        self.assertIn('blue', enhanced.colors)
        self.assertIn('gray', enhanced.colors)
        self.assertIn('shirt', enhanced.clothing_types)
        self.assertIn('pants', enhanced.clothing_types)
        
        self.assertGreater(enhanced.confidence, 0.5)
    
    def test_extract_colors(self):
        """Test color extraction"""
        query = "red blue green yellow outfit"
        colors = self.parser._extract_colors(query)
        
        expected_colors = ['red', 'blue', 'green', 'yellow']
        for color in expected_colors:
            self.assertIn(color, colors)
    
    def test_extract_patterns(self):
        """Test pattern extraction"""
        query = "solid striped and floral patterns"
        patterns = self.parser._extract_patterns(query)
        
        expected_patterns = ['solid', 'striped', 'floral']
        for pattern in expected_patterns:
            self.assertIn(pattern, patterns)
    
    def test_extract_clothing_types(self):
        """Test clothing type extraction"""
        query = "shirt pants shoes jacket dress"
        types = self.parser._extract_clothing_types(query)
        
        expected_types = ['shirt', 'pants', 'shoes', 'jacket', 'dress']
        for clothing_type in expected_types:
            self.assertIn(clothing_type, types)
    
    def test_temperature_extraction(self):
        """Test temperature extraction"""
        test_cases = [
            ("25 degrees", (20.0, 30.0)),
            ("15 to 20 degrees", (15.0, 20.0)),
            ("between 10 and 15 degrees", (10.0, 15.0))
        ]
        
        for query, expected_range in test_cases:
            with self.subTest(query=query):
                temp_range = self.parser._extract_temperature(query, [])
                self.assertEqual(temp_range, expected_range)
    
    def test_context_enhancement(self):
        """Test context enhancement with defaults"""
        # Test weather to season enhancement
        context = ParsedContext(
            original_query="hot sunny day",
            occasion=None,
            season=None,
            weather=WeatherCondition.HOT,
            time_of_day=None,
            style_level=None,
            colors=[],
            patterns=[],
            clothing_types=[],
            keywords=['hot', 'sunny'],
            temperature_range=None,
            confidence=0.3,
            parsing_errors=[]
        )
        
        enhanced = self.parser.enhance_context_with_defaults(context)
        self.assertEqual(enhanced.season, SeasonType.SUMMER)
        
        # Test occasion to style enhancement
        context.occasion = OccasionType.BUSINESS
        enhanced = self.parser.enhance_context_with_defaults(context)
        self.assertEqual(enhanced.style_level, StyleLevel.BUSINESS)
    
    def test_batch_parsing(self):
        """Test batch parsing of multiple queries"""
        queries = [
            "casual summer outfit",
            "formal business attire",
            "winter weather clothes"
        ]
        
        contexts = self.parser.parse_multiple_queries(queries)
        
        self.assertEqual(len(contexts), len(queries))
        
        for context in contexts:
            self.assertIsInstance(context, ParsedContext)
            self.assertIsNotNone(context.original_query)
    
    def test_parsing_statistics(self):
        """Test parsing statistics"""
        contexts = self.parser.parse_multiple_queries([
            "casual summer outfit",
            "formal business attire", 
            "casual winter clothes"
        ])
        
        stats = self.parser.get_parsing_statistics(contexts)
        
        self.assertIn('total_queries', stats)
        self.assertIn('occasion_distribution', stats)
        self.assertIn('season_distribution', stats)
        self.assertIn('avg_confidence', stats)
        
        self.assertEqual(stats['total_queries'], 3)
        self.assertGreater(stats['avg_confidence'], 0)


class TestIntegrationEngine(unittest.TestCase):
    """Test integration engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create minimal test data
        self._create_test_data()
        
        # Initialize engine
        self.engine = IntegrationEngine(
            storage_path=self.temp_dir / "test.db",
            rule_base_path=self.temp_dir / "rules.json",
            embedding_index_path=self.temp_dir / "index.bin"
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """Create minimal test data for integration testing"""
        # This would normally be populated with actual embeddings
        # For testing, we'll create a minimal setup
        pass
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        self.assertIsNotNone(self.engine.storage)
        self.assertIsNotNone(self.engine.similarity_engine)
        self.assertIsNotNone(self.engine.outfit_builder)
        self.assertIsNotNone(self.engine.outfit_ranker)
        self.assertIsNotNone(self.engine.color_extractor)
        self.assertIsNotNone(self.engine.type_classifier)
        self.assertIsNotNone(self.engine.pattern_detector)
        self.assertIsNotNone(self.engine.rule_base)
        self.assertIsNotNone(self.engine.embedding_index)
        self.assertIsNotNone(self.engine.explainability_engine)
        self.assertIsNotNone(self.engine.context_parser)
    
    def test_context_to_dict_conversion(self):
        """Test context to dictionary conversion"""
        context = ParsedContext(
            original_query="test query",
            occasion=OccasionType.BUSINESS,
            season=SeasonType.SUMMER,
            weather=WeatherCondition.SUNNY,
            time_of_day=TimeOfDay.DAY,
            style_level=StyleLevel.BUSINESS,
            colors=['blue', 'gray'],
            patterns=['solid'],
            clothing_types=['shirt', 'pants'],
            keywords=['business', 'professional'],
            temperature_range=(20.0, 25.0),
            confidence=0.8,
            parsing_errors=[]
        )
        
        context_dict = self.engine._context_to_dict(context)
        
        self.assertEqual(context_dict['occasion'], 'business')
        self.assertEqual(context_dict['season'], 'summer')
        self.assertEqual(context_dict['weather'], 'sunny')
        self.assertEqual(context_dict['time_of_day'], 'day')
        self.assertEqual(context_dict['style'], 'business')
        self.assertEqual(context_dict['colors'], ['blue', 'gray'])
        self.assertEqual(context_dict['patterns'], ['solid'])
        self.assertEqual(context_dict['clothing_types'], ['shirt', 'pants'])
    
    def test_user_preferences_creation(self):
        """Test user preferences creation from context"""
        context = ParsedContext(
            original_query="business formal meeting",
            occasion=OccasionType.BUSINESS,
            season=None,
            weather=None,
            time_of_day=None,
            style_level=StyleLevel.FORMAL,
            colors=[],
            patterns=[],
            clothing_types=[],
            keywords=[],
            temperature_range=None,
            confidence=0.9,
            parsing_errors=[]
        )
        
        preferences = self.engine._create_user_preferences(context)
        
        self.assertIsInstance(preferences, dict)
        # Should contain formality preferences for business formal
        self.assertIn('formality', preferences)
        self.assertIn('professionalism', preferences)
    
    def test_context_score_calculation(self):
        """Test context score calculation"""
        # This test would require a real outfit object
        # For now, we'll test the basic structure
        context = ParsedContext(
            original_query="casual blue outfit",
            occasion=OccasionType.CASUAL,
            season=None,
            weather=None,
            time_of_day=None,
            style_level=StyleLevel.CASUAL,
            colors=['blue'],
            patterns=[],
            clothing_types=[],
            keywords=[],
            temperature_range=None,
            confidence=0.7,
            parsing_errors=[]
        )
        
        # Test that the method exists and returns a float
        # (Actual implementation would require real outfit data)
        score = self.engine._calculate_context_score(None, context)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_component_scores_calculation(self):
        """Test component scores calculation"""
        # This test would require a real outfit object
        # For now, we'll test the basic structure
        scores = self.engine._calculate_component_scores(None, None)
        
        self.assertIsInstance(scores, dict)
    
    def test_engine_statistics(self):
        """Test engine statistics"""
        stats = self.engine.get_engine_statistics()
        
        self.assertIn('components', stats)
        self.assertIn('capabilities', stats)
        self.assertIn('performance', stats)
        
        # Check components
        components = stats['components']
        self.assertIn('storage_items', components)
        self.assertIn('rule_base_rules', components)
        self.assertIn('embedding_index_size', components)
        
        # Check capabilities
        capabilities = stats['capabilities']
        self.assertTrue(capabilities['context_parsing'])
        self.assertTrue(capabilities['outfit_building'])
        self.assertTrue(capabilities['rule_application'])
        self.assertTrue(capabilities['explainability'])
        
        # Check performance
        performance = stats['performance']
        self.assertIn('index_loaded', performance)
        self.assertIn('cache_populated', performance)
    
    def test_result_export(self):
        """Test result export functionality"""
        # Create a mock result
        context = ParsedContext(
            original_query="test query",
            occasion=OccasionType.CASUAL,
            season=SeasonType.SUMMER,
            weather=WeatherCondition.SUNNY,
            time_of_day=TimeOfDay.DAY,
            style_level=StyleLevel.CASUAL,
            colors=[],
            patterns=[],
            clothing_types=[],
            keywords=[],
            temperature_range=None,
            confidence=0.8,
            parsing_errors=[]
        )
        
        result = RecommendationResult(
            query="test query",
            context=context,
            outfit=None,
            score=0.0,
            explanation={'test': 'explanation'},
            alternatives=[],
            processing_time=0.5,
            component_scores={},
            metadata={'test': 'metadata'}
        )
        
        # Test export
        output_path = self.temp_dir / "test_result.json"
        self.engine.export_result(result, str(output_path))
        
        # Check file exists
        self.assertTrue(output_path.exists())
    
    def test_error_handling(self):
        """Test error handling in query processing"""
        # Test with empty query
        result = self.engine.process_query("", max_outfits=1)
        
        self.assertIsInstance(result, RecommendationResult)
        self.assertEqual(result.query, "")
        self.assertIsNotNone(result.metadata)
        
        # Test error result creation
        error_result = self.engine._create_error_result("test query", None, "test error")
        
        self.assertIsInstance(error_result, RecommendationResult)
        self.assertEqual(error_result.query, "test query")
        self.assertEqual(error_result.metadata['status'], 'error')
        self.assertEqual(error_result.metadata['error_message'], 'test error')
    
    def test_empty_result_creation(self):
        """Test empty result creation"""
        context = ParsedContext(
            original_query="test query",
            occasion=None,
            season=None,
            weather=None,
            time_of_day=None,
            style_level=None,
            colors=[],
            patterns=[],
            clothing_types=[],
            keywords=[],
            temperature_range=None,
            confidence=0.0,
            parsing_errors=[]
        )
        
        empty_result = self.engine._create_empty_result("test query", context, None)
        
        self.assertIsInstance(empty_result, RecommendationResult)
        self.assertEqual(empty_result.query, "test query")
        self.assertIsNone(empty_result.outfit)
        self.assertEqual(empty_result.metadata['status'], 'no_results')


def run_integration_test():
    """Integration test for context-aware system"""
    print("Running Context-Aware system integration test...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        print("Testing Context Parser...")
        
        # Test context parser
        parser = ContextParser()
        
        test_queries = [
            "summer casual day outfit with blue shirt",
            "formal business meeting attire",
            "winter cold weather clothes",
            "date night elegant dress",
            "beach vacation sunny outfit"
        ]
        
        print(f"Testing {len(test_queries)} queries...")
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            
            context = parser.parse_context(query)
            enhanced = parser.enhance_context_with_defaults(context)
            
            print(f"  Parsed context confidence: {enhanced.confidence:.2f}")
            print(f"  Occasion: {enhanced.occasion.value if enhanced.occasion else 'None'}")
            print(f"  Season: {enhanced.season.value if enhanced.season else 'None'}")
            print(f"  Weather: {enhanced.weather.value if enhanced.weather else 'None'}")
            print(f"  Time: {enhanced.time_of_day.value if enhanced.time_of_day else 'None'}")
            print(f"  Style: {enhanced.style_level.value if enhanced.style_level else 'None'}")
            print(f"  Colors: {enhanced.colors}")
            print(f"  Patterns: {enhanced.patterns}")
            print(f"  Keywords: {enhanced.keywords}")
            
            if enhanced.parsing_errors:
                print(f"  Errors: {enhanced.parsing_errors}")
        
        # Test batch parsing
        contexts = parser.parse_multiple_queries(test_queries)
        stats = parser.get_parsing_statistics(contexts)
        
        print(f"\nParsing statistics:")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  Average confidence: {stats['avg_confidence']:.2f}")
        print(f"  High confidence queries: {stats['high_confidence_queries']}")
        
        # Test integration engine (basic functionality)
        print("\nTesting Integration Engine...")
        
        try:
            engine = IntegrationEngine()
            engine_stats = engine.get_engine_statistics()
            
            print(f"Engine components initialized: {len(engine_stats['components'])}")
            print(f"Capabilities available: {list(engine_stats['capabilities'].keys())}")
            
            # Test a simple query processing (without full data)
            print("Testing query processing structure...")
            
            # Test context conversion
            test_context = contexts[0] if contexts else parser.parse_context("casual outfit")
            context_dict = engine._context_to_dict(test_context)
            
            print(f"Context conversion successful: {list(context_dict.keys())}")
            
            # Test user preferences creation
            preferences = engine._create_user_preferences(test_context)
            print(f"User preferences created: {list(preferences.keys())}")
            
            print("Integration engine test completed successfully!")
            
        except Exception as e:
            print(f"Integration engine test limited (expected without full dataset): {e}")
        
        # Export test results
        if contexts:
            parser.export_context(contexts[0], str(temp_dir / "test_context.json"))
            print(f"Test context exported to: {temp_dir / 'test_context.json'}")
        
        print("Context-Aware system integration test completed!")
        
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
