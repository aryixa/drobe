"""
Task 14 - Context-Aware Styling + Integration: Integration Engine
Combine wardrobe + rules + context for final recommendations
"""

import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json

# Import all previous components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.embeddings.embedding_storage import EmbeddingStorage
from src.similarity.cosine_sim import CosineSimilarityEngine
from src.outfit_builder.core_engine import OutfitBuilder, ClothingItem, Outfit
from src.outfit_builder.ranking import OutfitRanker, RankingMethod
from src.smart_tags.color_extractor import ColorExtractor
from src.smart_tags.type_classifier import TypeClassifier
from src.smart_tags.pattern_detector import PatternDetector
from src.rag_system.rule_base import RuleBase
from src.rag_system.embedding_index import EmbeddingIndex
from src.rag_system.explainability import ExplainabilityEngine
from .context_parser import ContextParser, ParsedContext


@dataclass
class RecommendationResult:
    """Complete recommendation result with all components"""
    query: str
    context: ParsedContext
    outfit: Outfit
    score: float
    explanation: Dict
    alternatives: List[Dict]
    processing_time: float
    component_scores: Dict[str, float]
    metadata: Dict


class IntegrationEngine:
    """
    Main integration engine that combines all ML pipeline components
    - Context parsing
    - Outfit building
    - Rule application
    - Explainability
    """
    
    def __init__(self, 
                 storage_path: Optional[Union[str, Path]] = None,
                 rule_base_path: Optional[Union[str, Path]] = None,
                 embedding_index_path: Optional[Union[str, Path]] = None):
        """Initialize all components"""
        print("Initializing Integration Engine...")
        
        # Initialize core components
        self.storage = EmbeddingStorage(storage_path)
        self.similarity_engine = CosineSimilarityEngine(self.storage)
        self.outfit_builder = OutfitBuilder(self.storage)
        self.outfit_ranker = OutfitRanker()
        
        # Initialize smart tags
        self.color_extractor = ColorExtractor()
        self.type_classifier = TypeClassifier()
        self.pattern_detector = PatternDetector()
        
        # Initialize RAG system
        self.rule_base = RuleBase(rule_base_path)
        self.embedding_index = EmbeddingIndex()
        self.explainability_engine = ExplainabilityEngine(self.rule_base, self.embedding_index)
        
        # Initialize context parser
        self.context_parser = ContextParser()
        
        # Build embedding index if needed
        if self.embedding_index.index is None:
            self.embedding_index.build_index(self.rule_base.rules)
            self.embedding_index.save_index()
        
        print("Integration Engine initialized successfully")
    
    def process_query(self, 
                      query: str,
                      max_outfits: int = 5,
                      ranking_method: RankingMethod = RankingMethod.HYBRID) -> RecommendationResult:
        """
        Process a complete outfit recommendation query
        
        Args:
            query: Natural language input query
            max_outfits: Maximum number of outfit recommendations
            ranking_method: Method for ranking outfits
            
        Returns:
            Complete recommendation result
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Parse context
            context = self.context_parser.parse_context(query)
            enhanced_context = self.context_parser.enhance_context_with_defaults(context)
            
            # Step 2: Generate outfit candidates
            outfit_candidates = self.outfit_builder.generate_outfit_combinations(max_combinations=50)
            
            # Step 3: Apply context filtering
            filtered_outfits = self._filter_outfits_by_context(outfit_candidates, enhanced_context)
            
            # Step 4: Rank outfits
            ranked_outfits = self.outfit_ranker.rank_outfits(
                filtered_outfits, 
                method=ranking_method,
                user_preferences=self._create_user_preferences(enhanced_context)
            )
            
            # Step 5: Select top outfits
            top_outfits = ranked_outfits[:max_outfits]
            
            # Step 6: Generate explanations
            recommendations = []
            for i, ranked_outfit in enumerate(top_outfits):
                # Create recommendation dict
                recommendation = {
                    'id': f'rec_{i}',
                    'items': [item.path for item in ranked_outfit.outfit.items],
                    'score': ranked_outfit.score,
                    'outfit': ranked_outfit.outfit
                }
                
                # Generate explanation
                context_dict = self._context_to_dict(enhanced_context)
                explanation = self.explainability_engine.generate_explanation(
                    recommendation, 
                    context_dict
                )
                
                recommendations.append({
                    'recommendation': recommendation,
                    'explanation': explanation.to_dict()
                })
            
            # Step 7: Select best recommendation
            if recommendations:
                best_rec = recommendations[0]
                best_outfit = best_rec['recommendation']['outfit']
                best_score = best_rec['recommendation']['score']
                best_explanation = best_rec['explanation']
                
                # Generate alternatives
                alternatives = recommendations[1:3] if len(recommendations) > 1 else []
                
                # Calculate component scores
                component_scores = self._calculate_component_scores(best_outfit, enhanced_context)
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Create result
                result = RecommendationResult(
                    query=query,
                    context=enhanced_context,
                    outfit=best_outfit,
                    score=best_score,
                    explanation=best_explanation,
                    alternatives=alternatives,
                    processing_time=processing_time,
                    component_scores=component_scores,
                    metadata={
                        'total_candidates': len(outfit_candidates),
                        'filtered_candidates': len(filtered_outfits),
                        'ranking_method': ranking_method.value,
                        'context_confidence': enhanced_context.confidence
                    }
                )
                
                return result
            
            else:
                # No recommendations found
                return self._create_empty_result(query, enhanced_context, start_time)
                
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            return self._create_error_result(query, start_time, str(e))
    
    def _filter_outfits_by_context(self, 
                                  outfits: List[Outfit], 
                                  context: ParsedContext) -> List[Outfit]:
        """Filter outfits based on context constraints"""
        filtered_outfits = []
        
        for outfit in outfits:
            if self._outfit_matches_context(outfit, context):
                filtered_outfits.append(outfit)
        
        return filtered_outfits
    
    def _outfit_matches_context(self, outfit: Outfit, context: ParsedContext) -> bool:
        """Check if outfit matches context constraints"""
        # Color constraints
        if context.colors:
            outfit_colors = []
            for item in outfit.items:
                # Extract colors from item tags or use color extractor if available
                if hasattr(item, 'color') and item.color:
                    outfit_colors.append(item.color.lower())
            
            if context.colors:
                # Check if any context colors match outfit colors
                color_match = any(
                    context_color in outfit_colors 
                    for context_color in context.colors
                )
                if not color_match:
                    return False
        
        # Pattern constraints
        if context.patterns:
            outfit_patterns = []
            for item in outfit.items:
                if hasattr(item, 'pattern') and item.pattern:
                    outfit_patterns.append(item.pattern.lower())
            
            if context.patterns:
                # Check if patterns match (or if solid is requested and outfit has solid)
                pattern_match = any(
                    context_pattern in outfit_patterns 
                    for context_pattern in context.patterns
                )
                if not pattern_match:
                    return False
        
        # Style constraints
        if context.style_level:
            # Check outfit formality matches requested style
            if hasattr(outfit, 'formality_score'):
                style_mapping = {
                    'very_casual': (0.0, 0.3),
                    'casual': (0.2, 0.5),
                    'smart_casual': (0.4, 0.6),
                    'business_casual': (0.5, 0.7),
                    'business': (0.6, 0.8),
                    'formal': (0.7, 0.9),
                    'very_formal': (0.8, 1.0),
                    'cocktail': (0.6, 0.8),
                    'party': (0.4, 0.7)
                }
                
                style_range = style_mapping.get(context.style_level.value, (0.0, 1.0))
                if not (style_range[0] <= outfit.formality_score <= style_range[1]):
                    return False
        
        # Temperature constraints
        if context.temperature_range:
            # Check if outfit is appropriate for temperature
            temp_min, temp_max = context.temperature_range
            
            # Warm weather clothing
            if temp_max > 25:  # Hot
                # Avoid heavy fabrics, dark colors
                if hasattr(outfit, 'has_heavy_fabric') and outfit.has_heavy_fabric:
                    return False
            
            # Cold weather clothing
            if temp_min < 10:  # Cold
                # Require warm layers
                if hasattr(outfit, 'has_warm_layers') and not outfit.has_warm_layers:
                    return False
        
        return True
    
    def _create_user_preferences(self, context: ParsedContext) -> Dict[str, float]:
        """Create user preferences from context"""
        preferences = {}
        
        # Style preferences
        if context.style_level:
            style_weights = {
                'very_casual': {'style_compatibility': 0.8, 'comfort': 0.9},
                'casual': {'style_compatibility': 0.7, 'comfort': 0.8},
                'smart_casual': {'style_compatibility': 0.9, 'color_harmony': 0.8},
                'business': {'style_compatibility': 0.9, 'formality': 0.9},
                'formal': {'style_compatibility': 0.9, 'color_harmony': 0.9},
                'party': {'style_compatibility': 0.8, 'visual_appeal': 0.9}
            }
            
            style_prefs = style_weights.get(context.style_level.value, {})
            preferences.update(style_prefs)
        
        # Weather preferences
        if context.weather:
            if context.weather.value in ['hot', 'sunny']:
                preferences['comfort'] = 0.9
                preferences['breathability'] = 0.8
            elif context.weather.value in ['cold', 'snowy']:
                preferences['warmth'] = 0.9
                preferences['layering'] = 0.8
            elif context.weather.value in ['rainy']:
                preferences['water_resistance'] = 0.8
        
        # Occasion preferences
        if context.occasion:
            occasion_weights = {
                'business': {'formality': 0.9, 'professionalism': 0.9},
                'casual': {'comfort': 0.8, 'relaxed': 0.7},
                'formal': {'elegance': 0.9, 'sophistication': 0.9},
                'party': {'visual_appeal': 0.9},
                'sports': {'comfort': 0.9, 'functionality': 0.9},
                'date': {'style_compatibility': 0.8, 'visual_appeal': 0.8}
            }
            
            occasion_prefs = occasion_weights.get(context.occasion.value, {})
            preferences.update(occasion_prefs)
        
        return preferences
    
    def _context_to_dict(self, context: ParsedContext) -> Dict:
        """Convert context to dictionary for explainability engine"""
        return {
            'occasion': context.occasion.value if context.occasion else None,
            'season': context.season.value if context.season else None,
            'weather': context.weather.value if context.weather else None,
            'time_of_day': context.time_of_day.value if context.time_of_day else None,
            'style': context.style_level.value if context.style_level else None,
            'colors': context.colors,
            'patterns': context.patterns,
            'clothing_types': context.clothing_types,
            'formality': context.style_level.value if context.style_level else 'casual'
        }
    
    def _calculate_component_scores(self, outfit: Outfit, context: ParsedContext) -> Dict[str, float]:
        """Calculate individual component scores"""
        scores = {}
        
        # Style compatibility score
        if hasattr(outfit, 'style_score'):
            scores['style_compatibility'] = outfit.style_score
        
        # Color harmony score
        if hasattr(outfit, 'color_score'):
            scores['color_harmony'] = outfit.color_score
        
        # Pattern balance score
        if hasattr(outfit, 'pattern_score'):
            scores['pattern_balance'] = outfit.pattern_score
        
        # Formality consistency score
        if hasattr(outfit, 'formality_score'):
            scores['formality_consistency'] = outfit.formality_score
        
        # Visual appeal score
        if hasattr(outfit, 'visual_appeal'):
            scores['visual_appeal'] = outfit.visual_appeal
        
        # Context matching score
        context_score = self._calculate_context_score(outfit, context)
        scores['context_match'] = context_score
        
        return scores
    
    def _calculate_context_score(self, outfit: Outfit, context: ParsedContext) -> float:
        """Calculate how well outfit matches context"""
        score = 0.0
        max_score = 0.0
        
        # Color matching
        if context.colors:
            max_score += 0.3
            outfit_colors = [getattr(item, 'color', 'unknown') for item in outfit.items]
            color_match = any(
                context_color in [c.lower() for c in outfit_colors]
                for context_color in context.colors
            )
            if color_match:
                score += 0.3
        
        # Pattern matching
        if context.patterns:
            max_score += 0.2
            outfit_patterns = [getattr(item, 'pattern', 'solid') for item in outfit.items]
            pattern_match = any(
                context_pattern in [p.lower() for p in outfit_patterns]
                for context_pattern in context.patterns
            )
            if pattern_match:
                score += 0.2
        
        # Style matching
        if context.style_level and hasattr(outfit, 'formality_score'):
            max_score += 0.3
            style_mapping = {
                'very_casual': 0.2,
                'casual': 0.4,
                'smart_casual': 0.6,
                'business_casual': 0.7,
                'business': 0.8,
                'formal': 0.9,
                'very_formal': 1.0
            }
            target_formality = style_mapping.get(context.style_level.value, 0.5)
            formality_diff = abs(outfit.formality_score - target_formality)
            if formality_diff < 0.2:
                score += 0.3
            elif formality_diff < 0.4:
                score += 0.15
        
        # Weather appropriateness
        if context.weather:
            max_score += 0.2
            # Simplified weather scoring
            if context.weather.value in ['hot', 'sunny']:
                # Prefer lighter colors and breathable fabrics
                if hasattr(outfit, 'has_light_colors') and outfit.has_light_colors:
                    score += 0.2
            elif context.weather.value in ['cold', 'snowy']:
                # Prefer warmer clothing
                if hasattr(outfit, 'has_warm_layers') and outfit.has_warm_layers:
                    score += 0.2
        
        return score / max_score if max_score > 0 else 0.0
    
    def _create_empty_result(self, query: str, context: ParsedContext, start_time: datetime) -> RecommendationResult:
        """Create empty result when no recommendations found"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return RecommendationResult(
            query=query,
            context=context,
            outfit=None,
            score=0.0,
            explanation={'error': 'No suitable outfits found for the given context'},
            alternatives=[],
            processing_time=processing_time,
            component_scores={},
            metadata={'status': 'no_results', 'context_confidence': context.confidence}
        )
    
    def _create_error_result(self, query: str, start_time: datetime, error: str) -> RecommendationResult:
        """Create error result"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return RecommendationResult(
            query=query,
            context=None,
            outfit=None,
            score=0.0,
            explanation={'error': error},
            alternatives=[],
            processing_time=processing_time,
            component_scores={},
            metadata={'status': 'error', 'error_message': error}
        )
    
    def batch_process_queries(self, queries: List[str]) -> List[RecommendationResult]:
        """Process multiple queries"""
        results = []
        
        print(f"Processing {len(queries)} queries...")
        
        for i, query in enumerate(queries):
            try:
                result = self.process_query(query)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(queries)} queries")
                    
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                continue
        
        print(f"Completed processing {len(results)} queries")
        return results
    
    def export_result(self, result: RecommendationResult, output_path: str):
        """Export recommendation result to JSON"""
        result_dict = {
            'query': result.query,
            'context': {
                'original_query': result.context.original_query if result.context else None,
                'occasion': result.context.occasion.value if result.context and result.context.occasion else None,
                'season': result.context.season.value if result.context and result.context.season else None,
                'weather': result.context.weather.value if result.context and result.context.weather else None,
                'time_of_day': result.context.time_of_day.value if result.context and result.context.time_of_day else None,
                'style_level': result.context.style_level.value if result.context and result.context.style_level else None,
                'colors': result.context.colors if result.context else [],
                'patterns': result.context.patterns if result.context else [],
                'confidence': result.context.confidence if result.context else 0.0
            },
            'outfit': {
                'items': [item.path for item in result.outfit.items] if result.outfit else [],
                'score': result.outfit.score if result.outfit else 0.0,
                'style_score': getattr(result.outfit, 'style_score', 0.0) if result.outfit else 0.0,
                'color_score': getattr(result.outfit, 'color_score', 0.0) if result.outfit else 0.0,
                'pattern_score': getattr(result.outfit, 'pattern_score', 0.0) if result.outfit else 0.0,
                'formality_score': getattr(result.outfit, 'formality_score', 0.0) if result.outfit else 0.0
            },
            'recommendation_score': result.score,
            'explanation': result.explanation,
            'alternatives': result.alternatives,
            'processing_time': result.processing_time,
            'component_scores': result.component_scores,
            'metadata': result.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"Result exported to: {output_path}")
    
    def get_engine_statistics(self) -> Dict:
        """Get integration engine statistics"""
        stats = {
            'components': {
                'storage_items': len(self.storage.get_all_embeddings()),
                'rule_base_rules': len(self.rule_base.rules),
                'embedding_index_size': self.embedding_index.get_index_statistics().get('total_rules', 0),
                'outfit_builder_items': len(self.outfit_builder.items_cache.get('top', [])) + 
                                      len(self.outfit_builder.items_cache.get('bottom', [])) + 
                                      len(self.outfit_builder.items_cache.get('shoes', []))
            },
            'capabilities': {
                'context_parsing': True,
                'outfit_building': True,
                'rule_application': True,
                'explainability': True,
                'smart_tags': True,
                'similarity_search': True
            },
            'performance': {
                'index_loaded': self.embedding_index.index is not None,
                'cache_populated': bool(self.outfit_builder.items_cache)
            }
        }
        
        return stats


# Test function
def test_integration_engine():
    """Test the integration engine"""
    print("Testing Integration Engine...")
    
    # Create integration engine
    engine = IntegrationEngine()
    
    # Test queries
    test_queries = [
        "summer casual day outfit",
        "business formal meeting attire", 
        "winter cold weather clothes",
        "date night elegant dress"
    ]
    
    print(f"Testing {len(test_queries)} queries...")
    
    for query in test_queries:
        print(f"\nProcessing: {query}")
        
        try:
            result = engine.process_query(query, max_outfits=3)
            
            print(f"  Score: {result.score:.3f}")
            print(f"  Processing time: {result.processing_time:.2f}s")
            print(f"  Context confidence: {result.context.confidence:.2f}")
            print(f"  Outfit items: {len(result.outfit.items) if result.outfit else 0}")
            
            if result.explanation and 'primary_reason' in result.explanation:
                print(f"  Explanation: {result.explanation['primary_reason']}")
            
            if result.component_scores:
                print(f"  Component scores: {result.component_scores}")
            
            # Export result
            output_path = f"test_result_{hash(query) % 10000}.json"
            engine.export_result(result, output_path)
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Get engine statistics
    stats = engine.get_engine_statistics()
    print(f"\nEngine statistics: {stats}")
    
    print("Integration engine test completed!")


if __name__ == "__main__":
    test_integration_engine()
