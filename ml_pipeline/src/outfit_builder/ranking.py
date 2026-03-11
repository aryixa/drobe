"""
Outfit ranking algorithms and scoring systems
Advanced ranking methods for outfit combinations
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import json
from collections import defaultdict

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import EMBEDDING_DIM, SIMILARITY_THRESHOLD
from src.outfit_builder.core_engine import Outfit, ClothingItem, ClothingType, OutfitStyle


class RankingMethod(Enum):
    """Ranking algorithm types"""
    WEIGHTED_SUM = "weighted_sum"
    SIMILARITY_BASED = "similarity_based"
    LEARNING_BASED = "learning_based"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"


@dataclass
class RankingWeights:
    """Weights for different ranking criteria"""
    style_compatibility: float = 0.3
    color_harmony: float = 0.25
    pattern_balance: float = 0.15
    formality_consistency: float = 0.2
    visual_appeal: float = 0.1
    
    def normalize(self):
        """Normalize weights to sum to 1.0"""
        total = sum([self.style_compatibility, self.color_harmony, 
                    self.pattern_balance, self.formality_consistency, self.visual_appeal])
        if total > 0:
            self.style_compatibility /= total
            self.color_harmony /= total
            self.pattern_balance /= total
            self.formality_consistency /= total
            self.visual_appeal /= total


@dataclass
class RankingResult:
    """Result of outfit ranking"""
    outfit: Outfit
    score: float
    rank: int
    sub_scores: Dict[str, float]
    explanation: str


class OutfitRanker:
    """
    Advanced outfit ranking system
    - Multiple ranking algorithms
    - Customizable weights
    - Detailed scoring breakdown
    """
    
    def __init__(self, 
                 method: RankingMethod = RankingMethod.WEIGHTED_SUM,
                 weights: Optional[RankingWeights] = None):
        self.method = method
        self.weights = weights or RankingWeights()
        self.weights.normalize()
        
        # Scoring functions
        self.scoring_functions = {
            'style_compatibility': self._score_style_compatibility,
            'color_harmony': self._score_color_harmony,
            'pattern_balance': self._score_pattern_balance,
            'formality_consistency': self._score_formality_consistency,
            'visual_appeal': self._score_visual_appeal
        }
        
        # Color harmony rules
        self.color_harmony_rules = self._init_color_harmony_rules()
        
        # Pattern compatibility rules
        self.pattern_compatibility = self._init_pattern_compatibility()
        
        print(f"Outfit Ranker initialized")
        print(f"Method: {method.value}")
        print(f"Weights: {self.weights}")
    
    def _init_color_harmony_rules(self) -> Dict[str, List[str]]:
        """Initialize color harmony rules"""
        return {
            'monochromatic': ['black', 'white', 'gray'],
            'complementary': ['blue', 'orange', 'red', 'green', 'yellow', 'purple'],
            'analogous': ['blue', 'purple', 'green'],
            'neutral_base': ['black', 'white', 'gray', 'brown', 'beige']
        }
    
    def _init_pattern_compatibility(self) -> Dict[str, List[str]]:
        """Initialize pattern compatibility rules"""
        return {
            'solid': ['solid', 'striped', 'dotted'],
            'striped': ['solid', 'plain'],
            'dotted': ['solid', 'plain'],
            'floral': ['solid'],
            'geometric': ['solid'],
            'plaid': ['solid'],
            'animal': ['solid']
        }
    
    def rank_outfits(self, 
                    outfits: List[Outfit],
                    reference_outfit: Optional[Outfit] = None,
                    user_preferences: Optional[Dict] = None) -> List[RankingResult]:
        """
        Rank outfits using selected method
        
        Args:
            outfits: List of outfits to rank
            reference_outfit: Reference outfit for similarity-based ranking
            user_preferences: User preference weights
            
        Returns:
            List of ranking results
        """
        if not outfits:
            return []
        
        print(f"Ranking {len(outfits)} outfits using {self.method.value}...")
        
        # Apply ranking method
        if self.method == RankingMethod.WEIGHTED_SUM:
            results = self._rank_weighted_sum(outfits, user_preferences)
        elif self.method == RankingMethod.SIMILARITY_BASED:
            results = self._rank_similarity_based(outfits, reference_outfit)
        elif self.method == RankingMethod.RULE_BASED:
            results = self._rank_rule_based(outfits)
        elif self.method == RankingMethod.HYBRID:
            results = self._rank_hybrid(outfits, reference_outfit, user_preferences)
        else:
            # Default to weighted sum
            results = self._rank_weighted_sum(outfits, user_preferences)
        
        # Sort by score and assign ranks
        results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
        
        # Generate explanations
        for result in results:
            result.explanation = self._generate_explanation(result)
        
        print(f"Ranking completed. Top score: {results[0].score:.3f}")
        return results
    
    def _rank_weighted_sum(self, 
                          outfits: List[Outfit], 
                          user_preferences: Optional[Dict] = None) -> List[RankingResult]:
        """Rank outfits using weighted sum of criteria"""
        results = []
        
        # Apply user preferences if provided
        weights = self._apply_user_preferences(user_preferences)
        
        for outfit in outfits:
            sub_scores = {}
            total_score = 0.0
            
            # Calculate each criterion score
            for criterion, weight in [
                ('style_compatibility', weights.style_compatibility),
                ('color_harmony', weights.color_harmony),
                ('pattern_balance', weights.pattern_balance),
                ('formality_consistency', weights.formality_consistency),
                ('visual_appeal', weights.visual_appeal)
            ]:
                score = self.scoring_functions[criterion](outfit)
                sub_scores[criterion] = score
                total_score += score * weight
            
            result = RankingResult(
                outfit=outfit,
                score=total_score,
                rank=0,  # Will be assigned later
                sub_scores=sub_scores,
                explanation=""
            )
            results.append(result)
        
        return results
    
    def _rank_similarity_based(self, 
                              outfits: List[Outfit], 
                              reference_outfit: Optional[Outfit]) -> List[RankingResult]:
        """Rank outfits based on similarity to reference outfit"""
        if reference_outfit is None or reference_outfit.embedding is None:
            # Fall back to weighted sum
            return self._rank_weighted_sum(outfits)
        
        results = []
        
        for outfit in outfits:
            if outfit.embedding is None:
                continue
            
            # Calculate similarity to reference
            similarity = np.dot(reference_outfit.embedding, outfit.embedding)
            
            # Calculate sub-scores for explanation
            sub_scores = {}
            for criterion, func in self.scoring_functions.items():
                sub_scores[criterion] = func(outfit)
            
            result = RankingResult(
                outfit=outfit,
                score=similarity,
                rank=0,
                sub_scores=sub_scores,
                explanation=""
            )
            results.append(result)
        
        return results
    
    def _rank_rule_based(self, outfits: List[Outfit]) -> List[RankingResult]:
        """Rank outfits based on fashion rules"""
        results = []
        
        for outfit in outfits:
            score = 0.0
            sub_scores = {}
            
            # Apply fashion rules
            rules_score = self._apply_fashion_rules(outfit)
            sub_scores['fashion_rules'] = rules_score
            
            # Basic compatibility scores
            for criterion, func in self.scoring_functions.items():
                sub_scores[criterion] = func(outfit)
            
            # Weight rule-based scoring more heavily
            total_score = rules_score * 0.6 + sum(sub_scores.values()) * 0.4 / len(sub_scores)
            
            result = RankingResult(
                outfit=outfit,
                score=total_score,
                rank=0,
                sub_scores=sub_scores,
                explanation=""
            )
            results.append(result)
        
        return results
    
    def _rank_hybrid(self, 
                    outfits: List[Outfit], 
                    reference_outfit: Optional[Outfit],
                    user_preferences: Optional[Dict] = None) -> List[RankingResult]:
        """Hybrid ranking combining multiple methods"""
        # Get results from different methods
        weighted_results = self._rank_weighted_sum(outfits, user_preferences)
        similarity_results = self._rank_similarity_based(outfits, reference_outfit)
        rule_results = self._rank_rule_based(outfits)
        
        # Combine results
        combined_results = []
        
        for i, outfit in enumerate(outfits):
            # Find corresponding results
            weighted_score = weighted_results[i].score if i < len(weighted_results) else 0
            similarity_score = similarity_results[i].score if i < len(similarity_results) else 0
            rule_score = rule_results[i].score if i < len(rule_results) else 0
            
            # Weighted combination
            combined_score = (
                weighted_score * 0.4 + 
                similarity_score * 0.3 + 
                rule_score * 0.3
            )
            
            # Combine sub-scores
            sub_scores = {}
            for key in weighted_results[i].sub_scores:
                sub_scores[key] = weighted_results[i].sub_scores[key]
            
            sub_scores['similarity'] = similarity_score
            sub_scores['rule_based'] = rule_score
            
            result = RankingResult(
                outfit=outfit,
                score=combined_score,
                rank=0,
                sub_scores=sub_scores,
                explanation=""
            )
            combined_results.append(result)
        
        return combined_results
    
    def _score_style_compatibility(self, outfit: Outfit) -> float:
        """Score style compatibility of outfit items"""
        styles = [item.style for item in outfit.items if item.style]
        
        if len(styles) <= 1:
            return 1.0
        
        # Count style frequencies
        style_counts = defaultdict(int)
        for style in styles:
            style_counts[style.value] += 1
        
        # Calculate compatibility based on style mixing rules
        compatible_combinations = {
            (OutfitStyle.CASUAL, OutfitStyle.STREET): 0.8,
            (OutfitStyle.BUSINESS, OutfitStyle.FORMAL): 0.9,
            (OutfitStyle.CASUAL, OutfitStyle.SPORTS): 0.7,
            (OutfitStyle.DATE, OutfitStyle.PARTY): 0.8,
        }
        
        max_score = 0.0
        
        for i, style1 in enumerate(styles):
            for j, style2 in enumerate(styles[i+1:], i+1):
                if style1 == style2:
                    score = 1.0
                else:
                    score = compatible_combinations.get((style1, style2), 0.3)
                    score = compatible_combinations.get((style2, style1), score)
                
                max_score = max(max_score, score)
        
        return max_score
    
    def _score_color_harmony(self, outfit: Outfit) -> float:
        """Score color harmony in outfit"""
        colors = [item.color for item in outfit.items if item.color]
        
        if len(colors) <= 1:
            return 1.0
        
        # Check for color harmony patterns
        harmony_score = 0.0
        
        # Monochromatic (all neutral/base colors)
        neutral_colors = self.color_harmony_rules['neutral_base']
        if all(color in neutral_colors for color in colors):
            harmony_score = 0.9
        
        # Complementary colors
        complementary_pairs = [
            ('blue', 'orange'), ('red', 'green'), ('yellow', 'purple')
        ]
        for pair in complementary_pairs:
            if all(color in colors for color in pair):
                harmony_score = max(harmony_score, 0.8)
        
        # Analogous colors
        analogous_groups = [
            ['blue', 'purple'], ['green', 'blue'], ['yellow', 'orange']
        ]
        for group in analogous_groups:
            if all(color in colors for color in group):
                harmony_score = max(harmony_score, 0.7)
        
        # Neutral base with accent color
        neutral_count = sum(1 for color in colors if color in neutral_colors)
        if neutral_count >= 2 and len(colors) - neutral_count == 1:
            harmony_score = max(harmony_score, 0.85)
        
        # Default score based on color variety
        unique_colors = len(set(colors))
        if harmony_score == 0.0:
            harmony_score = max(0.3, 1.0 - (unique_colors - 1) * 0.2)
        
        return harmony_score
    
    def _score_pattern_balance(self, outfit: Outfit) -> float:
        """Score pattern balance in outfit"""
        patterns = [item.pattern for item in outfit.items if item.pattern]
        
        if len(patterns) <= 1:
            return 1.0
        
        # Pattern mixing rules
        solid_count = patterns.count('solid')
        pattern_count = len(patterns) - solid_count
        
        # Best: 0-1 patterned items
        if pattern_count == 0:
            return 1.0
        elif pattern_count == 1:
            return 0.9
        elif pattern_count == 2:
            # Check if patterns are compatible
            compatible_pairs = [
                ('striped', 'solid'), ('dotted', 'solid'), ('geometric', 'solid')
            ]
            for pair in compatible_pairs:
                if all(pattern in patterns for pattern in pair):
                    return 0.7
            return 0.3
        else:
            return 0.1  # Too many patterns
    
    def _score_formality_consistency(self, outfit: Outfit) -> float:
        """Score formality consistency in outfit"""
        formalities = [item.formality for item in outfit.items if item.formality is not None]
        
        if len(formalities) <= 1:
            return 1.0
        
        # Calculate standard deviation
        formality_std = np.std(formalities)
        
        # Lower standard deviation = higher consistency
        consistency_score = max(0.0, 1.0 - formality_std * 2)
        
        return consistency_score
    
    def _score_visual_appeal(self, outfit: Outfit) -> float:
        """Score overall visual appeal (heuristic)"""
        # Combine other scores with additional heuristics
        style_score = self._score_style_compatibility(outfit)
        color_score = self._score_color_harmony(outfit)
        pattern_score = self._score_pattern_balance(outfit)
        
        # Visual appeal is weighted combination
        appeal_score = (
            style_score * 0.4 + 
            color_score * 0.4 + 
            pattern_score * 0.2
        )
        
        # Bonus for complete outfits
        if outfit.has_complete_set():
            appeal_score = min(1.0, appeal_score + 0.1)
        
        return appeal_score
    
    def _apply_fashion_rules(self, outfit: Outfit) -> float:
        """Apply fashion rules to score outfit"""
        rules_score = 0.0
        total_rules = 0
        
        # Rule 1: Don't wear too many colors
        colors = [item.color for item in outfit.items if item.color]
        if len(set(colors)) <= 3:
            rules_score += 1.0
        total_rules += 1
        
        # Rule 2: Match belt with shoes (simplified)
        # This would require more detailed item classification
        
        # Rule 3: Occasion appropriateness
        styles = [item.style for item in outfit.items if item.style]
        if len(set(styles)) == 1:
            rules_score += 0.8
        elif len(set(styles)) == 2:
            rules_score += 0.6
        total_rules += 1
        
        # Rule 4: Season appropriateness
        seasons = [item.season for item in outfit.items if item.season]
        if len(set(seasons)) <= 1:
            rules_score += 0.7
        total_rules += 1
        
        return rules_score / total_rules if total_rules > 0 else 0.0
    
    def _apply_user_preferences(self, user_preferences: Optional[Dict]) -> RankingWeights:
        """Apply user preferences to weights"""
        if not user_preferences:
            return self.weights
        
        new_weights = RankingWeights()
        
        # Update weights based on preferences
        for key, value in user_preferences.items():
            if hasattr(new_weights, key):
                setattr(new_weights, key, value)
        
        new_weights.normalize()
        return new_weights
    
    def _generate_explanation(self, result: RankingResult) -> str:
        """Generate explanation for ranking result"""
        explanations = []
        
        # Find strongest and weakest criteria
        if result.sub_scores:
            sorted_scores = sorted(result.sub_scores.items(), key=lambda x: x[1], reverse=True)
            
            strongest = sorted_scores[0]
            weakest = sorted_scores[-1]
            
            if strongest[1] > 0.7:
                explanations.append(f"Excellent {strongest[0].replace('_', ' ')}")
            
            if weakest[1] < 0.4:
                explanations.append(f"Could improve {weakest[0].replace('_', ' ')}")
        
        # Add overall assessment
        if result.score > 0.8:
            explanations.append("Excellent outfit choice!")
        elif result.score > 0.6:
            explanations.append("Good outfit combination")
        elif result.score > 0.4:
            explanations.append("Decent outfit with room for improvement")
        else:
            explanations.append("Consider different combinations")
        
        return ". ".join(explanations)
    
    def compare_rankings(self, 
                        outfits: List[Outfit],
                        methods: List[RankingMethod]) -> Dict[str, List[RankingResult]]:
        """Compare different ranking methods"""
        comparison = {}
        
        for method in methods:
            # Temporarily change method
            original_method = self.method
            self.method = method
            
            # Rank outfits
            results = self.rank_outfits(outfits)
            comparison[method.value] = results
            
            # Restore original method
            self.method = original_method
        
        return comparison
    
    def get_ranking_statistics(self, results: List[RankingResult]) -> Dict:
        """Get statistics about ranking results"""
        if not results:
            return {}
        
        scores = [result.score for result in results]
        
        stats = {
            'total_outfits': len(results),
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'score_distribution': {
                'excellent': len([s for s in scores if s > 0.8]),
                'good': len([s for s in scores if 0.6 < s <= 0.8]),
                'fair': len([s for s in scores if 0.4 < s <= 0.6]),
                'poor': len([s for s in scores if s <= 0.4])
            }
        }
        
        # Sub-score statistics
        if results[0].sub_scores:
            for criterion in results[0].sub_scores.keys():
                criterion_scores = [result.sub_scores[criterion] for result in results]
                stats[f'avg_{criterion}'] = np.mean(criterion_scores)
        
        return stats


# Test function
def test_outfit_ranking():
    """Test outfit ranking system"""
    print("Testing Outfit Ranking...")
    
    # Create dummy outfits
    from src.outfit_builder.core_engine import ClothingItem, ClothingType, OutfitStyle
    
    np.random.seed(42)
    
    # Create dummy items
    items = [
        ClothingItem("top1.jpg", np.random.rand(512), ClothingType.TOP, style=OutfitStyle.CASUAL, color="blue", pattern="solid", formality=0.3),
        ClothingItem("bottom1.jpg", np.random.rand(512), ClothingType.BOTTOM, style=OutfitStyle.CASUAL, color="blue", pattern="solid", formality=0.4),
        ClothingItem("shoes1.jpg", np.random.rand(512), ClothingType.SHOES, style=OutfitStyle.CASUAL, color="white", pattern="solid", formality=0.2),
        ClothingItem("top2.jpg", np.random.rand(512), ClothingType.TOP, style=OutfitStyle.FORMAL, color="white", pattern="solid", formality=0.8),
        ClothingItem("bottom2.jpg", np.random.rand(512), ClothingType.BOTTOM, style=OutfitStyle.FORMAL, color="black", pattern="solid", formality=0.9),
        ClothingItem("shoes2.jpg", np.random.rand(512), ClothingType.SHOES, style=OutfitStyle.FORMAL, color="black", pattern="solid", formality=0.7),
    ]
    
    # Create outfits
    outfits = [
        Outfit(items=[items[0], items[1], items[2]]),  # Casual outfit
        Outfit(items=[items[3], items[4], items[5]]),  # Formal outfit
        Outfit(items=[items[0], items[4], items[2]]),  # Mixed outfit
    ]
    
    # Test different ranking methods
    methods = [RankingMethod.WEIGHTED_SUM, RankingMethod.RULE_BASED, RankingMethod.HYBRID]
    
    for method in methods:
        print(f"\nTesting {method.value}...")
        
        ranker = OutfitRanker(method=method)
        results = ranker.rank_outfits(outfits)
        
        print(f"Top 3 rankings:")
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. Score: {result.score:.3f} - {result.explanation}")
        
        # Get statistics
        stats = ranker.get_ranking_statistics(results)
        print(f"Statistics: {stats}")
    
    print("Outfit ranking test completed!")


if __name__ == "__main__":
    test_outfit_ranking()
