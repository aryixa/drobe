"""
Task 13 - RAG + Explainability System: Explainability
LLM integration for generating explanations and showing WHY suggestions were made
"""

import json
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from .rule_base import RuleBase, FashionRule, RuleCategory
from .embedding_index import EmbeddingIndex


class ExplanationType(Enum):
    """Types of explanations"""
    RULE_BASED = "rule_based"
    SIMILARITY_BASED = "similarity_based"
    CONTEXT_AWARE = "context_aware"
    COMPREHENSIVE = "comprehensive"


@dataclass
class Explanation:
    """Explanation structure for outfit recommendations"""
    recommendation_id: str
    explanation_type: ExplanationType
    primary_reason: str
    supporting_rules: List[Dict]
    confidence_score: float
    context_factors: Dict[str, str]
    alternatives: List[str]
    additional_tips: List[str]
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert explanation to dictionary"""
        return {
            'recommendation_id': self.recommendation_id,
            'explanation_type': self.explanation_type.value,
            'primary_reason': self.primary_reason,
            'supporting_rules': self.supporting_rules,
            'confidence_score': self.confidence_score,
            'context_factors': self.context_factors,
            'alternatives': self.alternatives,
            'additional_tips': self.additional_tips,
            'timestamp': self.timestamp
        }


class ExplainabilityEngine:
    """
    Explainability system for outfit recommendations
    - Rule-based explanations
    - Context-aware reasoning
    - LLM integration for natural language explanations
    """
    
    def __init__(self, 
                 rule_base: RuleBase,
                 embedding_index: EmbeddingIndex):
        self.rule_base = rule_base
        self.embedding_index = embedding_index
        
        # Explanation templates
        self.templates = self._init_templates()
        
        print("Explainability Engine initialized")
    
    def _init_templates(self) -> Dict[str, str]:
        """Initialize explanation templates"""
        return {
            'color_harmony': "This outfit follows {color_scheme} color principles, creating {harmony_type} harmony.",
            'pattern_mixing': "The pattern combination works because {pattern_reason}.",
            'style_coordination': "This style is appropriate because {style_reason}.",
            'occasion_appropriate': "Perfect for {occasion} as it meets {occasion_requirements}.",
            'seasonal_guidelines': "Ideal for {season} weather with {season_features}.",
            'confidence_high': "This recommendation is highly suitable with {confidence}% confidence.",
            'confidence_medium': "This recommendation should work well with {confidence}% confidence.",
            'confidence_low': "This recommendation might work with {confidence}% confidence.",
            'alternative_suggestion': "Consider {alternative} as an alternative option."
        }
    
    def generate_explanation(self, 
                            recommendation: Dict,
                            context: Dict[str, Union[str, List[str]]],
                            explanation_type: ExplanationType = ExplanationType.COMPREHENSIVE) -> Explanation:
        """
        Generate explanation for outfit recommendation
        
        Args:
            recommendation: Recommendation dictionary
            context: Outfit context
            explanation_type: Type of explanation to generate
            
        Returns:
            Explanation object
        """
        # Extract recommendation details
        outfit_items = recommendation.get('items', [])
        score = recommendation.get('score', 0.0)
        recommendation_id = recommendation.get('id', f"rec_{datetime.now().timestamp()}")
        
        # Get relevant rules
        relevant_rules = self.embedding_index.get_rule_recommendations(
            context, self.rule_base, top_k=5
        )
        
        # Generate primary reason
        primary_reason = self._generate_primary_reason(relevant_rules, context, score)
        
        # Extract context factors
        context_factors = self._extract_context_factors(context)
        
        # Generate supporting rules
        supporting_rules = self._format_supporting_rules(relevant_rules)
        
        # Generate alternatives
        alternatives = self._generate_alternatives(relevant_rules, context)
        
        # Generate additional tips
        additional_tips = self._generate_additional_tips(relevant_rules, context)
        
        # Calculate confidence
        confidence_score = self._calculate_confidence(score, relevant_rules)
        
        # Create explanation
        explanation = Explanation(
            recommendation_id=recommendation_id,
            explanation_type=explanation_type,
            primary_reason=primary_reason,
            supporting_rules=supporting_rules,
            confidence_score=confidence_score,
            context_factors=context_factors,
            alternatives=alternatives,
            additional_tips=additional_tips,
            timestamp=datetime.now().isoformat()
        )
        
        return explanation
    
    def _generate_primary_reason(self, 
                               relevant_rules: List[Dict], 
                               context: Dict, 
                               score: float) -> str:
        """Generate primary reason for recommendation"""
        if not relevant_rules:
            return "This outfit combination follows basic fashion principles and should work well."
        
        # Get top rule
        top_rule = relevant_rules[0]
        rule_category = top_rule.get('category', '')
        rule_title = top_rule.get('title', '')
        rule_description = top_rule.get('description', '')
        
        # Generate reason based on category
        if rule_category == 'color_harmony':
            colors = context.get('colors', [])
            if len(colors) == 1:
                return f"This outfit uses a monochromatic color scheme with {colors[0]}, creating a cohesive and elegant look."
            elif len(colors) == 2:
                return f"This outfit combines {colors[0]} and {colors[1]} in a harmonious color relationship."
            else:
                return f"This outfit coordinates multiple colors {', '.join(colors)} effectively."
        
        elif rule_category == 'pattern_mixing':
            patterns = context.get('patterns', [])
            if len(patterns) <= 1:
                return "This outfit keeps patterns minimal for a clean, sophisticated appearance."
            else:
                return f"The pattern combination of {', '.join(patterns)} creates visual interest while maintaining harmony."
        
        elif rule_category == 'style_coordination':
            style = context.get('style', '')
            occasion = context.get('occasion', '')
            return f"This {style} outfit is well-coordinated and appropriate for {occasion} occasions."
        
        elif rule_category == 'occasion_appropriate':
            occasion = context.get('occasion', '')
            return f"This outfit is perfectly suited for {occasion} occasions, meeting all appropriate dress code requirements."
        
        elif rule_category == 'seasonal_guidelines':
            season = context.get('season', '')
            return f"This outfit is ideal for {season} weather with appropriate fabric choices and layering."
        
        else:
            return f"This recommendation follows {rule_title.lower()}, ensuring {rule_description.lower()}."
    
    def _extract_context_factors(self, context: Dict) -> Dict[str, str]:
        """Extract and format context factors"""
        factors = {}
        
        if 'occasion' in context:
            factors['occasion'] = f"Designed for {context['occasion']} occasions"
        
        if 'season' in context:
            factors['season'] = f"Optimized for {context['season']} weather"
        
        if 'colors' in context and context['colors']:
            factors['colors'] = f"Color palette: {', '.join(context['colors'])}"
        
        if 'patterns' in context and context['patterns']:
            factors['patterns'] = f"Pattern scheme: {', '.join(context['patterns'])}"
        
        if 'style' in context:
            factors['style'] = f"Style: {context['style']}"
        
        if 'formality' in context:
            factors['formality'] = f"Formality level: {context['formality']}"
        
        return factors
    
    def _format_supporting_rules(self, relevant_rules: List[Dict]) -> List[Dict]:
        """Format supporting rules for explanation"""
        supporting_rules = []
        
        for rule in relevant_rules[:3]:  # Top 3 rules
            formatted_rule = {
                'id': rule.get('rule_id', ''),
                'title': rule.get('title', ''),
                'category': rule.get('category', ''),
                'priority': rule.get('priority', ''),
                'relevance': rule.get('relevance_score', 0.0),
                'key_recommendation': rule.get('recommendations', [''])[0] if rule.get('recommendations') else ''
            }
            supporting_rules.append(formatted_rule)
        
        return supporting_rules
    
    def _generate_alternatives(self, 
                               relevant_rules: List[Dict], 
                               context: Dict) -> List[str]:
        """Generate alternative suggestions"""
        alternatives = []
        
        # Extract alternatives from rules
        for rule in relevant_rules[:3]:
            rule_examples = rule.get('examples', [])
            if rule_examples:
                alternatives.extend(rule_examples[:1])  # Take one example per rule
        
        # Add context-based alternatives
        colors = context.get('colors', [])
        if colors and len(colors) > 1:
            alternatives.append(f"Try a monochromatic look using only {colors[0]}")
        
        patterns = context.get('patterns', [])
        if patterns and 'solid' not in patterns:
            alternatives.append("Consider adding a solid-colored piece to balance the patterns")
        
        # Limit to top alternatives
        return alternatives[:3]
    
    def _generate_additional_tips(self, 
                                 relevant_rules: List[Dict], 
                                 context: Dict) -> List[str]:
        """Generate additional styling tips"""
        tips = []
        
        # Extract tips from rules
        for rule in relevant_rules[:2]:
            rule_recommendations = rule.get('recommendations', [])
            tips.extend(rule_recommendations[:2])  # Take two tips per rule
        
        # Add general tips based on context
        occasion = context.get('occasion', '')
        if occasion == 'business':
            tips.append("Ensure all pieces are well-pressed and in excellent condition")
        elif occasion == 'casual':
            tips.append("Pay attention to fit - even casual clothes should fit well")
        
        season = context.get('season', '')
        if season == 'winter':
            tips.append("Consider layering pieces that can be removed indoors")
        elif season == 'summer':
            tips.append("Choose breathable fabrics to stay comfortable")
        
        # Remove duplicates and limit
        unique_tips = list(dict.fromkeys(tips))  # Remove duplicates while preserving order
        return unique_tips[:4]
    
    def _calculate_confidence(self, 
                             score: float, 
                             relevant_rules: List[Dict]) -> float:
        """Calculate overall confidence score"""
        # Base confidence from recommendation score
        base_confidence = min(1.0, score)
        
        # Boost confidence based on rule relevance
        if relevant_rules:
            avg_rule_relevance = sum(rule.get('relevance_score', 0.0) for rule in relevant_rules) / len(relevant_rules)
            rule_confidence = min(1.0, avg_rule_relevance)
        else:
            rule_confidence = 0.5
        
        # Combine confidences
        overall_confidence = (base_confidence * 0.6) + (rule_confidence * 0.4)
        
        return round(overall_confidence, 3)
    
    def explain_outfit_score(self, 
                           outfit: Dict, 
                           context: Dict) -> Dict:
        """
        Explain why an outfit received a specific score
        
        Args:
            outfit: Outfit dictionary with items and score
            context: Outfit context
            
        Returns:
            Detailed score explanation
        """
        score = outfit.get('score', 0.0)
        items = outfit.get('items', [])
        
        # Get relevant rules
        relevant_rules = self.embedding_index.get_rule_recommendations(
            context, self.rule_base, top_k=10
        )
        
        # Analyze score components
        score_breakdown = {
            'overall_score': score,
            'components': {},
            'strengths': [],
            'weaknesses': [],
            'improvement_suggestions': []
        }
        
        # Analyze color contribution
        colors = context.get('colors', [])
        if colors:
            color_rules = [r for r in relevant_rules if r.get('category') == 'color_harmony']
            if color_rules:
                avg_color_relevance = sum(r.get('relevance_score', 0.0) for r in color_rules) / len(color_rules)
                score_breakdown['components']['color_harmony'] = avg_color_relevance
                
                if avg_color_relevance > 0.8:
                    score_breakdown['strengths'].append("Excellent color coordination")
                elif avg_color_relevance < 0.5:
                    score_breakdown['weaknesses'].append("Color coordination could be improved")
                    score_breakdown['improvement_suggestions'].append("Consider a more harmonious color scheme")
        
        # Analyze pattern contribution
        patterns = context.get('patterns', [])
        if patterns:
            pattern_rules = [r for r in relevant_rules if r.get('category') == 'pattern_mixing']
            if pattern_rules:
                avg_pattern_relevance = sum(r.get('relevance_score', 0.0) for r in pattern_rules) / len(pattern_rules)
                score_breakdown['components']['pattern_mixing'] = avg_pattern_relevance
                
                if avg_pattern_relevance > 0.8:
                    score_breakdown['strengths'].append("Well-balanced pattern combination")
                elif avg_pattern_relevance < 0.5:
                    score_breakdown['weaknesses'].append("Pattern mixing could be refined")
                    if len(patterns) > 1:
                        score_breakdown['improvement_suggestions'].append("Consider reducing the number of patterns")
                    else:
                        score_breakdown['improvement_suggestions'].append("Try adding a subtle pattern for visual interest")
        
        # Analyze style appropriateness
        style_rules = [r for r in relevant_rules if r.get('category') == 'style_coordination']
        if style_rules:
            avg_style_relevance = sum(r.get('relevance_score', 0.0) for r in style_rules) / len(style_rules)
            score_breakdown['components']['style_coordination'] = avg_style_relevance
            
            if avg_style_relevance > 0.8:
                score_breakdown['strengths'].append("Excellent style coordination")
            elif avg_style_relevance < 0.5:
                score_breakdown['weaknesses'].append("Style coordination needs attention")
                score_breakdown['improvement_suggestions'].append("Ensure all pieces match in formality level")
        
        # Analyze occasion appropriateness
        occasion_rules = [r for r in relevant_rules if r.get('category') == 'occasion_appropriate']
        if occasion_rules:
            avg_occasion_relevance = sum(r.get('relevance_score', 0.0) for r in occasion_rules) / len(occasion_rules)
            score_breakdown['components']['occasion_appropriate'] = avg_occasion_relevance
            
            if avg_occasion_relevance > 0.8:
                score_breakdown['strengths'].append("Perfect for the occasion")
            elif avg_occasion_relevance < 0.5:
                score_breakdown['weaknesses'].append("May not be ideal for the occasion")
                score_breakdown['improvement_suggestions'].append("Consider adjusting formality level for the occasion")
        
        return score_breakdown
    
    def generate_comparison_explanation(self, 
                                     outfit1: Dict, 
                                     outfit2: Dict, 
                                     context: Dict) -> Dict:
        """
        Generate explanation comparing two outfits
        
        Args:
            outfit1: First outfit
            outfit2: Second outfit
            context: Outfit context
            
        Returns:
            Comparison explanation
        """
        score1 = outfit1.get('score', 0.0)
        score2 = outfit2.get('score', 0.0)
        
        # Get explanations for both
        explanation1 = self.generate_explanation(outfit1, context)
        explanation2 = self.generate_explanation(outfit2, context)
        
        # Generate comparison
        comparison = {
            'outfit1': {
                'score': score1,
                'primary_reason': explanation1.primary_reason,
                'confidence': explanation1.confidence_score
            },
            'outfit2': {
                'score': score2,
                'primary_reason': explanation2.primary_reason,
                'confidence': explanation2.confidence_score
            },
            'winner': 'outfit1' if score1 > score2 else 'outfit2' if score2 > score1 else 'tie',
            'score_difference': abs(score1 - score2),
            'key_differences': []
        }
        
        # Identify key differences
        if explanation1.confidence_score > explanation2.confidence_score:
            comparison['key_differences'].append("First outfit has stronger rule support")
        elif explanation2.confidence_score > explanation1.confidence_score:
            comparison['key_differences'].append("Second outfit has stronger rule support")
        
        # Add rule-based differences
        rules1 = {r['id'] for r in explanation1.supporting_rules}
        rules2 = {r['id'] for r in explanation2.supporting_rules}
        
        if rules1 != rules2:
            unique_rules1 = rules1 - rules2
            unique_rules2 = rules2 - rules1
            
            if unique_rules1:
                comparison['key_differences'].append(f"First outfit follows unique rules: {list(unique_rules1)}")
            if unique_rules2:
                comparison['key_differences'].append(f"Second outfit follows unique rules: {list(unique_rules2)}")
        
        return comparison
    
    def export_explanation(self, explanation: Explanation, output_path: Union[str, Path]):
        """Export explanation to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(explanation.to_dict(), f, indent=2)
        
        print(f"Explanation exported to: {output_path}")
    
    def batch_explain(self, 
                     recommendations: List[Dict], 
                     context: Dict) -> List[Explanation]:
        """Generate explanations for multiple recommendations"""
        explanations = []
        
        print(f"Generating explanations for {len(recommendations)} recommendations...")
        
        for i, recommendation in enumerate(recommendations):
            try:
                explanation = self.generate_explanation(recommendation, context)
                explanations.append(explanation)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(recommendations)} explanations")
                    
            except Exception as e:
                print(f"Error generating explanation for recommendation {i}: {e}")
                continue
        
        print(f"Generated {len(explanations)} explanations")
        return explanations


# Test function
def test_explainability():
    """Test the explainability system"""
    print("Testing Explainability Engine...")
    
    # Create components
    rule_base = RuleBase()
    embedding_index = EmbeddingIndex()
    
    # Build index
    embedding_index.build_index(rule_base.rules)
    
    # Create explainability engine
    engine = ExplainabilityEngine(rule_base, embedding_index)
    
    # Test recommendation
    recommendation = {
        'id': 'test_rec_1',
        'items': ['blue_shirt', 'gray_pants', 'black_shoes'],
        'score': 0.85
    }
    
    context = {
        'occasion': 'business',
        'season': 'winter',
        'colors': ['blue', 'gray', 'black'],
        'patterns': ['solid'],
        'style': 'professional'
    }
    
    # Generate explanation
    explanation = engine.generate_explanation(recommendation, context)
    
    print(f"Generated explanation:")
    print(f"  Primary reason: {explanation.primary_reason}")
    print(f"  Confidence: {explanation.confidence_score}")
    print(f"  Supporting rules: {len(explanation.supporting_rules)}")
    print(f"  Context factors: {list(explanation.context_factors.keys())}")
    print(f"  Alternatives: {len(explanation.alternatives)}")
    print(f"  Additional tips: {len(explanation.additional_tips)}")
    
    # Test score explanation
    outfit = {
        'items': ['blue_shirt', 'gray_pants', 'black_shoes'],
        'score': 0.85
    }
    
    score_explanation = engine.explain_outfit_score(outfit, context)
    print(f"\nScore explanation:")
    print(f"  Overall score: {score_explanation['overall_score']}")
    print(f"  Components: {list(score_explanation['components'].keys())}")
    print(f"  Strengths: {score_explanation['strengths']}")
    print(f"  Weaknesses: {score_explanation['weaknesses']}")
    
    # Test comparison
    outfit2 = {
        'items': ['red_shirt', 'blue_jeans', 'brown_shoes'],
        'score': 0.72
    }
    
    comparison = engine.generate_comparison_explanation(outfit, outfit2, context)
    print(f"\nComparison:")
    print(f"  Winner: {comparison['winner']}")
    print(f"  Score difference: {comparison['score_difference']:.3f}")
    print(f"  Key differences: {comparison['key_differences']}")
    
    # Test export
    engine.export_explanation(explanation, "test_explanation.json")
    
    print("Explainability test completed!")


if __name__ == "__main__":
    test_explainability()
