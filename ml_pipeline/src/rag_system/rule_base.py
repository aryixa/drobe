"""
Task 13 - RAG + Explainability System: Rule Base
Fashion rules and guidelines database for outfit recommendations
"""

import json
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict
import numpy as np

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import RULE_BASE_PATH


class RuleCategory(Enum):
    """Categories of fashion rules"""
    COLOR_HARMONY = "color_harmony"
    PATTERN_MIXING = "pattern_mixing"
    STYLE_COORDINATION = "style_coordination"
    OCCASION_APPROPRIATE = "occasion_appropriate"
    SEASONAL_GUIDELINES = "seasonal_guidelines"
    BODY_TYPE_GUIDELINES = "body_type_guidelines"
    ACCESSORY_RULES = "accessory_rules"
    LAYERING_PRINCIPLES = "layering_principles"
    PROPORTION_BALANCE = "proportion_balance"
    FABRIC_COORDINATION = "fabric_coordination"


class RulePriority(Enum):
    """Priority levels for rules"""
    CRITICAL = "critical"      # Must follow
    IMPORTANT = "important"    # Should follow
    RECOMMENDED = "recommended"  # Nice to follow
    OPTIONAL = "optional"      # Can follow


class OccasionType(Enum):
    """Types of occasions"""
    CASUAL = "casual"
    BUSINESS = "business"
    FORMAL = "formal"
    PARTY = "party"
    DATE = "date"
    SPORTS = "sports"
    OUTDOOR = "outdoor"
    BEACH = "beach"
    TRAVEL = "travel"


class SeasonType(Enum):
    """Season types"""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"
    ALL_SEASON = "all_season"


@dataclass
class FashionRule:
    """Individual fashion rule"""
    id: str
    title: str
    description: str
    category: RuleCategory
    priority: RulePriority
    occasions: List[OccasionType]
    seasons: List[SeasonType]
    conditions: Dict[str, Union[str, List[str]]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert rule to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'category': self.category.value,
            'priority': self.priority.value,
            'occasions': [occ.value for occ in self.occasions],
            'seasons': [sea.value for sea in self.seasons],
            'conditions': self.conditions,
            'recommendations': self.recommendations,
            'exceptions': self.exceptions,
            'examples': self.examples,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FashionRule':
        """Create rule from dictionary"""
        return cls(
            id=data['id'],
            title=data['title'],
            description=data['description'],
            category=RuleCategory(data['category']),
            priority=RulePriority(data['priority']),
            occasions=[OccasionType(occ) for occ in data['occasions']],
            seasons=[SeasonType(sea) for sea in data['seasons']],
            conditions=data.get('conditions', {}),
            recommendations=data.get('recommendations', []),
            exceptions=data.get('exceptions', []),
            examples=data.get('examples', []),
            confidence=data.get('confidence', 1.0)
        )


class RuleBase:
    """
    Fashion rule database and management system
    - Comprehensive fashion rules
    - Rule categorization and prioritization
    - Context-aware rule filtering
    """
    
    def __init__(self, rule_file: Optional[Union[str, Path]] = None):
        self.rule_file = Path(rule_file) if rule_file else RULE_BASE_PATH
        self.rules: List[FashionRule] = []
        
        # Initialize with default rules
        self._initialize_default_rules()
        
        # Load existing rules if file exists
        if self.rule_file.exists():
            self.load_rules()
        else:
            self.save_rules()  # Save default rules
        
        print(f"Rule Base initialized with {len(self.rules)} rules")
    
    def _initialize_default_rules(self):
        """Initialize with comprehensive default fashion rules"""
        default_rules = [
            # Color Harmony Rules
            FashionRule(
                id="color_monochromatic",
                title="Monochromatic Color Scheme",
                description="Wear items from the same color family for a cohesive look",
                category=RuleCategory.COLOR_HARMONY,
                priority=RulePriority.IMPORTANT,
                occasions=[OccasionType.BUSINESS, OccasionType.FORMAL, OccasionType.CASUAL],
                seasons=[SeasonType.ALL_SEASON],
                conditions={"color_count": "1-2"},
                recommendations=[
                    "Choose shades and tints of the same color",
                    "Add texture for visual interest",
                    "Use different materials to create depth"
                ],
                examples=["All black outfit", "Navy blue suit with light blue shirt"],
                confidence=0.9
            ),
            
            FashionRule(
                id="color_complementary",
                title="Complementary Color Scheme",
                description="Combine opposite colors on the color wheel for high contrast",
                category=RuleCategory.COLOR_HARMONY,
                priority=RulePriority.RECOMMENDED,
                occasions=[OccasionType.CASUAL, OccasionType.PARTY, OccasionType.DATE],
                seasons=[SeasonType.ALL_SEASON],
                conditions={"color_relationship": "complementary"},
                recommendations=[
                    "Use one color as dominant, other as accent",
                    "Balance bold colors with neutrals",
                    "Consider color intensity and saturation"
                ],
                examples=["Blue shirt with orange accessories", "Red dress with green jewelry"],
                confidence=0.8
            ),
            
            FashionRule(
                id="color_analogous",
                title="Analogous Color Scheme",
                description="Use adjacent colors on the color wheel for harmony",
                category=RuleCategory.COLOR_HARMONY,
                priority=RulePriority.IMPORTANT,
                occasions=[OccasionType.BUSINESS, OccasionType.CASUAL, OccasionType.DATE],
                seasons=[SeasonType.ALL_SEASON],
                conditions={"color_relationship": "analogous"},
                recommendations=[
                    "Choose 2-3 adjacent colors",
                    "Vary the intensity of each color",
                    "Use neutral colors as base"
                ],
                examples=["Blue, purple, and navy outfit", "Yellow, orange, and cream combination"],
                confidence=0.85
            ),
            
            # Pattern Mixing Rules
            FashionRule(
                id="pattern_solid_base",
                title="Solid Color Base",
                description="Start with solid colors and add one patterned item",
                category=RuleCategory.PATTERN_MIXING,
                priority=RulePriority.CRITICAL,
                occasions=[OccasionType.BUSINESS, OccasionType.FORMAL, OccasionType.CASUAL],
                seasons=[SeasonType.ALL_SEASON],
                conditions={"pattern_count": "0-1"},
                recommendations=[
                    "Use solid colors as foundation",
                    "Add one patterned item as focal point",
                    "Keep other elements simple"
                ],
                examples=["Solid suit with patterned shirt", "Plain dress with patterned scarf"],
                confidence=0.95
            ),
            
            FashionRule(
                id="pattern_scale_variation",
                title="Vary Pattern Scales",
                description="Mix patterns of different scales to avoid visual conflict",
                category=RuleCategory.PATTERN_MIXING,
                priority=RulePriority.IMPORTANT,
                occasions=[OccasionType.CASUAL, OccasionType.PARTY],
                seasons=[SeasonType.ALL_SEASON],
                conditions={"pattern_count": "2+"},
                recommendations=[
                    "Combine large-scale with small-scale patterns",
                    "Use different pattern types (stripes with dots)",
                    "Maintain color harmony between patterns"
                ],
                examples=["Large floral print with small polka dots", "Wide stripes with thin pinstripes"],
                confidence=0.8
            ),
            
            FashionRule(
                id="pattern_color_coordination",
                title="Coordinate Pattern Colors",
                description="Ensure mixed patterns share common colors",
                category=RuleCategory.PATTERN_MIXING,
                priority=RulePriority.CRITICAL,
                occasions=[OccasionType.CASUAL, OccasionType.PARTY],
                seasons=[SeasonType.ALL_SEASON],
                conditions={"pattern_count": "2+"},
                recommendations=[
                    "Share at least one color between patterns",
                    "Use neutral colors to bridge patterns",
                    "Match pattern intensity levels"
                ],
                examples=["Blue stripes with blue floral print", "Red plaid with red polka dots"],
                confidence=0.9
            ),
            
            # Style Coordination Rules
            FashionRule(
                id="style_consistency",
                title="Maintain Style Consistency",
                description="Keep all items in the same style family",
                category=RuleCategory.STYLE_COORDINATION,
                priority=RulePriority.IMPORTANT,
                occasions=[OccasionType.BUSINESS, OccasionType.FORMAL, OccasionType.CASUAL],
                seasons=[SeasonType.ALL_SEASON],
                conditions={"style_mismatch": "avoid"},
                recommendations=[
                    "Match formal levels of all items",
                    "Consider fabric textures and weights",
                    "Ensure accessories complement main pieces"
                ],
                examples=["Business suit with business shoes", "Casual jeans with casual sneakers"],
                confidence=0.85
            ),
            
            FashionRule(
                id="style_focal_point",
                title="Create a Focal Point",
                description="Have one standout piece in your outfit",
                category=RuleCategory.STYLE_COORDINATION,
                priority=RulePriority.RECOMMENDED,
                occasions=[OccasionType.PARTY, OccasionType.DATE, OccasionType.CASUAL],
                seasons=[SeasonType.ALL_SEASON],
                conditions={"focal_pieces": "1"},
                recommendations=[
                    "Choose one item to be the star",
                    "Keep other pieces supporting, not competing",
                    "Use color, pattern, or texture for focus"
                ],
                examples=["Statement necklace with simple dress", "Bright colored jacket with neutral outfit"],
                confidence=0.8
            ),
            
            # Occasion Appropriate Rules
            FashionRule(
                id="business_formal",
                title="Business Formal Attire",
                description="Professional attire for formal business settings",
                category=RuleCategory.OCCASION_APPROPRIATE,
                priority=RulePriority.CRITICAL,
                occasions=[OccasionType.BUSINESS],
                seasons=[SeasonType.ALL_SEASON],
                conditions={"formality": "high"},
                recommendations=[
                    "Wear tailored suits or separates",
                    "Choose conservative colors (navy, black, gray)",
                    "Ensure proper fit and condition",
                    "Minimal accessories and makeup"
                ],
                examples=["Navy suit with white shirt", "Black dress with blazer"],
                confidence=0.95
            ),
            
            FashionRule(
                id="casual_comfort",
                title="Casual Comfort",
                description="Relaxed yet put-together casual attire",
                category=RuleCategory.OCCASION_APPROPRIATE,
                priority=RulePriority.IMPORTANT,
                occasions=[OccasionType.CASUAL],
                seasons=[SeasonType.ALL_SEASON],
                conditions={"formality": "low"},
                recommendations=[
                    "Choose comfortable, well-fitting pieces",
                    "Mix casual with slightly elevated items",
                    "Pay attention to fabric quality",
                    "Keep grooming polished"
                ],
                examples=["Dark jeans with nice sweater", "Casual dress with denim jacket"],
                confidence=0.8
            ),
            
            # Seasonal Guidelines
            FashionRule(
                id="summer_lightweight",
                title="Summer Lightweight Fabrics",
                description="Choose breathable, lightweight fabrics for summer",
                category=RuleCategory.SEASONAL_GUIDELINES,
                priority=RulePriority.IMPORTANT,
                occasions=[OccasionType.CASUAL, OccasionType.BEACH, OccasionType.OUTDOOR],
                seasons=[SeasonType.SUMMER],
                conditions={"temperature": "hot"},
                recommendations=[
                    "Wear natural fibers (cotton, linen)",
                    "Choose light colors to reflect heat",
                    "Opt for loose-fitting garments",
                    "Consider moisture-wicking materials"
                ],
                examples=["Linen shirt with cotton shorts", "Lightweight dress with sandals"],
                confidence=0.9
            ),
            
            FashionRule(
                id="winter_layering",
                title="Winter Layering",
                description="Layer appropriately for cold weather while maintaining style",
                category=RuleCategory.SEASONAL_GUIDELINES,
                priority=RulePriority.IMPORTANT,
                occasions=[OccasionType.BUSINESS, OccasionType.CASUAL, OccasionType.OUTDOOR],
                seasons=[SeasonType.WINTER],
                conditions={"temperature": "cold"},
                recommendations=[
                    "Start with base layer (thermal or thin shirt)",
                    "Add insulating middle layer",
                    "Finish with protective outer layer",
                    "Ensure layers can be removed indoors"
                ],
                examples=["Thermal shirt, sweater, winter coat", "Blouse, cardigan, wool coat"],
                confidence=0.9
            ),
            
            # Body Type Guidelines
            FashionRule(
                id="proportion_balance",
                title="Balance Proportions",
                description="Create visual balance with clothing proportions",
                category=RuleCategory.BODY_TYPE_GUIDELINES,
                priority=RulePriority.IMPORTANT,
                occasions=[OccasionType.BUSINESS, OccasionType.CASUAL, OccasionType.FORMAL],
                seasons=[SeasonType.ALL_SEASON],
                conditions={"body_type": "any"},
                recommendations=[
                    "Consider your natural body proportions",
                    "Use clothing lines to create balance",
                    "Pay attention to horizontal and vertical lines",
                    "Choose appropriate garment lengths"
                ],
                examples=["High-waisted pants for shorter torso", "A-line dress for balanced proportions"],
                confidence=0.8
            ),
            
            # Accessory Rules
            FashionRule(
                id="accessory_balance",
                title="Balance Accessories",
                description="Choose accessories that complement, not overwhelm",
                category=RuleCategory.ACCESSORY_RULES,
                priority=RulePriority.RECOMMENDED,
                occasions=[OccasionType.BUSINESS, OccasionType.CASUAL, OccasionType.PARTY],
                seasons=[SeasonType.ALL_SEASON],
                conditions={"accessory_count": "moderate"},
                recommendations=[
                    "Limit to 3-4 statement pieces",
                    "Coordinate metal finishes",
                    "Scale accessories to your body size",
                    "Consider occasion appropriateness"
                ],
                examples=["Watch, belt, and simple necklace", "Statement earrings with bracelet"],
                confidence=0.8
            ),
            
            # Layering Principles
            FashionRule(
                id="layer_visual_interest",
                title="Create Visual Interest with Layers",
                description="Use layers to add depth and interest to outfits",
                category=RuleCategory.LAYERING_PRINCIPLES,
                priority=RulePriority.RECOMMENDED,
                occasions=[OccasionType.CASUAL, OccasionType.BUSINESS],
                seasons=[SeasonType.FALL, SeasonType.WINTER, SeasonType.SPRING],
                conditions={"layer_count": "2-3"},
                recommendations=[
                    "Vary textures and materials",
                    "Use different lengths for visual interest",
                    "Ensure layers can be seen and appreciated",
                    "Consider functionality and comfort"
                ],
                examples=["Shirt, sweater, and blazer", "Dress with cardigan and scarf"],
                confidence=0.8
            )
        ]
        
        self.rules.extend(default_rules)
    
    def add_rule(self, rule: FashionRule):
        """Add a new rule to the database"""
        # Check if rule ID already exists
        existing_ids = [r.id for r in self.rules]
        if rule.id in existing_ids:
            raise ValueError(f"Rule with ID '{rule.id}' already exists")
        
        self.rules.append(rule)
        print(f"Added rule: {rule.title}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID"""
        for i, rule in enumerate(self.rules):
            if rule.id == rule_id:
                removed_rule = self.rules.pop(i)
                print(f"Removed rule: {removed_rule.title}")
                return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[FashionRule]:
        """Get a rule by ID"""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None
    
    def filter_rules(self, 
                    category: Optional[RuleCategory] = None,
                    priority: Optional[RulePriority] = None,
                    occasion: Optional[OccasionType] = None,
                    season: Optional[SeasonType] = None,
                    min_confidence: float = 0.0) -> List[FashionRule]:
        """Filter rules based on criteria"""
        filtered_rules = self.rules
        
        if category:
            filtered_rules = [r for r in filtered_rules if r.category == category]
        
        if priority:
            filtered_rules = [r for r in filtered_rules if r.priority == priority]
        
        if occasion:
            filtered_rules = [r for r in filtered_rules if occasion in r.occasions]
        
        if season:
            filtered_rules = [r for r in filtered_rules if season in r.seasons or SeasonType.ALL_SEASON in r.seasons]
        
        if min_confidence > 0:
            filtered_rules = [r for r in filtered_rules if r.confidence >= min_confidence]
        
        return filtered_rules
    
    def search_rules(self, query: str, top_k: int = 10) -> List[Tuple[FashionRule, float]]:
        """Search rules by query text (simple keyword matching)"""
        query_lower = query.lower()
        scored_rules = []
        
        for rule in self.rules:
            score = 0.0
            
            # Search in title
            if query_lower in rule.title.lower():
                score += 2.0
            
            # Search in description
            if query_lower in rule.description.lower():
                score += 1.5
            
            # Search in recommendations
            for rec in rule.recommendations:
                if query_lower in rec.lower():
                    score += 1.0
                    break
            
            # Search in examples
            for example in rule.examples:
                if query_lower in example.lower():
                    score += 0.5
                    break
            
            # Search in conditions
            for key, value in rule.conditions.items():
                if query_lower in str(value).lower():
                    score += 0.5
                    break
            
            if score > 0:
                scored_rules.append((rule, score))
        
        # Sort by score and return top k
        scored_rules.sort(key=lambda x: x[1], reverse=True)
        return scored_rules[:top_k]
    
    def get_rules_for_outfit(self, 
                           outfit_context: Dict[str, Union[str, List[str]]]) -> List[FashionRule]:
        """Get relevant rules for a specific outfit context"""
        # Extract context information
        occasion_str = outfit_context.get('occasion', '').lower()
        season_str = outfit_context.get('season', '').lower()
        colors = outfit_context.get('colors', [])
        patterns = outfit_context.get('patterns', [])
        style = outfit_context.get('style', '').lower()
        
        # Convert strings to enums
        occasion = None
        if occasion_str:
            for occ in OccasionType:
                if occ.value == occasion_str:
                    occasion = occ
                    break
        
        season = None
        if season_str:
            for sea in SeasonType:
                if sea.value == season_str:
                    season = sea
                    break
        
        # Get base filtered rules
        relevant_rules = self.filter_rules(
            occasion=occasion,
            season=season
        )
        
        # Score rules based on outfit context
        scored_rules = []
        for rule in relevant_rules:
            context_score = 0.0
            
            # Check color relevance
            if colors and rule.category == RuleCategory.COLOR_HARMONY:
                context_score += 1.0
            
            # Check pattern relevance
            if patterns and rule.category == RuleCategory.PATTERN_MIXING:
                if len(patterns) > 1:
                    context_score += 1.0
                elif len(patterns) == 1:
                    context_score += 0.5
            
            # Check style relevance
            if style and rule.category == RuleCategory.STYLE_COORDINATION:
                context_score += 0.5
            
            # Check priority
            priority_weights = {
                RulePriority.CRITICAL: 1.0,
                RulePriority.IMPORTANT: 0.8,
                RulePriority.RECOMMENDED: 0.6,
                RulePriority.OPTIONAL: 0.4
            }
            context_score *= priority_weights.get(rule.priority, 0.5)
            
            scored_rules.append((rule, context_score))
        
        # Sort by context score
        scored_rules.sort(key=lambda x: x[1], reverse=True)
        
        return [rule for rule, _ in scored_rules]
    
    def get_rule_statistics(self) -> Dict:
        """Get statistics about the rule base"""
        category_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        occasion_counts = defaultdict(int)
        season_counts = defaultdict(int)
        
        for rule in self.rules:
            category_counts[rule.category.value] += 1
            priority_counts[rule.priority.value] += 1
            
            for occasion in rule.occasions:
                occasion_counts[occasion.value] += 1
            
            for season in rule.seasons:
                season_counts[season.value] += 1
        
        return {
            'total_rules': len(self.rules),
            'category_distribution': dict(category_counts),
            'priority_distribution': dict(priority_counts),
            'occasion_distribution': dict(occasion_counts),
            'season_distribution': dict(season_counts),
            'avg_confidence': np.mean([r.confidence for r in self.rules]),
            'high_confidence_rules': len([r for r in self.rules if r.confidence >= 0.9])
        }
    
    def save_rules(self, file_path: Optional[Union[str, Path]] = None):
        """Save rules to JSON file"""
        file_path = Path(file_path) if file_path else self.rule_file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        rules_data = [rule.to_dict() for rule in self.rules]
        
        with open(file_path, 'w') as f:
            json.dump(rules_data, f, indent=2)
        
        print(f"Saved {len(self.rules)} rules to: {file_path}")
    
    def load_rules(self, file_path: Optional[Union[str, Path]] = None):
        """Load rules from JSON file"""
        file_path = Path(file_path) if file_path else self.rule_file
        
        if not file_path.exists():
            print(f"Rule file not found: {file_path}")
            return
        
        with open(file_path, 'r') as f:
            rules_data = json.load(f)
        
        self.rules = [FashionRule.from_dict(rule_data) for rule_data in rules_data]
        print(f"Loaded {len(self.rules)} rules from: {file_path}")
    
    def export_rule_summaries(self, output_path: Union[str, Path]):
        """Export rule summaries for documentation"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summaries = []
        for rule in self.rules:
            summary = {
                'id': rule.id,
                'title': rule.title,
                'category': rule.category.value,
                'priority': rule.priority.value,
                'description': rule.description,
                'key_recommendations': rule.recommendations[:3] if rule.recommendations else [],
                'occasions': [occ.value for occ in rule.occasions],
                'seasons': [sea.value for sea in rule.seasons]
            }
            summaries.append(summary)
        
        with open(output_path, 'w') as f:
            json.dump(summaries, f, indent=2)
        
        print(f"Exported rule summaries to: {output_path}")


# Test function
def test_rule_base():
    """Test the rule base system"""
    print("Testing Rule Base...")
    
    # Create rule base
    rule_base = RuleBase()
    
    # Test statistics
    stats = rule_base.get_rule_statistics()
    print(f"Rule base stats: {stats}")
    
    # Test filtering
    color_rules = rule_base.filter_rules(category=RuleCategory.COLOR_HARMONY)
    print(f"Color harmony rules: {len(color_rules)}")
    
    critical_rules = rule_base.filter_rules(priority=RulePriority.CRITICAL)
    print(f"Critical rules: {len(critical_rules)}")
    
    # Test search
    search_results = rule_base.search_rules("color", top_k=5)
    print(f"Search results for 'color': {len(search_results)}")
    for rule, score in search_results:
        print(f"  {rule.title} (score: {score:.1f})")
    
    # Test outfit context
    outfit_context = {
        'occasion': 'business',
        'season': 'winter',
        'colors': ['blue', 'gray'],
        'patterns': ['solid'],
        'style': 'professional'
    }
    
    relevant_rules = rule_base.get_rules_for_outfit(outfit_context)
    print(f"Relevant rules for outfit: {len(relevant_rules)}")
    
    # Test adding custom rule
    custom_rule = FashionRule(
        id="custom_test",
        title="Test Rule",
        description="A test rule for demonstration",
        category=RuleCategory.STYLE_COORDINATION,
        priority=RulePriority.RECOMMENDED,
        occasions=[OccasionType.CASUAL],
        seasons=[SeasonType.ALL_SEASON],
        recommendations=["Test recommendation"],
        confidence=0.7
    )
    
    rule_base.add_rule(custom_rule)
    print(f"Added custom rule. Total rules: {len(rule_base.rules)}")
    
    # Test saving and loading
    rule_base.save_rules()
    print("Rules saved successfully")
    
    print("Rule base test completed!")


if __name__ == "__main__":
    test_rule_base()
