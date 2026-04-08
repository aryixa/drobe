"""
Task 14 - Context-Aware Styling + Integration: Context Parser
Convert natural language input to structured context
"""

import re
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, time
import json

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.rag_system.rule_base import OccasionType, SeasonType


class WeatherCondition(Enum):
    """Weather conditions"""
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    SNOWY = "snowy"
    WINDY = "windy"
    HOT = "hot"
    COLD = "cold"
    MILD = "mild"
    HUMID = "humid"


class TimeOfDay(Enum):
    """Time of day categories"""
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"
    DAY = "day"
    ALL_DAY = "all_day"


class StyleLevel(Enum):
    """Style formality levels"""
    VERY_CASUAL = "very_casual"
    CASUAL = "casual"
    SMART_CASUAL = "smart_casual"
    BUSINESS = "business"
    BUSINESS_CASUAL = "business_casual"
    FORMAL = "formal"
    VERY_FORMAL = "very_formal"
    COCKTAIL = "cocktail"
    PARTY = "party"


@dataclass
class ParsedContext:
    """Structured context from natural language input"""
    original_query: str
    occasion: Optional[OccasionType]
    season: Optional[SeasonType]
    weather: Optional[WeatherCondition]
    time_of_day: Optional[TimeOfDay]
    style_level: Optional[StyleLevel]
    colors: List[str]
    patterns: List[str]
    clothing_types: List[str]
    keywords: List[str]
    temperature_range: Optional[Tuple[float, float]]
    confidence: float
    parsing_errors: List[str]


class ContextParser:
    """
    Natural language context parser for outfit recommendations
    - Extract occasion, weather, time, style
    - Convert to structured context
    - Handle various input formats
    """
    
    def __init__(self):
        # Initialize keyword mappings
        self.occasion_keywords = self._init_occasion_keywords()
        self.season_keywords = self._init_season_keywords()
        self.weather_keywords = self._init_weather_keywords()
        self.time_keywords = self._init_time_keywords()
        self.style_keywords = self._init_style_keywords()
        self.color_keywords = self._init_color_keywords()
        self.pattern_keywords = self._init_pattern_keywords()
        self.clothing_type_keywords = self._init_clothing_type_keywords()
        
        print("Context Parser initialized")
    
    def _init_occasion_keywords(self) -> Dict[str, OccasionType]:
        """Initialize occasion keyword mappings"""
        return {
            # Work/Business
            'work': OccasionType.BUSINESS,
            'office': OccasionType.BUSINESS,
            'business': OccasionType.BUSINESS,
            'meeting': OccasionType.BUSINESS,
            'conference': OccasionType.BUSINESS,
            'presentation': OccasionType.BUSINESS,
            'interview': OccasionType.BUSINESS,
            'professional': OccasionType.BUSINESS,
            
            # Casual
            'casual': OccasionType.CASUAL,
            'everyday': OccasionType.CASUAL,
            'daily': OccasionType.CASUAL,
            'regular': OccasionType.CASUAL,
            'weekend': OccasionType.CASUAL,
            'relaxed': OccasionType.CASUAL,
            'comfortable': OccasionType.CASUAL,
            
            # Formal
            'formal': OccasionType.FORMAL,
            'black tie': OccasionType.FORMAL,
            'wedding': OccasionType.FORMAL,
            'gala': OccasionType.FORMAL,
            'ceremony': OccasionType.FORMAL,
            'elegant': OccasionType.FORMAL,
            'sophisticated': OccasionType.FORMAL,
            
            # Party
            'party': OccasionType.PARTY,
            'celebration': OccasionType.PARTY,
            'birthday': OccasionType.PARTY,
            'club': OccasionType.PARTY,
            'night out': OccasionType.PARTY,
            'festive': OccasionType.PARTY,
            
            # Date
            'date': OccasionType.DATE,
            'dinner': OccasionType.DATE,
            'romantic': OccasionType.DATE,
            'valentine': OccasionType.DATE,
            'anniversary': OccasionType.DATE,
            
            # Sports
            'sports': OccasionType.SPORTS,
            'gym': OccasionType.SPORTS,
            'workout': OccasionType.SPORTS,
            'exercise': OccasionType.SPORTS,
            'running': OccasionType.SPORTS,
            'fitness': OccasionType.SPORTS,
            'athletic': OccasionType.SPORTS,
            
            # Outdoor
            'outdoor': OccasionType.OUTDOOR,
            'hiking': OccasionType.OUTDOOR,
            'camping': OccasionType.OUTDOOR,
            'picnic': OccasionType.OUTDOOR,
            'beach': OccasionType.BEACH,
            'park': OccasionType.OUTDOOR,
            'garden': OccasionType.OUTDOOR,
            
            # Beach
            'beach': OccasionType.BEACH,
            'swim': OccasionType.BEACH,
            'pool': OccasionType.BEACH,
            'vacation': OccasionType.BEACH,
            'resort': OccasionType.BEACH,
            'tropical': OccasionType.BEACH,
            
            # Travel
            'travel': OccasionType.TRAVEL,
            'trip': OccasionType.TRAVEL,
            'airport': OccasionType.TRAVEL,
            'flight': OccasionType.TRAVEL,
            'vacation': OccasionType.TRAVEL,
            'holiday': OccasionType.TRAVEL,
        }
    
    def _init_season_keywords(self) -> Dict[str, SeasonType]:
        """Initialize season keyword mappings"""
        return {
            'spring': SeasonType.SPRING,
            'summer': SeasonType.SUMMER,
            'fall': SeasonType.FALL,
            'autumn': SeasonType.FALL,
            'winter': SeasonType.WINTER,
            'all season': SeasonType.ALL_SEASON,
            'year-round': SeasonType.ALL_SEASON,
        }
    
    def _init_weather_keywords(self) -> Dict[str, WeatherCondition]:
        """Initialize weather keyword mappings"""
        return {
            'sunny': WeatherCondition.SUNNY,
            'clear': WeatherCondition.SUNNY,
            'bright': WeatherCondition.SUNNY,
            'cloudy': WeatherCondition.CLOUDY,
            'overcast': WeatherCondition.CLOUDY,
            'grey': WeatherCondition.CLOUDY,
            'gray': WeatherCondition.CLOUDY,
            'rainy': WeatherCondition.RAINY,
            'rain': WeatherCondition.RAINY,
            'wet': WeatherCondition.RAINY,
            'showers': WeatherCondition.RAINY,
            'snowy': WeatherCondition.SNOWY,
            'snow': WeatherCondition.SNOWY,
            'flurries': WeatherCondition.SNOWY,
            'blizzard': WeatherCondition.SNOWY,
            'windy': WeatherCondition.WINDY,
            'breezy': WeatherCondition.WINDY,
            'hot': WeatherCondition.HOT,
            'warm': WeatherCondition.HOT,
            'heat': WeatherCondition.HOT,
            'cold': WeatherCondition.COLD,
            'chilly': WeatherCondition.COLD,
            'cool': WeatherCondition.COLD,
            'freezing': WeatherCondition.COLD,
            'mild': WeatherCondition.MILD,
            'moderate': WeatherCondition.MILD,
            'pleasant': WeatherCondition.MILD,
            'humid': WeatherCondition.HUMID,
            'sticky': WeatherCondition.HUMID,
            'muggy': WeatherCondition.HUMID,
        }
    
    def _init_time_keywords(self) -> Dict[str, TimeOfDay]:
        """Initialize time of day keyword mappings"""
        return {
            'morning': TimeOfDay.MORNING,
            'am': TimeOfDay.MORNING,
            'breakfast': TimeOfDay.MORNING,
            'dawn': TimeOfDay.MORNING,
            'early': TimeOfDay.MORNING,
            'afternoon': TimeOfDay.AFTERNOON,
            'pm': TimeOfDay.AFTERNOON,
            'lunch': TimeOfDay.AFTERNOON,
            'midday': TimeOfDay.AFTERNOON,
            'noon': TimeOfDay.AFTERNOON,
            'evening': TimeOfDay.EVENING,
            'dinner': TimeOfDay.EVENING,
            'night': TimeOfDay.EVENING,
            'sunset': TimeOfDay.EVENING,
            'late': TimeOfDay.EVENING,
            'nighttime': TimeOfDay.NIGHT,
            'overnight': TimeOfDay.NIGHT,
            'midnight': TimeOfDay.NIGHT,
            'late night': TimeOfDay.NIGHT,
            'day': TimeOfDay.DAY,
            'daytime': TimeOfDay.DAY,
            'all day': TimeOfDay.ALL_DAY,
        }
    
    def _init_style_keywords(self) -> Dict[str, StyleLevel]:
        """Initialize style level keyword mappings"""
        return {
            'very casual': StyleLevel.VERY_CASUAL,
            'ultra casual': StyleLevel.VERY_CASUAL,
            'super casual': StyleLevel.VERY_CASUAL,
            'casual': StyleLevel.CASUAL,
            'relaxed': StyleLevel.CASUAL,
            'comfortable': StyleLevel.CASUAL,
            'smart casual': StyleLevel.SMART_CASUAL,
            'casual chic': StyleLevel.SMART_CASUAL,
            'business casual': StyleLevel.BUSINESS_CASUAL,
            'business': StyleLevel.BUSINESS,
            'professional': StyleLevel.BUSINESS,
            'formal': StyleLevel.FORMAL,
            'dressy': StyleLevel.FORMAL,
            'elegant': StyleLevel.FORMAL,
            'very formal': StyleLevel.VERY_FORMAL,
            'black tie': StyleLevel.VERY_FORMAL,
            'cocktail': StyleLevel.COCKTAIL,
            'party': StyleLevel.PARTY,
            'festive': StyleLevel.PARTY,
        }
    
    def _init_color_keywords(self) -> List[str]:
        """Initialize color keywords"""
        return [
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
            'brown', 'black', 'white', 'gray', 'grey', 'beige', 'cream',
            'navy', 'maroon', 'olive', 'teal', 'cyan', 'magenta', 'lime',
            'indigo', 'violet', 'gold', 'silver', 'bronze', 'copper'
        ]
    
    def _init_pattern_keywords(self) -> List[str]:
        """Initialize pattern keywords"""
        return [
            'solid', 'striped', 'stripes', 'checkered', 'plaid', 'floral',
            'polka dot', 'dots', 'geometric', 'abstract', 'animal print',
            'camouflage', 'tie dye', 'marble', 'textured', 'plain'
        ]
    
    def _init_clothing_type_keywords(self) -> List[str]:
        """Initialize clothing type keywords"""
        return [
            'shirt', 't-shirt', 'blouse', 'top', 'sweater', 'hoodie',
            'pants', 'jeans', 'trousers', 'shorts', 'skirt', 'dress',
            'jacket', 'coat', 'blazer', 'cardigan', 'shoes', 'boots',
            'sneakers', 'sandals', 'heels', 'flats', 'accessories'
        ]
    
    def parse_context(self, query: str) -> ParsedContext:
        """
        Parse natural language query into structured context
        
        Args:
            query: Natural language input query
            
        Returns:
            ParsedContext with extracted information
        """
        query_lower = query.lower()
        errors = []
        
        # Extract different context elements
        occasion = self._extract_occasion(query_lower, errors)
        season = self._extract_season(query_lower, errors)
        weather = self._extract_weather(query_lower, errors)
        time_of_day = self._extract_time(query_lower, errors)
        style_level = self._extract_style(query_lower, errors)
        colors = self._extract_colors(query_lower)
        patterns = self._extract_patterns(query_lower)
        clothing_types = self._extract_clothing_types(query_lower)
        temperature_range = self._extract_temperature(query_lower, errors)
        
        # Extract all keywords
        keywords = self._extract_all_keywords(query_lower)
        
        # Calculate confidence based on extraction success
        extracted_elements = [
            occasion, season, weather, time_of_day, style_level
        ]
        non_none_elements = sum(1 for elem in extracted_elements if elem is not None)
        confidence = min(1.0, non_none_elements / 3.0)  # Normalize to 0-1
        
        return ParsedContext(
            original_query=query,
            occasion=occasion,
            season=season,
            weather=weather,
            time_of_day=time_of_day,
            style_level=style_level,
            colors=colors,
            patterns=patterns,
            clothing_types=clothing_types,
            keywords=keywords,
            temperature_range=temperature_range,
            confidence=confidence,
            parsing_errors=errors
        )
    
    def _extract_occasion(self, query: str, errors: List[str]) -> Optional[OccasionType]:
        """Extract occasion from query"""
        best_match = None
        best_score = 0
        
        for keyword, occasion in self.occasion_keywords.items():
            if keyword in query:
                # Score based on keyword specificity
                score = len(keyword.split())  # Multi-word keywords get higher score
                if score > best_score:
                    best_score = score
                    best_match = occasion
        
        return best_match
    
    def _extract_season(self, query: str, errors: List[str]) -> Optional[SeasonType]:
        """Extract season from query"""
        for keyword, season in self.season_keywords.items():
            if keyword in query:
                return season
        return None
    
    def _extract_weather(self, query: str, errors: List[str]) -> Optional[WeatherCondition]:
        """Extract weather condition from query"""
        for keyword, weather in self.weather_keywords.items():
            if keyword in query:
                return weather
        return None
    
    def _extract_time(self, query: str, errors: List[str]) -> Optional[TimeOfDay]:
        """Extract time of day from query"""
        for keyword, time_of_day in self.time_keywords.items():
            if keyword in query:
                return time_of_day
        return None
    
    def _extract_style(self, query: str, errors: List[str]) -> Optional[StyleLevel]:
        """Extract style level from query"""
        # Check multi-word keywords first
        for keyword, style in sorted(self.style_keywords.items(), key=lambda x: len(x[0]), reverse=True):
            if keyword in query:
                return style
        return None
    
    def _extract_colors(self, query: str) -> List[str]:
        """Extract color keywords from query"""
        colors = []
        for color in self.color_keywords:
            if color in query:
                colors.append(color)
        return colors
    
    def _extract_patterns(self, query: str) -> List[str]:
        """Extract pattern keywords from query"""
        patterns = []
        for pattern in self.pattern_keywords:
            if pattern in query:
                patterns.append(pattern)
        return patterns
    
    def _extract_clothing_types(self, query: str) -> List[str]:
        """Extract clothing type keywords from query"""
        types = []
        for clothing_type in self.clothing_type_keywords:
            if clothing_type in query:
                types.append(clothing_type)
        return types
    
    def _extract_all_keywords(self, query: str) -> List[str]:
        """Extract all relevant keywords from query"""
        keywords = []
        
        # Combine all keyword lists
        all_keywords = (
            list(self.occasion_keywords.keys()) +
            list(self.season_keywords.keys()) +
            list(self.weather_keywords.keys()) +
            list(self.time_keywords.keys()) +
            list(self.style_keywords.keys()) +
            self.color_keywords +
            self.pattern_keywords +
            self.clothing_type_keywords
        )
        
        # Find matches
        for keyword in all_keywords:
            if keyword in query:
                keywords.append(keyword)
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_temperature(self, query: str, errors: List[str]) -> Optional[Tuple[float, float]]:
        """Extract temperature range from query"""
        # Look for temperature patterns
        temp_patterns = [
            r'(\d+)\s*degrees?\s*(?:f|°f)',
            r'(\d+)\s*degrees?\s*(?:c|°c)',
            r'(\d+)\s*to\s*(\d+)\s*degrees?',
            r'between\s*(\d+)\s*and\s*(\d+)\s*degrees?',
        ]
        
        for pattern in temp_patterns:
            match = re.search(pattern, query)
            if match:
                try:
                    if len(match.groups()) == 1:
                        # Single temperature
                        temp = float(match.group(1))
                        return (temp - 5, temp + 5)  # ±5 degree range
                    else:
                        # Temperature range
                        temp_min = float(match.group(1))
                        temp_max = float(match.group(2))
                        return (temp_min, temp_max)
                except ValueError:
                    errors.append(f"Could not parse temperature: {match.group(0)}")
                    continue
        
        return None
    
    def enhance_context_with_defaults(self, context: ParsedContext) -> ParsedContext:
        """Enhance context with intelligent defaults"""
        enhanced_context = context
        
        # Default season based on weather
        if enhanced_context.season is None and enhanced_context.weather:
            if enhanced_context.weather in [WeatherCondition.HOT, WeatherCondition.SUNNY]:
                enhanced_context.season = SeasonType.SUMMER
            elif enhanced_context.weather in [WeatherCondition.COLD, WeatherCondition.SNOWY]:
                enhanced_context.season = SeasonType.WINTER
            elif enhanced_context.weather in [WeatherCondition.MILD]:
                enhanced_context.season = SeasonType.SPRING
        
        # Default style based on occasion
        if enhanced_context.style_level is None and enhanced_context.occasion:
            style_mapping = {
                OccasionType.BUSINESS: StyleLevel.BUSINESS,
                OccasionType.CASUAL: StyleLevel.CASUAL,
                OccasionType.FORMAL: StyleLevel.FORMAL,
                OccasionType.PARTY: StyleLevel.PARTY,
                OccasionType.DATE: StyleLevel.SMART_CASUAL,
                OccasionType.SPORTS: StyleLevel.VERY_CASUAL,
                OccasionType.OUTDOOR: StyleLevel.CASUAL,
                OccasionType.BEACH: StyleLevel.VERY_CASUAL,
                OccasionType.TRAVEL: StyleLevel.CASUAL,
            }
            enhanced_context.style_level = style_mapping.get(enhanced_context.occasion)
        
        # Default time based on style
        if enhanced_context.time_of_day is None and enhanced_context.style_level:
            if enhanced_context.style_level in [StyleLevel.FORMAL, StyleLevel.VERY_FORMAL]:
                enhanced_context.time_of_day = TimeOfDay.EVENING
            elif enhanced_context.style_level in [StyleLevel.BUSINESS]:
                enhanced_context.time_of_day = TimeOfDay.DAY
            else:
                enhanced_context.time_of_day = TimeOfDay.ALL_DAY
        
        return enhanced_context
    
    def parse_multiple_queries(self, queries: List[str]) -> List[ParsedContext]:
        """Parse multiple queries"""
        contexts = []
        for query in queries:
            try:
                context = self.parse_context(query)
                enhanced_context = self.enhance_context_with_defaults(context)
                contexts.append(enhanced_context)
            except Exception as e:
                print(f"Error parsing query '{query}': {e}")
                continue
        return contexts
    
    def export_context(self, context: ParsedContext, output_path: str):
        """Export context to JSON"""
        context_dict = {
            'original_query': context.original_query,
            'occasion': context.occasion.value if context.occasion else None,
            'season': context.season.value if context.season else None,
            'weather': context.weather.value if context.weather else None,
            'time_of_day': context.time_of_day.value if context.time_of_day else None,
            'style_level': context.style_level.value if context.style_level else None,
            'colors': context.colors,
            'patterns': context.patterns,
            'clothing_types': context.clothing_types,
            'keywords': context.keywords,
            'temperature_range': context.temperature_range,
            'confidence': context.confidence,
            'parsing_errors': context.parsing_errors
        }
        
        with open(output_path, 'w') as f:
            json.dump(context_dict, f, indent=2)
        
        print(f"Context exported to: {output_path}")
    
    def get_parsing_statistics(self, contexts: List[ParsedContext]) -> Dict:
        """Get statistics from parsed contexts"""
        if not contexts:
            return {}
        
        occasion_counts = {}
        season_counts = {}
        weather_counts = {}
        time_counts = {}
        style_counts = {}
        
        confidences = []
        
        for context in contexts:
            # Count occurrences
            if context.occasion:
                occasion_counts[context.occasion.value] = occasion_counts.get(context.occasion.value, 0) + 1
            
            if context.season:
                season_counts[context.season.value] = season_counts.get(context.season.value, 0) + 1
            
            if context.weather:
                weather_counts[context.weather.value] = weather_counts.get(context.weather.value, 0) + 1
            
            if context.time_of_day:
                time_counts[context.time_of_day.value] = time_counts.get(context.time_of_day.value, 0) + 1
            
            if context.style_level:
                style_counts[context.style_level.value] = style_counts.get(context.style_level.value, 0) + 1
            
            confidences.append(context.confidence)
        
        return {
            'total_queries': len(contexts),
            'occasion_distribution': occasion_counts,
            'season_distribution': season_counts,
            'weather_distribution': weather_counts,
            'time_distribution': time_counts,
            'style_distribution': style_counts,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'high_confidence_queries': len([c for c in confidences if c >= 0.8])
        }


# Test function
def test_context_parser():
    """Test the context parser"""
    print("Testing Context Parser...")
    
    parser = ContextParser()
    
    # Test queries
    test_queries = [
        "summer casual day outfit",
        "business formal meeting attire",
        "winter cold weather clothes",
        "date night elegant dress",
        "beach vacation sunny outfit",
        "gym workout athletic wear",
        "rainy day comfortable clothes",
        "evening party cocktail dress"
    ]
    
    print(f"Testing {len(test_queries)} queries...")
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        context = parser.parse_context(query)
        enhanced_context = parser.enhance_context_with_defaults(context)
        
        print(f"  Occasion: {enhanced_context.occasion.value if enhanced_context.occasion else 'None'}")
        print(f"  Season: {enhanced_context.season.value if enhanced_context.season else 'None'}")
        print(f"  Weather: {enhanced_context.weather.value if enhanced_context.weather else 'None'}")
        print(f"  Time: {enhanced_context.time_of_day.value if enhanced_context.time_of_day else 'None'}")
        print(f"  Style: {enhanced_context.style_level.value if enhanced_context.style_level else 'None'}")
        print(f"  Colors: {enhanced_context.colors}")
        print(f"  Patterns: {enhanced_context.patterns}")
        print(f"  Confidence: {enhanced_context.confidence:.2f}")
        
        if enhanced_context.parsing_errors:
            print(f"  Errors: {enhanced_context.parsing_errors}")
    
    # Test batch parsing
    print(f"\nBatch parsing...")
    contexts = parser.parse_multiple_queries(test_queries)
    
    stats = parser.get_parsing_statistics(contexts)
    print(f"Parsing statistics: {stats}")
    
    # Test export
    if contexts:
        parser.export_context(contexts[0], "test_context.json")
    
    print("Context parser test completed!")


if __name__ == "__main__":
    test_context_parser()
