"""
Task 11 - Outfit Builder (CORE FEATURE)
Main innovation: Combining tops, bottoms, shoes to create optimal outfit combinations
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import itertools
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import json
from collections import defaultdict
import time

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import EMBEDDING_DIM, SIMILARITY_THRESHOLD
from src.embeddings.embedding_storage import EmbeddingStorage
from src.similarity.cosine_sim import CosineSimilarityEngine


class ClothingType(Enum):
    """Clothing item categories"""
    TOP = "top"
    BOTTOM = "bottom"
    SHOES = "shoes"
    DRESS = "dress"
    OUTERWEAR = "outerwear"
    ACCESSORY = "accessory"


class OutfitStyle(Enum):
    """Outfit style categories"""
    CASUAL = "casual"
    FORMAL = "formal"
    SPORTS = "sports"
    BUSINESS = "business"
    PARTY = "party"
    DATE = "date"
    STREET = "street"


@dataclass
class ClothingItem:
    """Individual clothing item representation"""
    path: str
    embedding: np.ndarray
    clothing_type: ClothingType
    style: Optional[OutfitStyle] = None
    color: Optional[str] = None
    pattern: Optional[str] = None
    brand: Optional[str] = None
    season: Optional[str] = None
    formality: Optional[float] = None  # 0-1 scale
    tags: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize embedding"""
        if isinstance(self.embedding, list):
            self.embedding = np.array(self.embedding)
        
        # Ensure embedding is normalized
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            self.embedding = self.embedding / norm


@dataclass
class Outfit:
    """Complete outfit representation"""
    items: List[ClothingItem]
    score: float = 0.0
    style_score: float = 0.0
    color_score: float = 0.0
    pattern_score: float = 0.0
    formality_score: float = 0.0
    embedding: Optional[np.ndarray] = None
    compatibility: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate outfit embedding and scores"""
        if self.embedding is None and self.items:
            self.embedding = self._calculate_outfit_embedding()
            self._calculate_compatibility_scores()
    
    def _calculate_outfit_embedding(self) -> np.ndarray:
        """Calculate outfit embedding as average of item embeddings"""
        if not self.items:
            return np.zeros(EMBEDDING_DIM)
        
        # Average embeddings of all items
        embeddings = np.array([item.embedding for item in self.items])
        outfit_embedding = np.mean(embeddings, axis=0)
        
        # Normalize
        norm = np.linalg.norm(outfit_embedding)
        if norm > 0:
            outfit_embedding = outfit_embedding / norm
        
        return outfit_embedding
    
    def _calculate_compatibility_scores(self):
        """Calculate various compatibility scores"""
        if len(self.items) < 2:
            return
        
        # Style compatibility
        styles = [item.style for item in self.items if item.style]
        if len(styles) > 1:
            style_counts = defaultdict(int)
            for style in styles:
                style_counts[style.value] += 1
            max_count = max(style_counts.values())
            self.style_score = max_count / len(styles)
        
        # Color compatibility (simplified)
        colors = [item.color for item in self.items if item.color]
        if len(colors) > 1:
            unique_colors = len(set(colors))
            self.color_score = 1.0 - (unique_colors - 1) / len(colors)
        
        # Pattern compatibility (avoid conflicting patterns)
        patterns = [item.pattern for item in self.items if item.pattern]
        if len(patterns) > 1:
            solid_count = patterns.count("solid")
            pattern_count = len(patterns) - solid_count
            if pattern_count <= 1:  # At most one patterned item
                self.pattern_score = 1.0
            else:
                self.pattern_score = max(0, 1.0 - (pattern_count - 1) * 0.3)
        
        # Formality consistency
        formalities = [item.formality for item in self.items if item.formality is not None]
        if len(formalities) > 1:
            formality_std = np.std(formalities)
            self.formality_score = max(0, 1.0 - formality_std * 2)
    
    def get_items_by_type(self, clothing_type: ClothingType) -> List[ClothingItem]:
        """Get items of specific type"""
        return [item for item in self.items if item.clothing_type == clothing_type]
    
    def has_complete_set(self) -> bool:
        """Check if outfit has complete set (top, bottom, shoes)"""
        types = {item.clothing_type for item in self.items}
        return {ClothingType.TOP, ClothingType.BOTTOM, ClothingType.SHOES}.issubset(types)


class OutfitBuilder:
    """
    Core outfit building engine
    - Combines tops, bottoms, shoes
    - Computes outfit averages
    - Ranks best combinations
    """
    
    def __init__(self, 
                 storage: Optional[EmbeddingStorage] = None,
                 similarity_engine: Optional[CosineSimilarityEngine] = None):
        self.storage = storage or EmbeddingStorage()
        self.similarity_engine = similarity_engine or CosineSimilarityEngine(self.storage)
        
        # Clothing items cache
        self.items_cache: Dict[ClothingType, List[ClothingItem]] = {}
        
        # Outfit generation parameters
        self.max_combinations = 1000
        self.top_k_outfits = 50
        
        # Load clothing items
        self._load_clothing_items()
        
        print(f"Outfit Builder initialized")
        print(f"Cached items: {sum(len(items) for items in self.items_cache.values())}")
    
    def _load_clothing_items(self):
        """Load and categorize clothing items from storage"""
        print("Loading clothing items...")
        
        # Get all embeddings
        embeddings_dict = self.storage.get_all_embeddings()
        
        for image_path, embedding in embeddings_dict.items():
            # Get tags for this image
            tags = self.storage.get_tags(image_path)
            
            # Determine clothing type from tags
            clothing_type = self._infer_clothing_type(tags)
            
            if clothing_type:
                # Extract additional attributes
                item = ClothingItem(
                    path=image_path,
                    embedding=embedding,
                    clothing_type=clothing_type,
                    color=self._extract_primary_color(tags),
                    pattern=self._extract_pattern(tags),
                    style=self._extract_style(tags),
                    formality=self._extract_formality(tags),
                    tags=tags
                )
                
                if clothing_type not in self.items_cache:
                    self.items_cache[clothing_type] = []
                
                self.items_cache[clothing_type].append(item)
        
        # Print statistics
        for clothing_type, items in self.items_cache.items():
            print(f"  {clothing_type.value}: {len(items)} items")
    
    def _infer_clothing_type(self, tags: Dict[str, Dict]) -> Optional[ClothingType]:
        """Infer clothing type from tags"""
        if 'type' not in tags:
            return None
        
        type_tags = tags['type']
        
        # Map type tags to clothing types
        type_mapping = {
            'shirt': ClothingType.TOP,
            't-shirt': ClothingType.TOP,
            'blouse': ClothingType.TOP,
            'top': ClothingType.TOP,
            'pants': ClothingType.BOTTOM,
            'jeans': ClothingType.BOTTOM,
            'trousers': ClothingType.BOTTOM,
            'skirt': ClothingType.BOTTOM,
            'shorts': ClothingType.BOTTOM,
            'shoes': ClothingType.SHOES,
            'sneakers': ClothingType.SHOES,
            'boots': ClothingType.SHOES,
            'dress': ClothingType.DRESS,
            'jacket': ClothingType.OUTERWEAR,
            'coat': ClothingType.OUTERWEAR,
            'accessory': ClothingType.ACCESSORY
        }
        
        # Find best match
        for tag_value, confidence in type_tags.items():
            tag_lower = tag_value.lower()
            if tag_lower in type_mapping:
                return type_mapping[tag_lower]
        
        return None
    
    def _extract_primary_color(self, tags: Dict[str, Dict]) -> Optional[str]:
        """Extract primary color from tags"""
        if 'color' not in tags:
            return None
        
        color_tags = tags['color']
        if not color_tags:
            return None
        
        # Return color with highest confidence
        return max(color_tags.items(), key=lambda x: x[1])[0]
    
    def _extract_pattern(self, tags: Dict[str, Dict]) -> Optional[str]:
        """Extract pattern from tags"""
        if 'pattern' not in tags:
            return "solid"  # Default to solid
        
        pattern_tags = tags['pattern']
        if not pattern_tags:
            return "solid"
        
        return max(pattern_tags.items(), key=lambda x: x[1])[0]
    
    def _extract_style(self, tags: Dict[str, Dict]) -> Optional[OutfitStyle]:
        """Extract style from tags"""
        if 'style' not in tags:
            return None
        
        style_tags = tags['style']
        if not style_tags:
            return None
        
        # Map to style enum
        style_mapping = {
            'casual': OutfitStyle.CASUAL,
            'formal': OutfitStyle.FORMAL,
            'sports': OutfitStyle.SPORTS,
            'business': OutfitStyle.BUSINESS,
            'party': OutfitStyle.PARTY,
            'date': OutfitStyle.DATE,
            'street': OutfitStyle.STREET
        }
        
        for tag_value, confidence in style_tags.items():
            tag_lower = tag_value.lower()
            if tag_lower in style_mapping:
                return style_mapping[tag_lower]
        
        return None
    
    def _extract_formality(self, tags: Dict[str, Dict]) -> Optional[float]:
        """Extract formality score from tags"""
        if 'formality' not in tags:
            return None
        
        formality_tags = tags['formality']
        if not formality_tags:
            return None
        
        # Return highest confidence as formality score
        return max(formality_tags.values())
    
    def generate_outfit_combinations(self, 
                                   required_types: List[ClothingType] = None,
                                   max_combinations: Optional[int] = None) -> List[Outfit]:
        """
        Generate all possible outfit combinations
        
        Args:
            required_types: Required clothing types for outfits
            max_combinations: Maximum number of combinations to generate
            
        Returns:
            List of outfit combinations
        """
        if required_types is None:
            required_types = [ClothingType.TOP, ClothingType.BOTTOM, ClothingType.SHOES]
        
        max_combinations = max_combinations or self.max_combinations
        
        # Get items for each required type
        type_items = []
        for clothing_type in required_types:
            items = self.items_cache.get(clothing_type, [])
            if not items:
                print(f"Warning: No items found for {clothing_type.value}")
                return []
            type_items.append(items)
        
        # Generate combinations
        combinations = []
        count = 0
        
        print(f"Generating outfit combinations from {len(type_items)} types...")
        
        for item_combo in itertools.product(*type_items):
            if count >= max_combinations:
                break
            
            # Create outfit
            outfit = Outfit(items=list(item_combo))
            combinations.append(outfit)
            count += 1
        
        print(f"Generated {len(combinations)} outfit combinations")
        return combinations
    
    def rank_outfits(self, 
                    outfits: List[Outfit],
                    reference_outfit: Optional[Outfit] = None,
                    style_preference: Optional[OutfitStyle] = None,
                    color_preference: Optional[str] = None,
                    formality_preference: Optional[float] = None) -> List[Outfit]:
        """
        Rank outfits by various criteria
        
        Args:
            outfits: List of outfits to rank
            reference_outfit: Reference outfit for similarity comparison
            style_preference: Preferred style
            color_preference: Preferred color
            formality_preference: Preferred formality level
            
        Returns:
            Ranked list of outfits
        """
        print(f"Ranking {len(outfits)} outfits...")
        
        # Calculate scores for each outfit
        for outfit in outfits:
            score = 0.0
            
            # Base compatibility score
            outfit.score = (outfit.style_score + outfit.color_score + 
                          outfit.pattern_score + outfit.formality_score) / 4.0
            
            # Reference outfit similarity
            if reference_outfit and outfit.embedding is not None:
                similarity = np.dot(outfit.embedding, reference_outfit.embedding)
                outfit.score += similarity * 0.3
            
            # Style preference
            if style_preference:
                style_matches = [item for item in outfit.items if item.style == style_preference]
                if style_matches:
                    outfit.score += len(style_matches) / len(outfit.items) * 0.2
            
            # Color preference
            if color_preference:
                color_matches = [item for item in outfit.items if item.color == color_preference]
                if color_matches:
                    outfit.score += len(color_matches) / len(outfit.items) * 0.1
            
            # Formality preference
            if formality_preference is not None:
                outfit_formalities = [item.formality for item in outfit.items if item.formality is not None]
                if outfit_formalities:
                    avg_formality = np.mean(outfit_formalities)
                    formality_diff = abs(avg_formality - formality_preference)
                    outfit.score += (1.0 - formality_diff) * 0.1
        
        # Sort by score
        ranked_outfits = sorted(outfits, key=lambda x: x.score, reverse=True)
        
        print(f"Ranked outfits (top 5 scores): {[f'{o.score:.3f}' for o in ranked_outfits[:5]]}")
        return ranked_outfits
    
    def build_outfit_from_items(self, 
                               top_item: Union[str, ClothingItem],
                               bottom_item: Union[str, ClothingItem],
                               shoes_item: Union[str, ClothingItem]) -> Outfit:
        """
        Build outfit from specific items
        
        Args:
            top_item: Top item (path or ClothingItem)
            bottom_item: Bottom item (path or ClothingItem)
            shoes_item: Shoes item (path or ClothingItem)
            
        Returns:
            Complete outfit
        """
        items = []
        
        # Convert paths to ClothingItem objects
        for item, clothing_type in [(top_item, ClothingType.TOP), 
                                   (bottom_item, ClothingType.BOTTOM), 
                                   (shoes_item, ClothingType.SHOES)]:
            if isinstance(item, str):
                # Find item by path
                found_item = None
                for cached_item in self.items_cache.get(clothing_type, []):
                    if cached_item.path == item:
                        found_item = cached_item
                        break
                
                if found_item is None:
                    # Load from storage
                    embedding = self.storage.load_embedding(item)
                    if embedding is not None:
                        tags = self.storage.get_tags(item)
                        found_item = ClothingItem(
                            path=item,
                            embedding=embedding,
                            clothing_type=clothing_type,
                            tags=tags
                        )
                
                if found_item:
                    items.append(found_item)
            else:
                items.append(item)
        
        if len(items) != 3:
            raise ValueError("Could not find all required items")
        
        return Outfit(items=items)
    
    def find_similar_outfits(self, 
                           reference_outfit: Outfit,
                           top_k: int = 10) -> List[Outfit]:
        """
        Find outfits similar to reference outfit
        
        Args:
            reference_outfit: Reference outfit
            top_k: Number of similar outfits to return
            
        Returns:
            List of similar outfits
        """
        if reference_outfit.embedding is None:
            return []
        
        # Generate combinations
        combinations = self.generate_outfit_combinations()
        
        # Calculate similarities
        similar_outfits = []
        for outfit in combinations:
            if outfit.embedding is not None:
                similarity = np.dot(reference_outfit.embedding, outfit.embedding)
                outfit.score = similarity
                similar_outfits.append(outfit)
        
        # Sort by similarity and return top k
        similar_outfits.sort(key=lambda x: x.score, reverse=True)
        return similar_outfits[:top_k]
    
    def get_outfit_suggestions(self, 
                             partial_items: List[Union[str, ClothingItem]],
                             suggestions_per_type: int = 5) -> Dict[ClothingType, List[ClothingItem]]:
        """
        Get suggestions to complete a partial outfit
        
        Args:
            partial_items: List of existing outfit items
            suggestions_per_type: Number of suggestions per type
            
        Returns:
            Dictionary mapping types to suggested items
        """
        # Convert partial items to ClothingItem objects
        items = []
        for item in partial_items:
            if isinstance(item, str):
                # Find in cache
                found = None
                for clothing_type, cached_items in self.items_cache.items():
                    for cached_item in cached_items:
                        if cached_item.path == item:
                            found = cached_item
                            break
                    if found:
                        break
                if found:
                    items.append(found)
            else:
                items.append(item)
        
        if not items:
            return {}
        
        # Calculate partial outfit embedding
        partial_embedding = np.mean([item.embedding for item in items], axis=0)
        partial_embedding = partial_embedding / np.linalg.norm(partial_embedding)
        
        # Find missing types
        existing_types = {item.clothing_type for item in items}
        required_types = {ClothingType.TOP, ClothingType.BOTTOM, ClothingType.SHOES}
        missing_types = required_types - existing_types
        
        suggestions = {}
        
        for missing_type in missing_types:
            type_items = self.items_cache.get(missing_type, [])
            if not type_items:
                continue
            
            # Calculate similarities
            item_similarities = []
            for item in type_items:
                similarity = np.dot(partial_embedding, item.embedding)
                item_similarities.append((item, similarity))
            
            # Sort and take top suggestions
            item_similarities.sort(key=lambda x: x[1], reverse=True)
            suggestions[missing_type] = [item for item, _ in item_similarities[:suggestions_per_type]]
        
        return suggestions
    
    def optimize_outfit(self, 
                       initial_outfit: Outfit,
                       optimization_target: str = "overall") -> Outfit:
        """
        Optimize an outfit for specific criteria
        
        Args:
            initial_outfit: Initial outfit to optimize
            optimization_target: Target to optimize ("style", "color", "formality", "overall")
            
        Returns:
            Optimized outfit
        """
        # Get suggestions for each item type
        suggestions = self.get_outfit_suggestions(initial_outfit.items, suggestions_per_type=10)
        
        best_outfit = initial_outfit
        best_score = initial_outfit.score
        
        # Try different combinations
        for clothing_type, suggested_items in suggestions.items():
            if not suggested_items:
                continue
            
            # Replace each item type with suggestions
            current_items_by_type = {item.clothing_type: item for item in initial_outfit.items}
            
            for suggested_item in suggested_items:
                # Create new outfit with suggested item
                new_items = []
                for req_type in [ClothingType.TOP, ClothingType.BOTTOM, ClothingType.SHOES]:
                    if req_type == clothing_type:
                        new_items.append(suggested_item)
                    else:
                        new_items.append(current_items_by_type[req_type])
                
                new_outfit = Outfit(items=new_items)
                
                # Re-rank with optimization target
                ranked = self.rank_outfits([new_outfit])
                
                if ranked and ranked[0].score > best_score:
                    best_outfit = ranked[0]
                    best_score = ranked[0].score
        
        return best_outfit
    
    def get_statistics(self) -> Dict:
        """Get outfit builder statistics"""
        stats = {
            'total_items': sum(len(items) for items in self.items_cache.values()),
            'items_by_type': {clothing_type.value: len(items) 
                            for clothing_type, items in self.items_cache.items()},
            'available_types': list(self.items_cache.keys()),
            'max_combinations': self.max_combinations,
            'top_k_outfits': self.top_k_outfits
        }
        
        # Add style distribution if available
        style_counts = defaultdict(int)
        for items in self.items_cache.values():
            for item in items:
                if item.style:
                    style_counts[item.style.value] += 1
        
        stats['style_distribution'] = dict(style_counts)
        
        return stats


# Test function
def test_outfit_builder():
    """Test the outfit builder"""
    print("Testing Outfit Builder...")
    
    # Create dummy storage and data
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    storage = EmbeddingStorage(temp_dir / "test.db")
    
    # Create dummy clothing items
    np.random.seed(42)
    
    # Create dummy embeddings and tags
    dummy_items = [
        ("top1.jpg", ClothingType.TOP, "blue", "solid", OutfitStyle.CASUAL, 0.3),
        ("top2.jpg", ClothingType.TOP, "white", "striped", OutfitStyle.FORMAL, 0.8),
        ("bottom1.jpg", ClothingType.BOTTOM, "blue", "solid", OutfitStyle.CASUAL, 0.4),
        ("bottom2.jpg", ClothingType.BOTTOM, "black", "solid", OutfitStyle.FORMAL, 0.9),
        ("shoes1.jpg", ClothingType.SHOES, "white", "solid", OutfitStyle.CASUAL, 0.2),
        ("shoes2.jpg", ClothingType.SHOES, "black", "solid", OutfitStyle.FORMAL, 0.7)
    ]
    
    for path, clothing_type, color, pattern, style, formality in dummy_items:
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
    builder = OutfitBuilder(storage)
    
    # Test statistics
    stats = builder.get_statistics()
    print(f"Builder stats: {stats}")
    
    # Test outfit generation
    combinations = builder.generate_outfit_combinations(max_combinations=10)
    print(f"Generated {len(combinations)} combinations")
    
    # Test outfit building from specific items
    outfit = builder.build_outfit_from_items("top1.jpg", "bottom1.jpg", "shoes1.jpg")
    print(f"Built outfit with {len(outfit.items)} items")
    print(f"Outfit score: {outfit.score:.3f}")
    print(f"Style score: {outfit.style_score:.3f}")
    print(f"Color score: {outfit.color_score:.3f}")
    
    # Test ranking
    if combinations:
        ranked = builder.rank_outfits(combinations[:5])
        print(f"Top ranked outfit score: {ranked[0].score:.3f}")
    
    # Test suggestions
    suggestions = builder.get_outfit_suggestions(["top1.jpg"])
    print(f"Suggestions: {list(suggestions.keys())}")
    
    print("Outfit builder test completed!")


if __name__ == "__main__":
    test_outfit_builder()
