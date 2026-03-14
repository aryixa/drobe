"""
Task 13 - RAG + Explainability System: Embedding Index
SentenceTransformer embeddings and FAISS index for fast rule retrieval
"""

import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import SENTENCE_TRANSFORMER_MODEL, FAISS_INDEX_PATH
from .rule_base import RuleBase, FashionRule


class EmbeddingIndex:
    """
    Embedding-based retrieval system for fashion rules
    - SentenceTransformer for text embeddings
    - FAISS for fast similarity search
    - Semantic rule retrieval
    """
    
    def __init__(self, 
                 model_name: str = SENTENCE_TRANSFORMER_MODEL,
                 index_path: Optional[Union[str, Path]] = None):
        self.model_name = model_name
        self.index_path = Path(index_path) if index_path else FAISS_INDEX_PATH
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        # Initialize sentence transformer
        self.model = None
        self._load_model()
        
        # FAISS index
        self.index = None
        self.rule_ids = []  # Map index positions to rule IDs
        
        # Embedding cache
        self.embedding_cache = {}
        
        print(f"Embedding Index initialized with model: {model_name}")
    
    def _load_model(self):
        """Load SentenceTransformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Loaded SentenceTransformer model: {self.model_name}")
            
            # Get embedding dimension
            test_embedding = self.model.encode("test")
            self.embedding_dim = len(test_embedding)
            print(f"Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise
    
    def create_embeddings(self, rules: List[FashionRule]) -> np.ndarray:
        """Create embeddings for fashion rules"""
        print(f"Creating embeddings for {len(rules)} rules...")
        
        # Prepare text for each rule
        rule_texts = []
        self.rule_ids = []
        
        for rule in rules:
            # Combine relevant text fields
            text_parts = [
                rule.title,
                rule.description,
                " ".join(rule.recommendations),
                " ".join(rule.examples),
                rule.category.value,
                rule.priority.value
            ]
            
            # Add conditions
            for key, value in rule.conditions.items():
                text_parts.append(f"{key}: {value}")
            
            # Add occasions and seasons
            text_parts.extend([occ.value for occ in rule.occasions])
            text_parts.extend([sea.value for sea in rule.seasons])
            
            # Combine all text
            full_text = ". ".join(text_parts)
            rule_texts.append(full_text)
            self.rule_ids.append(rule.id)
        
        # Create embeddings
        embeddings = self.model.encode(
            rule_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_index(self, rules: List[FashionRule]):
        """Build FAISS index for rule retrieval"""
        print("Building FAISS index...")
        
        # Create embeddings
        embeddings = self.create_embeddings(rules)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
        
        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
        
        print(f"FAISS index built with {self.index.ntotal} embeddings")
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               min_similarity: float = 0.0) -> List[Tuple[str, float]]:
        """
        Search for similar rules using semantic similarity
        
        Args:
            query: Search query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (rule_id, similarity) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Create query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search index
        similarities, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Process results
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity >= min_similarity and idx < len(self.rule_ids):
                rule_id = self.rule_ids[idx]
                results.append((rule_id, float(similarity)))
        
        return results
    
    def search_by_context(self, 
                         context: Dict[str, Union[str, List[str]]],
                         top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search rules based on outfit context
        
        Args:
            context: Outfit context dictionary
            top_k: Number of results to return
            
        Returns:
            List of (rule_id, similarity) tuples
        """
        # Convert context to search query
        query_parts = []
        
        # Add occasion
        if 'occasion' in context:
            query_parts.append(f"occasion: {context['occasion']}")
        
        # Add season
        if 'season' in context:
            query_parts.append(f"season: {context['season']}")
        
        # Add colors
        if 'colors' in context and context['colors']:
            colors_text = ", ".join(context['colors'])
            query_parts.append(f"colors: {colors_text}")
        
        # Add patterns
        if 'patterns' in context and context['patterns']:
            patterns_text = ", ".join(context['patterns'])
            query_parts.append(f"patterns: {patterns_text}")
        
        # Add style
        if 'style' in context:
            query_parts.append(f"style: {context['style']}")
        
        # Add formality
        if 'formality' in context:
            query_parts.append(f"formality: {context['formality']}")
        
        # Add body type if available
        if 'body_type' in context:
            query_parts.append(f"body type: {context['body_type']}")
        
        # Combine into query
        query = ". ".join(query_parts)
        
        return self.search(query, top_k)
    
    def get_rule_recommendations(self, 
                               context: Dict[str, Union[str, List[str]]],
                               rule_base: RuleBase,
                               top_k: int = 5) -> List[Dict]:
        """
        Get rule recommendations with explanations
        
        Args:
            context: Outfit context
            rule_base: Rule base instance
            top_k: Number of recommendations
            
        Returns:
            List of recommendation dictionaries
        """
        # Search for relevant rules
        search_results = self.search_by_context(context, top_k)
        
        recommendations = []
        
        for rule_id, similarity in search_results:
            rule = rule_base.get_rule(rule_id)
            if rule:
                recommendation = {
                    'rule_id': rule.id,
                    'title': rule.title,
                    'description': rule.description,
                    'category': rule.category.value,
                    'priority': rule.priority.value,
                    'similarity': similarity,
                    'recommendations': rule.recommendations,
                    'examples': rule.examples,
                    'confidence': rule.confidence,
                    'relevance_score': similarity * rule.confidence  # Combined score
                }
                recommendations.append(recommendation)
        
        # Sort by combined relevance score
        recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return recommendations[:top_k]
    
    def save_index(self, index_path: Optional[Union[str, Path]] = None):
        """Save FAISS index and metadata"""
        index_path = Path(index_path) if index_path else self.index_path
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.index is None:
            print("No index to save")
            return
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata = {
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'rule_ids': self.rule_ids,
                'total_rules': len(self.rule_ids)
            }
            
            metadata_path = index_path.with_suffix('.meta')
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"Index saved to: {index_path}")
            print(f"Metadata saved to: {metadata_path}")
            
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def load_index(self, index_path: Optional[Union[str, Path]] = None):
        """Load FAISS index and metadata"""
        index_path = Path(index_path) if index_path else self.index_path
        
        if not index_path.exists():
            print(f"Index file not found: {index_path}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            metadata_path = index_path.with_suffix('.meta')
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.model_name = metadata['model_name']
                self.embedding_dim = metadata['embedding_dim']
                self.rule_ids = metadata['rule_ids']
                
                # Reload model if different
                if self.model is None or self.model_name != self.model._modules['0'].auto_model.name_or_path:
                    self._load_model()
            
            print(f"Index loaded from: {index_path}")
            print(f"Loaded {self.index.ntotal} embeddings")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def update_index(self, rules: List[FashionRule]):
        """Update index with new rules"""
        print("Updating index...")
        
        # Rebuild entire index (simpler approach)
        self.build_index(rules)
        
        # Save updated index
        self.save_index()
        
        print("Index updated successfully")
    
    def get_index_statistics(self) -> Dict:
        """Get index statistics"""
        if self.index is None:
            return {'status': 'not_built'}
        
        stats = {
            'status': 'built',
            'total_rules': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name,
            'index_size_mb': self.index.ntotal * self.embedding_dim * 4 / (1024 * 1024)  # Rough estimate
        }
        
        return stats
    
    def batch_search(self, 
                     queries: List[str], 
                     top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """Batch search for multiple queries"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Create embeddings for all queries
        query_embeddings = self.model.encode(queries, convert_to_numpy=True)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        # Search index
        all_similarities, all_indices = self.index.search(query_embeddings.astype(np.float32), top_k)
        
        # Process results
        batch_results = []
        for similarities, indices in zip(all_similarities, all_indices):
            results = []
            for similarity, idx in zip(similarities, indices):
                if idx < len(self.rule_ids):
                    rule_id = self.rule_ids[idx]
                    results.append((rule_id, float(similarity)))
            batch_results.append(results)
        
        return batch_results
    
    def find_similar_rules(self, 
                          rule_id: str, 
                          rule_base: RuleBase, 
                          top_k: int = 5) -> List[Tuple[FashionRule, float]]:
        """Find rules similar to a given rule"""
        rule = rule_base.get_rule(rule_id)
        if not rule:
            return []
        
        # Create search query from rule
        query_text = f"{rule.title}. {rule.description}. {' '.join(rule.recommendations)}"
        
        # Search for similar rules
        search_results = self.search(query_text, top_k * 2)  # Get more to filter out self
        
        # Filter out the original rule
        similar_rules = []
        for similar_rule_id, similarity in search_results:
            if similar_rule_id != rule_id:
                similar_rule = rule_base.get_rule(similar_rule_id)
                if similar_rule:
                    similar_rules.append((similar_rule, similarity))
            
            if len(similar_rules) >= top_k:
                break
        
        return similar_rules


# Test function
def test_embedding_index():
    """Test the embedding index system"""
    print("Testing Embedding Index...")
    
    # Create rule base
    rule_base = RuleBase()
    rules = rule_base.rules
    
    # Create embedding index
    index = EmbeddingIndex()
    
    # Build index
    index.build_index(rules)
    
    # Test search
    query = "color coordination for business outfit"
    results = index.search(query, top_k=5)
    
    print(f"Search results for '{query}':")
    for rule_id, similarity in results:
        rule = rule_base.get_rule(rule_id)
        print(f"  {rule.title} (similarity: {similarity:.3f})")
    
    # Test context search
    context = {
        'occasion': 'business',
        'season': 'winter',
        'colors': ['blue', 'gray'],
        'patterns': ['solid']
    }
    
    context_results = index.search_by_context(context, top_k=3)
    print(f"\nContext search results:")
    for rule_id, similarity in context_results:
        rule = rule_base.get_rule(rule_id)
        print(f"  {rule.title} (similarity: {similarity:.3f})")
    
    # Test recommendations
    recommendations = index.get_rule_recommendations(context, rule_base, top_k=3)
    print(f"\nRecommendations:")
    for rec in recommendations:
        print(f"  {rec['title']} (relevance: {rec['relevance_score']:.3f})")
    
    # Test saving and loading
    index.save_index()
    
    # Create new index and load
    new_index = EmbeddingIndex()
    loaded = new_index.load_index()
    
    if loaded:
        print("Index loaded successfully")
        
        # Test search with loaded index
        loaded_results = new_index.search(query, top_k=3)
        print(f"Loaded index search results: {len(loaded_results)}")
    
    # Get statistics
    stats = index.get_index_statistics()
    print(f"Index statistics: {stats}")
    
    print("Embedding index test completed!")


if __name__ == "__main__":
    test_embedding_index()
