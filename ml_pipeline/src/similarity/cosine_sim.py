"""
Task 10 - Similarity System
Cosine similarity calculations for finding similar outfits
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import faiss
from tqdm import tqdm

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import (
    SIMILARITY_THRESHOLD, 
    EMBEDDINGS_DIR, 
    FAISS_INDEX_PATH,
    EMBEDDING_DIM
)
from src.embeddings.embedding_storage import EmbeddingStorage


class CosineSimilarityEngine:
    """
    Cosine similarity engine for finding similar clothing items
    - Fast similarity calculations
    - Nearest neighbor search
    - Ranking and filtering
    """
    
    def __init__(self, storage: Optional[EmbeddingStorage] = None):
        self.storage = storage or EmbeddingStorage()
        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.embedding_dim = EMBEDDING_DIM
        
        # FAISS index for fast search
        self.faiss_index = None
        self.image_paths = []
        self.embeddings_matrix = None
        
        # Initialize with existing embeddings
        self._load_embeddings()
        
        print(f"Cosine Similarity Engine initialized")
        print(f"Similarity threshold: {self.similarity_threshold}")
        print(f"Loaded embeddings: {len(self.image_paths)}")
    
    def _load_embeddings(self):
        """Load embeddings from storage"""
        try:
            self.embeddings_matrix, self.image_paths = self.storage.get_embedding_matrix()
            
            if len(self.image_paths) > 0:
                # Normalize embeddings for cosine similarity
                self.embeddings_matrix = normalize(self.embeddings_matrix, norm='l2')
                
                # Build FAISS index
                self._build_faiss_index()
            
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            self.embeddings_matrix = np.array([]).reshape(0, self.embedding_dim)
            self.image_paths = []
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        if len(self.embeddings_matrix) == 0:
            return
        
        try:
            # Create FAISS index for inner product (cosine similarity on normalized vectors)
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Add embeddings to index
            self.faiss_index.add(self.embeddings_matrix.astype(np.float32))
            
            print(f"FAISS index built with {self.faiss_index.ntotal} embeddings")
            
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            self.faiss_index = None
    
    def compute_cosine_similarity(self, 
                                 embedding1: np.ndarray, 
                                 embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize embeddings
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Ensure result is in [0, 1] range
        similarity = max(0, min(1, float(similarity)))
        
        return similarity
    
    def find_similar_items(self, 
                          query_embedding: np.ndarray,
                          top_k: int = 10,
                          threshold: Optional[float] = None) -> List[Dict]:
        """
        Find similar items using cosine similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of similar items to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar items with similarity scores
        """
        if len(self.image_paths) == 0:
            return []
        
        threshold = threshold or self.similarity_threshold
        
        # Normalize query embedding
        query_embedding = queryding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Use FAISS if available (much faster)
        if self.faiss_index is not None:
            similarities, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.image_paths)))
            
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if similarity >= threshold and idx < len(self.image_paths):
                    results.append({
                        'image_path': self.image_paths[idx],
                        'similarity': float(similarity),
                        'index': int(idx)
                    })
        else:
            # Fallback to sklearn (slower)
            similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
            
            results = []
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    results.append({
                        'image_path': self.image_paths[i],
                        'similarity': float(similarity),
                        'index': i
                    })
            
            # Sort by similarity and take top_k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            results = results[:top_k]
        
        return results
    
    def find_similar_by_path(self, 
                            image_path: Union[str, Path],
                            top_k: int = 10,
                            threshold: Optional[float] = None) -> List[Dict]:
        """
        Find similar items by image path
        
        Args:
            image_path: Path to query image
            top_k: Number of similar items to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar items with similarity scores
        """
        # Get embedding for query image
        query_embedding = self.storage.load_embedding(image_path)
        
        if query_embedding is None:
            raise ValueError(f"No embedding found for {image_path}")
        
        return self.find_similar_items(query_embedding, top_k, threshold)
    
    def batch_similarity(self, 
                        query_embeddings: List[np.ndarray],
                        top_k: int = 10,
                        threshold: Optional[float] = None) -> List[List[Dict]]:
        """
        Find similar items for multiple query embeddings
        
        Args:
            query_embeddings: List of query embedding vectors
            top_k: Number of similar items to return per query
            threshold: Minimum similarity threshold
            
        Returns:
            List of results for each query
        """
        results = []
        
        for embedding in tqdm(query_embeddings, desc="Computing similarities"):
            similar_items = self.find_similar_items(embedding, top_k, threshold)
            results.append(similar_items)
        
        return results
    
    def compute_similarity_matrix(self, 
                                 image_paths: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise similarity matrix
        
        Args:
            image_paths: Optional list of specific image paths
            
        Returns:
            Tuple of (similarity_matrix, image_paths)
        """
        if image_paths is None:
            if len(self.image_paths) == 0:
                return np.array([]), []
            similarity_matrix = cosine_similarity(self.embeddings_matrix)
            return similarity_matrix, self.image_paths
        
        # Compute for specific images
        embeddings_list = []
        valid_paths = []
        
        for path in image_paths:
            embedding = self.storage.load_embedding(path)
            if embedding is not None:
                embeddings_list.append(embedding)
                valid_paths.append(path)
        
        if not embeddings_list:
            return np.array([]), []
        
        embeddings_matrix = np.array(embeddings_list)
        embeddings_matrix = normalize(embeddings_matrix, norm='l2')
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        return similarity_matrix, valid_paths
    
    def get_outfit_suggestions(self, 
                               item_paths: List[str],
                               exclude_paths: Optional[List[str]] = None,
                               top_k: int = 5) -> Dict[str, List[Dict]]:
        """
        Get outfit suggestions based on similar items for each clothing piece
        
        Args:
            item_paths: List of current outfit items
            exclude_paths: Paths to exclude from suggestions
            top_k: Number of suggestions per item
            
        Returns:
            Dictionary mapping each item to its suggestions
        """
        exclude_paths = exclude_paths or []
        suggestions = {}
        
        for item_path in item_paths:
            # Find similar items
            similar_items = self.find_similar_by_path(item_path, top_k * 2)  # Get more to filter
            
            # Filter out current items and excluded items
            filtered_items = []
            for item in similar_items:
                if (item['image_path'] not in item_paths and 
                    item['image_path'] not in exclude_paths):
                    filtered_items.append(item)
            
            suggestions[item_path] = filtered_items[:top_k]
        
        return suggestions
    
    def rank_by_similarity(self, 
                           candidate_paths: List[str],
                           reference_path: str,
                           top_k: Optional[int] = None) -> List[Dict]:
        """
        Rank candidate items by similarity to reference item
        
        Args:
            candidate_paths: List of candidate item paths
            reference_path: Reference item path
            top_k: Number of top items to return
            
        Returns:
            Ranked list of candidates with similarity scores
        """
        reference_embedding = self.storage.load_embedding(reference_path)
        if reference_embedding is None:
            raise ValueError(f"No embedding found for reference: {reference_path}")
        
        # Compute similarities for all candidates
        candidate_similarities = []
        
        for candidate_path in candidate_paths:
            candidate_embedding = self.storage.load_embedding(candidate_path)
            if candidate_embedding is not None:
                similarity = self.compute_cosine_similarity(reference_embedding, candidate_embedding)
                candidate_similarities.append({
                    'image_path': candidate_path,
                    'similarity': similarity
                })
        
        # Sort by similarity
        candidate_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        if top_k:
            candidate_similarities = candidate_similarities[:top_k]
        
        return candidate_similarities
    
    def save_index(self, save_path: Optional[Union[str, Path]] = None):
        """
        Save FAISS index and metadata
        
        Args:
            save_path: Path to save index (default: FAISS_INDEX_PATH)
        """
        if self.faiss_index is None:
            print("No FAISS index to save")
            return
        
        save_path = Path(save_path) if save_path else FAISS_INDEX_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save FAISS index
            faiss.write_index(self.faiss_index, str(save_path))
            
            # Save metadata
            metadata = {
                'image_paths': self.image_paths,
                'embedding_dim': self.embedding_dim,
                'similarity_threshold': self.similarity_threshold
            }
            
            metadata_path = save_path.with_suffix('.meta')
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"FAISS index saved to: {save_path}")
            
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
    
    def load_index(self, load_path: Optional[Union[str, Path]] = None):
        """
        Load FAISS index and metadata
        
        Args:
            load_path: Path to load index (default: FAISS_INDEX_PATH)
        """
        load_path = Path(load_path) if load_path else FAISS_INDEX_PATH
        
        if not load_path.exists():
            print(f"FAISS index not found: {load_path}")
            return
        
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index(str(load_path))
            
            # Load metadata
            metadata_path = load_path.with_suffix('.meta')
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.image_paths = metadata['image_paths']
                self.embedding_dim = metadata['embedding_dim']
                self.similarity_threshold = metadata.get('similarity_threshold', SIMILARITY_THRESHOLD)
            
            print(f"FAISS index loaded from: {load_path}")
            print(f"Index contains {self.faiss_index.ntotal} embeddings")
            
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            self.faiss_index = None
    
    def get_statistics(self) -> Dict:
        """Get similarity engine statistics"""
        stats = {
            'total_embeddings': len(self.image_paths),
            'embedding_dim': self.embedding_dim,
            'similarity_threshold': self.similarity_threshold,
            'faiss_index_built': self.faiss_index is not None,
            'index_size': self.faiss_index.ntotal if self.faiss_index else 0
        }
        
        if len(self.image_paths) > 0:
            # Compute some similarity statistics
            sample_size = min(100, len(self.image_paths))
            sample_indices = np.random.choice(len(self.image_paths), sample_size, replace=False)
            sample_embeddings = self.embeddings_matrix[sample_indices]
            
            # Compute pairwise similarities for sample
            sample_similarities = cosine_similarity(sample_embeddings)
            
            # Remove self-similarities (diagonal)
            mask = ~np.eye(sample_similarities.shape[0], dtype=bool)
            sample_similarities = sample_similarities[mask]
            
            stats.update({
                'avg_similarity': float(np.mean(sample_similarities)),
                'std_similarity': float(np.std(sample_similarities)),
                'min_similarity': float(np.min(sample_similarities)),
                'max_similarity': float(np.max(sample_similarities))
            })
        
        return stats


# Test function
def test_cosine_similarity():
    """Test the cosine similarity engine"""
    print("Testing Cosine Similarity Engine...")
    
    # Create test embeddings
    np.random.seed(42)
    test_embeddings = np.random.rand(10, 512)
    test_embeddings = normalize(test_embeddings, norm='l2')
    
    # Test similarity computation
    engine = CosineSimilarityEngine()
    
    # Test single similarity
    similarity = engine.compute_cosine_similarity(test_embeddings[0], test_embeddings[1])
    print(f"Similarity between embeddings 0 and 1: {similarity:.4f}")
    
    # Test self-similarity
    self_sim = engine.compute_cosine_similarity(test_embeddings[0], test_embeddings[0])
    print(f"Self-similarity: {self_sim:.4f}")
    
    # Test with dummy storage
    from src.embeddings.embedding_storage import EmbeddingStorage
    import tempfile
    
    temp_dir = Path(tempfile.mkdtemp())
    storage = EmbeddingStorage(temp_dir / "test.db")
    
    # Store test embeddings
    for i, embedding in enumerate(test_embeddings):
        storage.store_embedding(f"test_{i}.jpg", embedding)
    
    # Create engine with storage
    engine_with_storage = CosineSimilarityEngine(storage)
    
    if len(engine_with_storage.image_paths) > 0:
        # Test finding similar items
        similar_items = engine_with_storage.find_similar_items(test_embeddings[0], top_k=3)
        print(f"Found {len(similar_items)} similar items")
        
        # Test similarity matrix
        sim_matrix, paths = engine_with_storage.compute_similarity_matrix()
        print(f"Similarity matrix shape: {sim_matrix.shape}")
        
        # Get statistics
        stats = engine_with_storage.get_statistics()
        print(f"Engine stats: {stats}")
    
    print("Cosine similarity test completed!")


if __name__ == "__main__":
    test_cosine_similarity()
