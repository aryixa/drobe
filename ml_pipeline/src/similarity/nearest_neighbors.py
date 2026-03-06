"""
Nearest neighbor search algorithms for similarity system
Optimized search methods for large-scale similarity queries
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import heapq
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from dataclasses import dataclass
from enum import Enum
import time

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import EMBEDDING_DIM
from src.embeddings.embedding_storage import EmbeddingStorage


class SearchMethod(Enum):
    """Search method types"""
    BRUTE_FORCE = "brute_force"
    KDTREE = "kdtree"
    BALLTREE = "balltree"
    FAISS = "faiss"
    HNSW = "hnsw"


@dataclass
class SearchResult:
    """Search result data structure"""
    image_path: str
    similarity: float
    index: int
    distance: Optional[float] = None


class NearestNeighborSearch:
    """
    Optimized nearest neighbor search for similarity queries
    - Multiple search algorithms
    - Performance optimization
    - Scalable indexing
    """
    
    def __init__(self, 
                 storage: Optional[EmbeddingStorage] = None,
                 method: SearchMethod = SearchMethod.FAISS):
        self.storage = storage or EmbeddingStorage()
        self.method = method
        self.embedding_dim = EMBEDDING_DIM
        
        # Search structures
        self.search_index = None
        self.embeddings_matrix = None
        self.image_paths = []
        
        # Performance metrics
        self.search_times = []
        
        # Initialize
        self._load_embeddings()
        self._build_index()
        
        print(f"Nearest Neighbor Search initialized")
        print(f"Method: {method.value}")
        print(f"Embeddings loaded: {len(self.image_paths)}")
    
    def _load_embeddings(self):
        """Load embeddings from storage"""
        try:
            self.embeddings_matrix, self.image_paths = self.storage.get_embedding_matrix()
            
            if len(self.image_paths) > 0:
                # Normalize for cosine similarity
                from sklearn.preprocessing import normalize
                self.embeddings_matrix = normalize(self.embeddings_matrix, norm='l2')
            
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            self.embeddings_matrix = np.array([]).reshape(0, self.embedding_dim)
            self.image_paths = []
    
    def _build_index(self):
        """Build search index based on selected method"""
        if len(self.embeddings_matrix) == 0:
            return
        
        start_time = time.time()
        
        try:
            if self.method == SearchMethod.BRUTE_FORCE:
                self._build_brute_force()
            elif self.method == SearchMethod.KDTREE:
                self._build_kdtree()
            elif self.method == SearchMethod.BALLTREE:
                self._build_balltree()
            elif self.method == SearchMethod.FAISS:
                self._build_faiss_index()
            elif self.method == SearchMethod.HNSW:
                self._build_hnsw_index()
            
            build_time = time.time() - start_time
            print(f"Index built in {build_time:.2f}s using {self.method.value}")
            
        except Exception as e:
            print(f"Error building index: {e}")
            self.search_index = None
    
    def _build_brute_force(self):
        """Build brute-force search (just store embeddings)"""
        self.search_index = self.embeddings_matrix
    
    def _build_kdtree(self):
        """Build KD-Tree index"""
        # Note: KD-Tree works with Euclidean distance, not cosine
        # We'll use it for approximate search
        self.search_index = NearestNeighbors(
            n_neighbors=10,
            algorithm='kd_tree',
            metric='euclidean'
        )
        self.search_index.fit(self.embeddings_matrix)
    
    def _build_balltree(self):
        """Build Ball-Tree index"""
        self.search_index = NearestNeighbors(
            n_neighbors=10,
            algorithm='ball_tree',
            metric='euclidean'
        )
        self.search_index.fit(self.embeddings_matrix)
    
    def _build_faiss_index(self):
        """Build FAISS index for fast search"""
        # Use IndexFlatIP for cosine similarity (inner product on normalized vectors)
        self.search_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index
        embeddings_float32 = self.embeddings_matrix.astype(np.float32)
        self.search_index.add(embeddings_float32)
    
    def _build_hnsw_index(self):
        """Build HNSW index for approximate nearest neighbor"""
        # HNSW is great for large datasets
        M = 16  # Number of connections
        ef_construction = 200  # Construction parameter
        
        self.search_index = faiss.IndexHNSWFlat(self.embedding_dim, M)
        self.search_index.hnsw.efConstruction = ef_construction
        
        # Add embeddings
        embeddings_float32 = self.embeddings_matrix.astype(np.float32)
        self.search_index.add(embeddings_float32)
        
        # Set search parameter
        self.search_index.hnsw.efSearch = 50
    
    def search(self, 
              query_embedding: np.ndarray,
              k: int = 10,
              threshold: float = 0.0) -> List[SearchResult]:
        """
        Search for nearest neighbors
        
        Args:
            query_embedding: Query embedding vector
            k: Number of neighbors to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        if len(self.image_paths) == 0:
            return []
        
        start_time = time.time()
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)
        
        results = []
        
        try:
            if self.method == SearchMethod.BRUTE_FORCE:
                results = self._search_brute_force(query_embedding, k, threshold)
            elif self.method in [SearchMethod.KDTREE, SearchMethod.BALLTREE]:
                results = self._search_sklearn(query_embedding, k, threshold)
            elif self.method in [SearchMethod.FAISS, SearchMethod.HNSW]:
                results = self._search_faiss(query_embedding, k, threshold)
            
        except Exception as e:
            print(f"Search error: {e}")
            results = []
        
        search_time = time.time() - start_time
        self.search_times.append(search_time)
        
        return results
    
    def _search_brute_force(self, 
                           query_embedding: np.ndarray,
                           k: int,
                           threshold: float) -> List[SearchResult]:
        """Brute-force search using cosine similarity"""
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        
        # Create results
        results = []
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                results.append(SearchResult(
                    image_path=self.image_paths[i],
                    similarity=float(similarity),
                    index=i
                ))
        
        # Sort by similarity and take top k
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]
    
    def _search_sklearn(self, 
                       query_embedding: np.ndarray,
                       k: int,
                       threshold: float) -> List[SearchResult]:
        """Search using sklearn NearestNeighbors"""
        distances, indices = self.search_index.kneighbors(query_embedding, k=k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.image_paths):
                # Convert Euclidean distance to approximate cosine similarity
                # This is approximate since we're using different distance metrics
                similarity = 1.0 / (1.0 + distance)
                
                if similarity >= threshold:
                    results.append(SearchResult(
                        image_path=self.image_paths[idx],
                        similarity=float(similarity),
                        index=int(idx),
                        distance=float(distance)
                    ))
        
        return results
    
    def _search_faiss(self, 
                     query_embedding: np.ndarray,
                     k: int,
                     threshold: float) -> List[SearchResult]:
        """Search using FAISS index"""
        query_float32 = query_embedding.astype(np.float32)
        
        # Search
        similarities, indices = self.search_index.search(query_float32, k)
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.image_paths) and similarity >= threshold:
                results.append(SearchResult(
                    image_path=self.image_paths[idx],
                    similarity=float(similarity),
                    index=int(idx)
                ))
        
        return results
    
    def batch_search(self, 
                     query_embeddings: List[np.ndarray],
                     k: int = 10,
                     threshold: float = 0.0) -> List[List[SearchResult]]:
        """
        Batch search for multiple queries
        
        Args:
            query_embeddings: List of query embeddings
            k: Number of neighbors per query
            threshold: Minimum similarity threshold
            
        Returns:
            List of results for each query
        """
        all_results = []
        
        for embedding in query_embeddings:
            results = self.search(embedding, k, threshold)
            all_results.append(results)
        
        return all_results
    
    def range_search(self, 
                     query_embedding: np.ndarray,
                     radius: float = 0.1) -> List[SearchResult]:
        """
        Range search - find all neighbors within radius
        
        Args:
            query_embedding: Query embedding vector
            radius: Search radius (similarity threshold)
            
        Returns:
            List of all neighbors within radius
        """
        # Use k = total number of items, then filter by radius
        max_k = len(self.image_paths)
        results = self.search(query_embedding, k=max_k, threshold=radius)
        
        return results
    
    def approximate_search(self, 
                          query_embedding: np.ndarray,
                          k: int = 10,
                          ef_search: int = 50) -> List[SearchResult]:
        """
        Approximate search (for HNSW index)
        
        Args:
            query_embedding: Query embedding vector
            k: Number of neighbors
            ef_search: Search parameter for HNSW
            
        Returns:
            List of search results
        """
        if self.method == SearchMethod.HNSW and self.search_index is not None:
            # Adjust search parameter
            original_ef = self.search_index.hnsw.efSearch
            self.search_index.hnsw.efSearch = ef_search
            
            results = self.search(query_embedding, k)
            
            # Restore original parameter
            self.search_index.hnsw.efSearch = original_ef
            
            return results
        else:
            # Fall back to regular search
            return self.search(query_embedding, k)
    
    def get_performance_stats(self) -> Dict:
        """Get search performance statistics"""
        if not self.search_times:
            return {}
        
        stats = {
            'total_searches': len(self.search_times),
            'avg_search_time': np.mean(self.search_times),
            'min_search_time': np.min(self.search_times),
            'max_search_time': np.max(self.search_times),
            'std_search_time': np.std(self.search_times),
            'method': self.method.value,
            'index_size': len(self.image_paths)
        }
        
        # Add method-specific stats
        if self.method in [SearchMethod.FAISS, SearchMethod.HNSW] and self.search_index:
            stats['faiss_index_size'] = self.search_index.ntotal
        
        return stats
    
    def save_index(self, save_path: Union[str, Path]):
        """Save search index to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.method in [SearchMethod.FAISS, SearchMethod.HNSW]:
                faiss.write_index(self.search_index, str(save_path))
                
                # Save metadata
                metadata = {
                    'method': self.method.value,
                    'image_paths': self.image_paths,
                    'embedding_dim': self.embedding_dim
                }
                
                import pickle
                metadata_path = save_path.with_suffix('.meta')
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
                
            print(f"Index saved to: {save_path}")
            
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def load_index(self, load_path: Union[str, Path]):
        """Load search index from file"""
        load_path = Path(load_path)
        
        if not load_path.exists():
            print(f"Index file not found: {load_path}")
            return
        
        try:
            if self.method in [SearchMethod.FAISS, SearchMethod.HNSW]:
                self.search_index = faiss.read_index(str(load_path))
                
                # Load metadata
                metadata_path = load_path.with_suffix('.meta')
                if metadata_path.exists():
                    import pickle
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    
                    self.image_paths = metadata['image_paths']
                    self.embedding_dim = metadata['embedding_dim']
                
            print(f"Index loaded from: {load_path}")
            print(f"Index contains {self.search_index.ntotal} embeddings")
            
        except Exception as e:
            print(f"Error loading index: {e}")
            self.search_index = None
    
    def compare_methods(self, 
                       query_embedding: np.ndarray,
                       k: int = 10) -> Dict[str, Dict]:
        """
        Compare different search methods
        
        Args:
            query_embedding: Query embedding vector
            k: Number of neighbors
            
        Returns:
            Dictionary comparing method performance
        """
        methods = [SearchMethod.BRUTE_FORCE, SearchMethod.FAISS, SearchMethod.HNSW]
        comparison = {}
        
        for method in methods:
            # Temporarily change method
            original_method = self.method
            self.method = method
            
            # Rebuild index
            self._build_index()
            
            # Time search
            start_time = time.time()
            results = self.search(query_embedding, k)
            search_time = time.time() - start_time
            
            comparison[method.value] = {
                'search_time': search_time,
                'results_count': len(results),
                'top_similarity': results[0].similarity if results else 0.0
            }
            
            # Restore original method
            self.method = original_method
            self._build_index()
        
        return comparison


# Test function
def test_nearest_neighbors():
    """Test nearest neighbor search"""
    print("Testing Nearest Neighbor Search...")
    
    # Create test data
    np.random.seed(42)
    test_embeddings = np.random.rand(100, 512)
    test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    
    # Create dummy storage
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    storage = EmbeddingStorage(temp_dir / "test.db")
    
    # Store test embeddings
    for i, embedding in enumerate(test_embeddings):
        storage.store_embedding(f"test_{i}.jpg", embedding)
    
    # Test different methods
    methods = [SearchMethod.BRUTE_FORCE, SearchMethod.FAISS]
    
    for method in methods:
        print(f"\nTesting {method.value}...")
        
        nn_search = NearestNeighborSearch(storage, method)
        
        if len(nn_search.image_paths) > 0:
            # Test single search
            query = test_embeddings[0]
            results = nn_search.search(query, k=5)
            
            print(f"Found {len(results)} results")
            if results:
                print(f"Top result: {results[0].image_path} (similarity: {results[0].similarity:.4f})")
            
            # Test batch search
            batch_queries = test_embeddings[:3]
            batch_results = nn_search.batch_search(batch_queries, k=3)
            
            print(f"Batch search completed for {len(batch_results)} queries")
            
            # Performance stats
            stats = nn_search.get_performance_stats()
            print(f"Performance: {stats}")
    
    print("Nearest neighbor search test completed!")


if __name__ == "__main__":
    test_nearest_neighbors()
