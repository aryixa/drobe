"""
Embedding storage and management system
Handles storage, retrieval, and indexing of image embeddings
"""

import sqlite3
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import hashlib

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import DATABASE_PATH, EMBEDDINGS_DIR


class EmbeddingStorage:
    """
    Storage system for image embeddings with metadata
    - SQLite database for metadata
    - File storage for embeddings
    - Indexing and retrieval capabilities
    """
    
    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        print(f"Embedding storage initialized")
        print(f"Database: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create embeddings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT UNIQUE NOT NULL,
                    image_hash TEXT NOT NULL,
                    embedding_path TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    file_size INTEGER,
                    image_width INTEGER,
                    image_height INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create tags table for smart tags integration
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS image_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    tag_type TEXT NOT NULL,  -- 'color', 'type', 'pattern'
                    tag_value TEXT NOT NULL,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_path) REFERENCES embeddings (image_path)
                )
            ''')
            
            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_path ON embeddings (image_path)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_hash ON embeddings (image_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tag_type ON image_tags (tag_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tag_value ON image_tags (tag_value)')
            
            conn.commit()
        
        print("Database tables initialized")
    
    def _get_image_hash(self, image_path: Union[str, Path]) -> str:
        """Generate hash for image file to detect duplicates"""
        image_path = Path(image_path)
        
        try:
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
            return file_hash.hexdigest()
        except Exception as e:
            print(f"Error generating hash for {image_path}: {e}")
            return ""
    
    def store_embedding(self, 
                        image_path: Union[str, Path], 
                        embedding: np.ndarray,
                        metadata: Optional[Dict] = None) -> bool:
        """
        Store an embedding with metadata
        
        Args:
            image_path: Path to original image
            embedding: 512-dimensional embedding vector
            metadata: Additional metadata (width, height, file_size, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        image_path = Path(image_path)
        
        # Generate image hash
        image_hash = self._get_image_hash(image_path)
        
        # Check for duplicates
        if self._embedding_exists(image_hash):
            print(f"Embedding already exists for {image_path}")
            return False
        
        # Save embedding to file
        embedding_filename = f"{image_hash}.npy"
        embedding_path = EMBEDDINGS_DIR / embedding_filename
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save embedding
            np.save(embedding_path, embedding)
            
            # Extract metadata
            file_size = metadata.get('file_size') if metadata else None
            image_width = metadata.get('width') if metadata else None
            image_height = metadata.get('height') if metadata else None
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO embeddings 
                    (image_path, image_hash, embedding_path, embedding_dim, 
                     file_size, image_width, image_height)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(image_path),
                    image_hash,
                    str(embedding_path),
                    embedding.shape[0],
                    file_size,
                    image_width,
                    image_height
                ))
                conn.commit()
            
            print(f"Stored embedding for {image_path}")
            return True
            
        except Exception as e:
            print(f"Error storing embedding for {image_path}: {e}")
            # Clean up embedding file if database insert failed
            if embedding_path.exists():
                embedding_path.unlink()
            return False
    
    def _embedding_exists(self, image_hash: str) -> bool:
        """Check if embedding already exists for given image hash"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM embeddings WHERE image_hash = ?', (image_hash,))
            return cursor.fetchone() is not None
    
    def load_embedding(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load embedding for a specific image
        
        Args:
            image_path: Path to original image
            
        Returns:
            Embedding vector or None if not found
        """
        image_path = Path(image_path)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT embedding_path FROM embeddings 
                WHERE image_path = ?
            ''', (str(image_path),))
            
            result = cursor.fetchone()
            
            if result:
                embedding_path = Path(result[0])
                if embedding_path.exists():
                    return np.load(embedding_path)
                else:
                    print(f"Embedding file not found: {embedding_path}")
            else:
                print(f"No embedding found for {image_path}")
        
        return None
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load all stored embeddings
        
        Returns:
            Dictionary mapping image paths to embeddings
        """
        embeddings = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT image_path, embedding_path FROM embeddings
            ''')
            
            for image_path, embedding_path in cursor.fetchall():
                embedding_path = Path(embedding_path)
                if embedding_path.exists():
                    try:
                        embedding = np.load(embedding_path)
                        embeddings[image_path] = embedding
                    except Exception as e:
                        print(f"Error loading embedding {embedding_path}: {e}")
                else:
                    print(f"Embedding file missing: {embedding_path}")
        
        print(f"Loaded {len(embeddings)} embeddings")
        return embeddings
    
    def get_embedding_matrix(self, image_paths: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Get embeddings as matrix and corresponding image paths
        
        Args:
            image_paths: Optional list of specific image paths
            
        Returns:
            Tuple of (embedding_matrix, image_paths)
        """
        if image_paths:
            embeddings_dict = {}
            for path in image_paths:
                embedding = self.load_embedding(path)
                if embedding is not None:
                    embeddings_dict[path] = embedding
        else:
            embeddings_dict = self.get_all_embeddings()
        
        if not embeddings_dict:
            return np.array([]), []
        
        # Convert to matrix
        image_paths = list(embeddings_dict.keys())
        embedding_matrix = np.array([embeddings_dict[path] for path in image_paths])
        
        return embedding_matrix, image_paths
    
    def add_tags(self, image_path: Union[str, Path], tags: Dict[str, Dict]):
        """
        Add tags for an image (for smart tags integration)
        
        Args:
            image_path: Path to image
            tags: Dictionary of tags {tag_type: {tag_value: confidence}}
        """
        image_path = Path(image_path)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for tag_type, tag_dict in tags.items():
                for tag_value, confidence in tag_dict.items():
                    cursor.execute('''
                        INSERT OR REPLACE INTO image_tags 
                        (image_path, tag_type, tag_value, confidence)
                        VALUES (?, ?, ?, ?)
                    ''', (str(image_path), tag_type, tag_value, confidence))
            
            conn.commit()
        
        print(f"Added tags for {image_path}: {list(tags.keys())}")
    
    def get_tags(self, image_path: Union[str, Path]) -> Dict[str, Dict[str, float]]:
        """
        Get all tags for an image
        
        Args:
            image_path: Path to image
            
        Returns:
            Dictionary of tags {tag_type: {tag_value: confidence}}
        """
        image_path = Path(image_path)
        tags = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT tag_type, tag_value, confidence FROM image_tags
                WHERE image_path = ?
            ''', (str(image_path),))
            
            for tag_type, tag_value, confidence in cursor.fetchall():
                if tag_type not in tags:
                    tags[tag_type] = {}
                tags[tag_type][tag_value] = confidence
        
        return tags
    
    def search_by_tags(self, tag_filters: Dict[str, List[str]]) -> List[str]:
        """
        Search for images by tags
        
        Args:
            tag_filters: Dictionary of tag filters {tag_type: [tag_values]}
            
        Returns:
            List of matching image paths
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build query
            query = '''
                SELECT DISTINCT e.image_path
                FROM embeddings e
                JOIN image_tags t ON e.image_path = t.image_path
                WHERE 1=1
            '''
            params = []
            
            for tag_type, tag_values in tag_filters.items():
                placeholders = ','.join(['?' for _ in tag_values])
                query += f' AND (t.tag_type = ? AND t.tag_value IN ({placeholders}))'
                params.extend([tag_type] + tag_values)
            
            cursor.execute(query, params)
            results = [row[0] for row in cursor.fetchall()]
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get storage statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Embedding stats
            cursor.execute('SELECT COUNT(*) FROM embeddings')
            total_embeddings = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(embedding_dim) FROM embeddings')
            avg_dim = cursor.fetchone()[0] or 0
            
            # Tag stats
            cursor.execute('SELECT COUNT(DISTINCT image_path) FROM image_tags')
            tagged_images = cursor.fetchone()[0]
            
            cursor.execute('SELECT tag_type, COUNT(DISTINCT tag_value) FROM image_tags GROUP BY tag_type')
            tag_stats = dict(cursor.fetchall())
        
        return {
            'total_embeddings': total_embeddings,
            'average_embedding_dim': avg_dim,
            'tagged_images': tagged_images,
            'tag_types': tag_stats,
            'database_size': self.db_path.stat().st_size if self.db_path.exists() else 0
        }


# Test function
def test_embedding_storage():
    """Test the embedding storage system"""
    storage = EmbeddingStorage()
    
    # Test with dummy embedding
    dummy_embedding = np.random.rand(512)
    dummy_metadata = {
        'file_size': 1024,
        'width': 224,
        'height': 224
    }
    
    # Store test embedding
    success = storage.store_embedding(
        "test_image.jpg", 
        dummy_embedding, 
        dummy_metadata
    )
    
    if success:
        print("Test embedding stored successfully")
        
        # Load embedding
        loaded = storage.load_embedding("test_image.jpg")
        if loaded is not None:
            print(f"Embedding loaded successfully: {loaded.shape}")
        
        # Add tags
        tags = {
            'color': {'blue': 0.8, 'dark': 0.6},
            'type': {'shirt': 0.9}
        }
        storage.add_tags("test_image.jpg", tags)
        
        # Get tags
        retrieved_tags = storage.get_tags("test_image.jpg")
        print(f"Retrieved tags: {retrieved_tags}")
        
        # Get stats
        stats = storage.get_statistics()
        print(f"Storage stats: {stats}")
    
    print("Embedding storage test completed!")


if __name__ == "__main__":
    test_embedding_storage()
