"""
Task 9 - Image Embeddings using ResNet
Extract 512-dimensional embeddings from clothing images
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import json
import pickle
from tqdm import tqdm

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import (
    RESNET_MODEL_NAME, 
    EMBEDDING_DIM, 
    IMAGE_SIZE, 
    MODELS_DIR, 
    EMBEDDINGS_DIR,
    SUPPORTED_FORMATS
)


class ResNetEmbeddingExtractor:
    """
    ResNet-based image embedding extractor
    - Load pretrained ResNet50
    - Remove classifier layer
    - Extract 512-d embeddings
    """
    
    def __init__(self, model_name: str = RESNET_MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = EMBEDDING_DIM
        
        # Initialize model and transforms
        self.model = None
        self.transforms = None
        self._load_model()
        self._setup_transforms()
        
        print(f"ResNet Embedding Extractor initialized")
        print(f"Device: {self.device}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def _load_model(self):
        """Load ResNet model and remove classifier"""
        print(f"Loading {self.model_name}...")
        
        # Load pretrained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the final classification layer
        # ResNet50: [conv1, bn1, relu, maxpool, layer1-4, avgpool, fc]
        # We remove 'fc' to get features before classification
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Set to evaluation mode
        self.model.eval()
        
        # Move to device
        self.model = self.model.to(self.device)
        
        print(f"Model loaded successfully - classifier removed")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transforms = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("Image transforms configured")
    
    def extract_embedding(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Extract embedding from a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            512-dimensional embedding vector
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if image_path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {image_path.suffix}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transforms(image).unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                
            # Convert to numpy and flatten
            embedding = features.cpu().numpy().flatten()
            
            # Verify embedding dimension
            if embedding.shape[0] != self.embedding_dim:
                raise ValueError(f"Expected {self.embedding_dim}-d embedding, got {embedding.shape[0]}-d")
            
            return embedding
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            raise
    
    def extract_batch_embeddings(self, image_paths: List[Union[str, Path]]) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping image paths to embeddings
        """
        embeddings = {}
        
        print(f"Extracting embeddings from {len(image_paths)} images...")
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                embedding = self.extract_embedding(image_path)
                embeddings[str(image_path)] = embedding
            except Exception as e:
                print(f"Failed to process {image_path}: {str(e)}")
                continue
        
        print(f"Successfully extracted {len(embeddings)} embeddings")
        return embeddings
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], save_path: Optional[Union[str, Path]] = None):
        """
        Save embeddings to file
        
        Args:
            embeddings: Dictionary of embeddings
            save_path: Path to save file (default: embeddings_dir/embeddings.pkl)
        """
        if save_path is None:
            save_path = EMBEDDINGS_DIR / "embeddings.pkl"
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save using pickle for numpy arrays
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"Embeddings saved to: {save_path}")
        print(f"Saved {len(embeddings)} embeddings")
    
    def load_embeddings(self, load_path: Optional[Union[str, Path]] = None) -> Dict[str, np.ndarray]:
        """
        Load embeddings from file
        
        Args:
            load_path: Path to load file (default: embeddings_dir/embeddings.pkl)
            
        Returns:
            Dictionary of embeddings
        """
        if load_path is None:
            load_path = EMBEDDINGS_DIR / "embeddings.pkl"
        
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        print(f"Loaded {len(embeddings)} embeddings from: {load_path}")
        return embeddings
    
    def save_model(self, save_path: Optional[Union[str, Path]] = None):
        """
        Save the modified ResNet model
        
        Args:
            save_path: Path to save model (default: models_dir/resnet_extractor.pth)
        """
        if save_path is None:
            save_path = MODELS_DIR / f"{self.model_name}_extractor.pth"
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'image_size': IMAGE_SIZE
        }, save_path)
        
        print(f"Model saved to: {save_path}")
    
    def load_model(self, load_path: Optional[Union[str, Path]] = None):
        """
        Load a saved model
        
        Args:
            load_path: Path to load model (default: models_dir/resnet_extractor.pth)
        """
        if load_path is None:
            load_path = MODELS_DIR / f"{self.model_name}_extractor.pth"
        
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_name = checkpoint['model_name']
        self.embedding_dim = checkpoint['embedding_dim']
        
        print(f"Model loaded from: {load_path}")
    
    def get_embedding_stats(self, embeddings: Dict[str, np.ndarray]) -> Dict:
        """
        Get statistics about embeddings
        
        Args:
            embeddings: Dictionary of embeddings
            
        Returns:
            Statistics dictionary
        """
        if not embeddings:
            return {}
        
        embedding_matrix = np.array(list(embeddings.values()))
        
        stats = {
            'num_embeddings': len(embeddings),
            'embedding_dim': embedding_matrix.shape[1],
            'mean': np.mean(embedding_matrix, axis=0),
            'std': np.std(embedding_matrix, axis=0),
            'min': np.min(embedding_matrix, axis=0),
            'max': np.max(embedding_matrix, axis=0),
            'norm_mean': np.mean(np.linalg.norm(embedding_matrix, axis=1)),
            'norm_std': np.std(np.linalg.norm(embedding_matrix, axis=1))
        }
        
        return stats


# Test function
def test_resnet_extractor():
    """Test the ResNet embedding extractor"""
    extractor = ResNetEmbeddingExtractor()
    
    # Test with sample image (if available)
    sample_dir = Path(__file__).parent.parent.parent / "data" / "sample_images"
    
    if sample_dir.exists():
        image_files = list(sample_dir.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in SUPPORTED_FORMATS]
        
        if image_files:
            print(f"Testing with {len(image_files)} sample images...")
            
            # Test single embedding
            embedding = extractor.extract_embedding(image_files[0])
            print(f"Single embedding shape: {embedding.shape}")
            print(f"Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
            
            # Test batch embeddings
            embeddings = extractor.extract_batch_embeddings(image_files[:3])
            stats = extractor.get_embedding_stats(embeddings)
            
            print(f"Batch stats:")
            print(f"  Number of embeddings: {stats['num_embeddings']}")
            print(f"  Embedding dimension: {stats['embedding_dim']}")
            print(f"  Average norm: {stats['norm_mean']:.4f}")
            
            # Save test embeddings
            extractor.save_embeddings(embeddings, EMBEDDINGS_DIR / "test_embeddings.pkl")
            
        else:
            print("No sample images found for testing")
    else:
        print("Sample images directory not found")
    
    # Save model
    extractor.save_model()
    print("Extractor test completed!")


if __name__ == "__main__":
    test_resnet_extractor()
