"""
Main entry point for Wardrobe Vault ML Pipeline
Phase 2 - ML Pipeline Implementation (Tasks 9-14)
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config.settings import BASE_DIR
from src.embeddings.resnet_extractor import ResNetEmbeddingExtractor
from src.similarity.cosine_sim import CosineSimilarityEngine
from src.outfit_builder.core_engine import OutfitBuilder
from src.smart_tags.color_extractor import ColorExtractor
from src.rag_system.rule_base import RuleBase
from src.context_aware.context_processor import ContextProcessor


def main():
    """Main function to run ML pipeline"""
    parser = argparse.ArgumentParser(description="Wardrobe Vault ML Pipeline")
    parser.add_argument("--task", type=int, choices=[9, 10, 11, 12, 13, 14], 
                       help="Run specific task (9-14)")
    parser.add_argument("--mode", choices=["extract", "train", "predict", "serve"], 
                       default="serve", help="Operation mode")
    parser.add_argument("--image-path", type=str, help="Path to image for processing")
    
    args = parser.parse_args()
    
    print("Wardrobe Vault ML Pipeline")
    print("=" * 40)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Task: {args.task if args.task else 'All'}")
    print(f"Mode: {args.mode}")
    print("=" * 40)
    
    if args.task == 9 or not args.task:
        print("\nTask 9: Image Embeddings (ResNet)")
        extractor = ResNetEmbeddingExtractor()
        print("ResNet embedding extractor initialized")
        
    if args.task == 10 or not args.task:
        print("\nTask 10: Similarity System")
        similarity_engine = CosineSimilarityEngine()
        print("Cosine similarity engine initialized")
        
    if args.task == 11 or not args.task:
        print("\nTask 11: Outfit Builder (CORE FEATURE)")
        outfit_builder = OutfitBuilder()
        print("Outfit builder initialized")
        
    if args.task == 12 or not args.task:
        print("\nTask 12: Smart Tags System")
        color_extractor = ColorExtractor()
        print("Smart tags system initialized")
        
    if args.task == 13 or not args.task:
        print("\nTask 13: RAG + Explainability")
        rule_base = RuleBase()
        print("RAG system initialized")
        
    if args.task == 14 or not args.task:
        print("\nTask 14: Context-Aware Styling + Integration")
        context_processor = ContextProcessor()
        print("Context-aware system initialized")
    
    print("\nML Pipeline initialized successfully!")
    print("Ready to process wardrobe recommendations...")


if __name__ == "__main__":
    main()
