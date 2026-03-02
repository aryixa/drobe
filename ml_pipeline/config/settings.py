"""
Configuration settings for ML Pipeline
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DATABASE_DIR = DATA_DIR / "database"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"

# Model configurations
RESNET_MODEL_NAME = "resnet50"
EMBEDDING_DIM = 512
SIMILARITY_THRESHOLD = 0.8

# Database settings
DATABASE_PATH = DATABASE_DIR / "wardrobe.db"
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss_index.bin"

# Image processing
IMAGE_SIZE = (224, 224)
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".webp"]

# API settings
API_HOST = "127.0.0.1"
API_PORT = 8000
API_DEBUG = True

# ML Model URLs (for downloading if needed)
RESNET_WEIGHTS_URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"

# RAG System settings
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
RULE_BASE_PATH = DATA_DIR / "style_rules.json"
