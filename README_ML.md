# Wardrobe Vault ML Pipeline

## Phase 2 - ML Pipeline Implementation (Tasks 9-14)

### Project Structure

```
ml_pipeline/
|-- config/                 # Configuration settings
|-- data/                   # Data storage
|   |-- models/            # Saved ML models
|   |-- embeddings/        # Image embeddings storage
|   |-- database/          # Database files
|   `-- sample_images/     # Sample clothing images
|-- src/                   # Source code modules
|   |-- embeddings/        # Task 9: ResNet embeddings
|   |-- similarity/        # Task 10: Cosine similarity
|   |-- outfit_builder/    # Task 11: CORE FEATURE
|   |-- smart_tags/        # Task 12: Color/type/pattern extraction
|   |-- rag_system/        # Task 13: RAG + Explainability
|   |-- context_aware/     # Task 14: Context-aware styling
|   `-- utils/             # Utility functions
|-- tests/                 # Test suite
|-- api/                   # FastAPI endpoints
|-- main.py               # Main entry point
`-- requirements.txt      # Python dependencies
```

### Installation

```bash
# Create virtual environment
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run all tasks
python ml_pipeline/main.py

# Run specific task
python ml_pipeline/main.py --task 9

# Different modes
python ml_pipeline/main.py --mode extract
python ml_pipeline/main.py --mode serve
```

### Tasks Overview

- **Task 9**: Image Embeddings using ResNet (512-d features)
- **Task 10**: Similarity System with cosine similarity
- **Task 11**: Outfit Builder (CORE FEATURE) - combines tops, bottoms, shoes
- **Task 12**: Smart Tags System - color, type, pattern extraction
- **Task 13**: RAG + Explainability - rule-based reasoning
- **Task 14**: Context-Aware Styling - weather, event, time integration

### API Endpoints

The system provides FastAPI endpoints for integration with the React frontend:

- `/embeddings/extract` - Extract image embeddings
- `/similarity/find` - Find similar items
- `/outfit/build` - Generate outfit combinations
- `/tags/extract` - Extract smart tags
- `/recommendations` - Get contextual recommendations

### Dependencies

Key libraries:
- PyTorch & torchvision (ResNet)
- OpenCV (image processing)
- FAISS (vector search)
- SentenceTransformers (text embeddings)
- FastAPI (API layer)
- scikit-learn (similarity calculations)
