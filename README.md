# The Drobe

Drobe is a AI powered wardrobe management application where you can upload your clothing, get context-aware outfit recommendations.
## What it does

**Wardrobe management** — upload garment photos, browse and filter your collection
**Smart tagging** — automatically labels each item by colour, type, pattern, and season
**AI Stylist** — input your weather, event, and time of day; get a recommended outfit with a natural-language explanation


## How it works
User uploads image
       ↓
Preprocess (resize 224×224, remove background)
       ↓
ResNet-50 → 512-dim embedding
       ↓
Linear classifier → smart tags (colour, type, pattern, season)
       ↓
FAISS vector index (per user)
       ↓
On recommendation request:
  Query vector (context) → FAISS ANN search → top-k candidates → LLM re-rank → outfit + explanation
