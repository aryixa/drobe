# ML Pipeline API Documentation

## Overview

The ML Pipeline API provides intelligent wardrobe recommendations through a RESTful interface. It combines computer vision, natural language processing, and fashion rule systems to provide personalized outfit suggestions.

## Quick Start

### 1. Start the API Server

```bash
cd ml_pipeline
python -m src.api.main
```

The API will be available at `http://127.0.0.1:8000`

### 2. Frontend Integration

The React frontend connects to the API through the `mlApi` service:

```typescript
import { mlApi } from '../services/mlApi';

// Get recommendation
const result = await mlApi.getRecommendation({
  query: "summer casual day outfit",
  max_outfits: 5
});
```

## API Endpoints

### Health Check
```http
GET /health
```

Returns API health status and available capabilities.

### Recommendation
```http
POST /recommend
```

Get outfit recommendations based on natural language query.

**Request Body:**
```json
{
  "query": "summer casual day outfit",
  "max_outfits": 5,
  "user_preferences": {
    "style_compatibility": 0.8,
    "color_harmony": 0.7
  }
}
```

**Response:**
```json
{
  "success": true,
  "query": "summer casual day outfit",
  "recommendation": {
    "items": ["blue_shirt.jpg", "khaki_pants.jpg", "brown_shoes.jpg"],
    "score": 0.85,
    "style_score": 0.82,
    "color_score": 0.88,
    "pattern_score": 0.90,
    "formality_score": 0.75
  },
  "alternatives": [...],
  "explanation": {
    "primary_reason": "This outfit is perfect for summer casual wear...",
    "supporting_rules": [...],
    "confidence": 0.87
  },
  "processing_time": 1.23,
  "metadata": {...}
}
```

### Image Analysis
```http
POST /analyze-image
```

Analyze clothing images for colors, type, and patterns.

**Request Body:**
```json
{
  "image_path": "/path/to/image.jpg",
  "analysis_type": "all"
}
```

**Response:**
```json
{
  "success": true,
  "image_path": "/path/to/image.jpg",
  "colors": [
    {
      "category": "blue",
      "rgb": [52, 152, 219],
      "percentage": 0.45,
      "confidence": 0.92
    }
  ],
  "type_classification": {
    "primary_type": "top",
    "sub_type": "t-shirt",
    "confidence": 0.88
  },
  "pattern_detection": {
    "pattern_type": "solid",
    "confidence": 0.95
  },
  "processing_time": 0.45,
  "errors": []
}
```

### Context Parsing
```http
GET /context-parse?query=your_query
```

Parse natural language into structured context.

**Response:**
```json
{
  "original_query": "formal business meeting for winter",
  "occasion": "business",
  "season": "winter",
  "weather": null,
  "time_of_day": null,
  "style_level": "business",
  "colors": [],
  "patterns": [],
  "clothing_types": [],
  "keywords": ["formal", "business", "winter"],
  "confidence": 0.85,
  "parsing_errors": []
}
```

### Batch Recommendations
```http
POST /batch-recommend
```

Process multiple queries simultaneously.

**Request Body:**
```json
{
  "queries": [
    "casual summer outfit",
    "formal business attire",
    "date night dress"
  ],
  "user_preferences": {}
}
```

### System Stats
```http
GET /stats
```

Get system statistics and performance metrics.

## React Integration

### Using the ML API Service

```typescript
import { useRecommendation } from '../hooks/useMLApi';

function RecommendationComponent() {
  const { recommendation, loading, error, getRecommendation } = useRecommendation();

  const handleRequest = () => {
    getRecommendation("summer casual day outfit");
  };

  return (
    <div>
      <button onClick={handleRequest} disabled={loading}>
        Get Recommendation
      </button>
      
      {loading && <p>Processing...</p>}
      {error && <p>Error: {error}</p>}
      {recommendation && (
        <div>
          <h3>Score: {recommendation.score}</h3>
          <p>Items: {recommendation.items.join(', ')}</p>
        </div>
      )}
    </div>
  );
}
```

### Using the Demo Component

```typescript
import { MLRecommendationDemo } from '../components/MLRecommendationDemo';

function App() {
  return (
    <div>
      <h1>Drobe Wardrobe System</h1>
      <MLRecommendationDemo />
    </div>
  );
}
```

## Query Examples

### Natural Language Queries
- "summer casual day outfit"
- "formal business meeting attire"
- "date night elegant dress"
- "winter cold weather clothes"
- "beach vacation sunny outfit"
- "gym workout athletic wear"

### Context Elements
The system can extract:
- **Occasion**: business, casual, formal, party, date, sports, outdoor, beach, travel
- **Season**: spring, summer, fall, winter
- **Weather**: sunny, rainy, cold, hot, mild, humid
- **Time**: morning, afternoon, evening, night
- **Style**: very casual, casual, smart casual, business, formal
- **Colors**: blue, red, green, black, white, gray, etc.
- **Patterns**: solid, striped, floral, geometric, etc.

## Error Handling

The API provides comprehensive error handling:

```typescript
const result = await mlApi.getRecommendationSafe(query);

if (!result.success) {
  console.error('Recommendation failed:', result.error);
  // Handle error gracefully
}
```

## Performance

- **Recommendation processing**: ~1-2 seconds
- **Image analysis**: ~0.5-1 second
- **Context parsing**: ~0.1 second
- **Batch processing**: Scales linearly with query count

## Troubleshooting

### Common Issues

1. **API Not Responding**
   - Check if the Python server is running
   - Verify the URL in the frontend matches the server

2. **No Recommendations Found**
   - Check if the wardrobe has clothing items
   - Verify the ML pipeline is properly initialized

3. **Image Analysis Fails**
   - Ensure image path is correct
   - Check if image format is supported (jpg, png, webp)

4. **Context Parsing Issues**
   - Try more specific queries
   - Check for parsing errors in response

### Debug Mode

Enable debug logging in the API server:

```bash
python -m src.api.main --log-level debug
```

## Architecture

### ML Pipeline Components
1. **Context Parser**: Natural language to structured context
2. **Integration Engine**: Combines all ML components
3. **Outfit Builder**: Creates and ranks outfit combinations
4. **Smart Tags**: Analyzes images for colors, types, patterns
5. **RAG System**: Provides explanations based on fashion rules
6. **Similarity Engine**: Finds similar items using embeddings

### Data Flow
```
User Query -> Context Parser -> Integration Engine -> Outfit Builder -> RAG System -> Response
```

## Development

### Adding New Features

1. **New Query Types**: Extend the context parser
2. **New Analysis Types**: Add to the smart tags system
3. **New Rules**: Update the rule base
4. **New Endpoints**: Add to the FastAPI application

### Testing

Run the test suite:

```bash
cd ml_pipeline
python -m pytest tests/ -v
```

## Deployment

### Production Setup

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Server Configuration**:
   ```python
   # In main.py
   uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4)
   ```

3. **Frontend Configuration**:
   ```typescript
   const mlApi = new MLApiService('https://your-api-domain.com');
   ```

### Monitoring

- Health checks at `/health`
- Performance metrics at `/stats`
- Error logging configured for production

## Support

For issues and questions:
1. Check the API health status
2. Review the error messages
3. Consult the troubleshooting section
4. Check the test cases for usage examples
