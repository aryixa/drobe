"""
Main API server for ML Pipeline
Connects the wardrobe recommendation system to the React frontend
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import uvicorn
import asyncio
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import ML pipeline components
from src.context_aware.integration_engine import IntegrationEngine
from src.context_aware.context_parser import ContextParser
from src.embeddings.embedding_storage import EmbeddingStorage
from src.smart_tags.color_extractor import ColorExtractor
from src.smart_tags.type_classifier import TypeClassifier
from src.smart_tags.pattern_detector import PatternDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Drobe ML Pipeline API",
    description="Intelligent wardrobe recommendation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for ML components
integration_engine: Optional[IntegrationEngine] = None
color_extractor: Optional[ColorExtractor] = None
type_classifier: Optional[TypeClassifier] = None
pattern_detector: Optional[PatternDetector] = None

# Pydantic models for API
class RecommendationRequest(BaseModel):
    query: str = Field(..., description="Natural language query for outfit recommendation")
    max_outfits: int = Field(default=5, ge=1, le=20, description="Maximum number of recommendations")
    user_preferences: Optional[Dict[str, float]] = Field(default=None, description="User preference weights")

class RecommendationResponse(BaseModel):
    success: bool
    query: str
    recommendation: Optional[Dict]
    alternatives: List[Dict]
    explanation: Optional[Dict]
    processing_time: float
    metadata: Dict

class ImageAnalysisRequest(BaseModel):
    image_path: str = Field(..., description="Path to image file")
    analysis_type: str = Field(default="all", description="Type of analysis: colors, type, pattern, or all")

class ImageAnalysisResponse(BaseModel):
    success: bool
    image_path: str
    colors: Optional[List[Dict]]
    type_classification: Optional[Dict]
    pattern_detection: Optional[Dict]
    processing_time: float
    errors: List[str]

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]
    capabilities: List[str]

class BatchRecommendationRequest(BaseModel):
    queries: List[str] = Field(..., description="List of queries to process")
    user_preferences: Optional[Dict[str, float]] = Field(default=None)

class BatchRecommendationResponse(BaseModel):
    success: bool
    results: List[Dict]
    total_processed: int
    total_time: float
    errors: List[str]

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize ML components on startup"""
    global integration_engine, color_extractor, type_classifier, pattern_detector
    
    try:
        logger.info("Initializing ML Pipeline components...")
        
        # Initialize core components
        integration_engine = IntegrationEngine()
        color_extractor = ColorExtractor()
        type_classifier = TypeClassifier()
        pattern_detector = PatternDetector()
        
        logger.info("ML Pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize ML Pipeline: {e}")
        # Continue without ML components for basic functionality

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    components = {
        "integration_engine": "ready" if integration_engine else "not_initialized",
        "color_extractor": "ready" if color_extractor else "not_initialized", 
        "type_classifier": "ready" if type_classifier else "not_initialized",
        "pattern_detector": "ready" if pattern_detector else "not_initialized"
    }
    
    capabilities = []
    if integration_engine:
        capabilities.extend([
            "context_parsing",
            "outfit_recommendation", 
            "explainability",
            "similarity_search",
            "rule_application"
        ])
    if color_extractor:
        capabilities.append("color_extraction")
    if type_classifier:
        capabilities.append("type_classification")
    if pattern_detector:
        capabilities.append("pattern_detection")
    
    return HealthResponse(
        status="healthy" if any("ready" in v for v in components.values()) else "degraded",
        components=components,
        capabilities=capabilities
    )

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendation(request: RecommendationRequest):
    """Get outfit recommendation based on natural language query"""
    if not integration_engine:
        raise HTTPException(status_code=503, detail="ML Pipeline not initialized")
    
    try:
        logger.info(f"Processing recommendation request: {request.query}")
        
        # Process the query
        result = integration_engine.process_query(
            query=request.query,
            max_outfits=request.max_outfits
        )
        
        # Format response
        recommendation_data = None
        if result.outfit:
            recommendation_data = {
                "items": [item.path for item in result.outfit.items],
                "score": result.score,
                "style_score": getattr(result.outfit, 'style_score', 0.0),
                "color_score": getattr(result.outfit, 'color_score', 0.0),
                "pattern_score": getattr(result.outfit, 'pattern_score', 0.0),
                "formality_score": getattr(result.outfit, 'formality_score', 0.0)
            }
        
        alternatives_data = []
        for alt in result.alternatives:
            if 'recommendation' in alt and 'explanation' in alt:
                alt_data = {
                    "items": alt['recommendation']['items'],
                    "score": alt['recommendation']['score'],
                    "explanation": alt['explanation']
                }
                alternatives_data.append(alt_data)
        
        return RecommendationResponse(
            success=True,
            query=result.query,
            recommendation=recommendation_data,
            alternatives=alternatives_data,
            explanation=result.explanation,
            processing_time=result.processing_time,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Error processing recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(request: ImageAnalysisRequest):
    """Analyze clothing image for colors, type, and patterns"""
    errors = []
    colors_data = None
    type_data = None
    pattern_data = None
    processing_time = 0.0
    
    try:
        import time
        start_time = time.time()
        
        # Color analysis
        if request.analysis_type in ["colors", "all"] and color_extractor:
            try:
                colors = color_extractor.extract_colors(request.image_path, top_k=5)
                colors_data = [
                    {
                        "category": color.category.value,
                        "rgb": color.rgb,
                        "percentage": color.percentage,
                        "confidence": color.confidence
                    }
                    for color in colors
                ]
            except Exception as e:
                errors.append(f"Color analysis failed: {str(e)}")
        
        # Type classification
        if request.analysis_type in ["type", "all"] and type_classifier:
            try:
                classification = type_classifier.classify_from_filename(request.image_path)
                type_data = {
                    "primary_type": classification.primary_type.value,
                    "sub_type": classification.sub_type.value if classification.sub_type else None,
                    "confidence": classification.confidence,
                    "alternatives": [
                        {"type": alt[0].value, "confidence": alt[1]} 
                        for alt in classification.alternative_types
                    ]
                }
            except Exception as e:
                errors.append(f"Type classification failed: {str(e)}")
        
        # Pattern detection
        if request.analysis_type in ["pattern", "all"] and pattern_detector:
            try:
                pattern_info = pattern_detector.detect_pattern(request.image_path)
                pattern_data = {
                    "pattern_type": pattern_info.pattern_type.value,
                    "confidence": pattern_info.confidence,
                    "characteristics": pattern_info.characteristics,
                    "evidence": pattern_info.evidence
                }
            except Exception as e:
                errors.append(f"Pattern detection failed: {str(e)}")
        
        processing_time = time.time() - start_time
        
        return ImageAnalysisResponse(
            success=len(errors) == 0,
            image_path=request.image_path,
            colors=colors_data,
            type_classification=type_data,
            pattern_detection=pattern_data,
            processing_time=processing_time,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-recommend", response_model=BatchRecommendationResponse)
async def batch_recommend(request: BatchRecommendationRequest, background_tasks: BackgroundTasks):
    """Process multiple recommendation queries"""
    if not integration_engine:
        raise HTTPException(status_code=503, detail="ML Pipeline not initialized")
    
    try:
        logger.info(f"Processing batch recommendation with {len(request.queries)} queries")
        
        import time
        start_time = time.time()
        
        # Process queries
        results = []
        errors = []
        
        for i, query in enumerate(request.queries):
            try:
                result = integration_engine.process_query(query, max_outfits=3)
                
                result_data = {
                    "query": query,
                    "success": True,
                    "recommendation": {
                        "items": [item.path for item in result.outfit.items] if result.outfit else [],
                        "score": result.score
                    } if result.outfit else None,
                    "explanation": result.explanation,
                    "processing_time": result.processing_time
                }
                results.append(result_data)
                
            except Exception as e:
                error_data = {
                    "query": query,
                    "success": False,
                    "error": str(e)
                }
                results.append(error_data)
                errors.append(f"Query '{query}' failed: {str(e)}")
        
        total_time = time.time() - start_time
        
        return BatchRecommendationResponse(
            success=len(errors) == 0,
            results=results,
            total_processed=len(request.queries),
            total_time=total_time,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Error processing batch recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/context-parse")
async def parse_context(query: str):
    """Parse natural language query into structured context"""
    if not integration_engine:
        raise HTTPException(status_code=503, detail="ML Pipeline not initialized")
    
    try:
        context = integration_engine.context_parser.parse_context(query)
        enhanced = integration_engine.context_parser.enhance_context_with_defaults(context)
        
        return {
            "original_query": enhanced.original_query,
            "occasion": enhanced.occasion.value if enhanced.occasion else None,
            "season": enhanced.season.value if enhanced.season else None,
            "weather": enhanced.weather.value if enhanced.weather else None,
            "time_of_day": enhanced.time_of_day.value if enhanced.time_of_day else None,
            "style_level": enhanced.style_level.value if enhanced.style_level else None,
            "colors": enhanced.colors,
            "patterns": enhanced.patterns,
            "clothing_types": enhanced.clothing_types,
            "keywords": enhanced.keywords,
            "confidence": enhanced.confidence,
            "parsing_errors": enhanced.parsing_errors
        }
        
    except Exception as e:
        logger.error(f"Error parsing context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_stats():
    """Get system statistics and performance metrics"""
    if not integration_engine:
        raise HTTPException(status_code=503, detail="ML Pipeline not initialized")
    
    try:
        stats = integration_engine.get_engine_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )

# Main function to run the server
def main():
    """Run the API server"""
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
