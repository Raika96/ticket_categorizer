"""
REST API for Support Ticket Classification
FastAPI-based production API with health checks and monitoring

Usage:
    # Development
    python3 api.py
    
    # Production (with uvicorn)
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
    
    # Test
    curl -X POST http://localhost:8000/classify \
      -H "Content-Type: application/json" \
      -d '{"title": "Cannot login", "description": "Error accessing account"}'
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import uvicorn
import json
import os
import sys
from datetime import datetime
import logging

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.infer import (
    load_baseline_model,
    load_distilbert_model,
    predict_ticket,
    predict_batch
)
from src.inference.gpt_fallback import hybrid_classify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Support Ticket Classification API",
    description="Automatically classify support tickets into categories",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
MODEL_CACHE = {}

# Request/Response Models
class TicketInput(BaseModel):
    """Single ticket input"""
    title: str = Field(..., description="Ticket title", example="Cannot login to account")
    description: str = Field(..., description="Ticket description", example="I'm getting an error message when trying to access my account")
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Cannot login to account",
                "description": "I'm getting an error message when trying to access my account"
            }
        }


class BatchTicketInput(BaseModel):
    """Batch ticket input"""
    tickets: List[TicketInput] = Field(..., description="List of tickets to classify")


class PredictionOutput(BaseModel):
    """Prediction output"""
    predicted_category: str = Field(..., description="Predicted category")
    confidence: float = Field(..., description="Confidence score (0-1)")
    needs_manual_review: bool = Field(..., description="Whether ticket needs manual review")
    model_type: str = Field(..., description="Model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    classification_method: str = Field(default="ml", description="Classification method used: ml, gpt, or gpt_unknown")
    all_probabilities: Optional[Dict[str, float]] = Field(None, description="Probabilities for all categories")
    ml_prediction: Optional[Dict] = Field(None, description="ML model prediction (if GPT fallback was used)")
    gpt_prediction: Optional[Dict] = Field(None, description="GPT prediction (if fallback was used)")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: str
    timestamp: str


class StatsResponse(BaseModel):
    """API statistics"""
    total_predictions: int
    predictions_by_category: Dict[str, int]
    average_confidence: float
    uptime_seconds: float


# Global stats
STATS = {
    'total_predictions': 0,
    'predictions_by_category': {},
    'confidences': [],
    'start_time': datetime.now()
}


def get_model(model_type='baseline'):
    """Get or load model from cache"""
    if model_type not in MODEL_CACHE:
        logger.info(f"Loading {model_type} model...")
        try:
            if model_type == 'baseline':
                MODEL_CACHE[model_type] = load_baseline_model()
            elif model_type == 'distilbert':
                MODEL_CACHE[model_type] = load_distilbert_model()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            logger.info(f"✅ {model_type} model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load {model_type} model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    
    return MODEL_CACHE[model_type]


def update_stats(result):
    """Update API statistics"""
    STATS['total_predictions'] += 1
    
    category = result['predicted_category']
    STATS['predictions_by_category'][category] = STATS['predictions_by_category'].get(category, 0) + 1
    
    STATS['confidences'].append(result['confidence'])


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Support Ticket Classification API...")
    try:
        # Preload default model (DistilBERT)
        get_model('distilbert')
        logger.info("✅ API ready!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "service": "Support Ticket Classification API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "classify": "/classify",
            "classify_batch": "/classify/batch",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint"""
    model_loaded = 'baseline' in MODEL_CACHE or 'distilbert' in MODEL_CACHE
    model_type = list(MODEL_CACHE.keys())[0] if model_loaded else "none"
    
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "model_type": model_type,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats", response_model=StatsResponse, tags=["Monitoring"])
async def get_stats():
    """Get API statistics"""
    uptime = (datetime.now() - STATS['start_time']).total_seconds()
    avg_confidence = sum(STATS['confidences']) / len(STATS['confidences']) if STATS['confidences'] else 0.0
    
    return {
        "total_predictions": STATS['total_predictions'],
        "predictions_by_category": STATS['predictions_by_category'],
        "average_confidence": avg_confidence,
        "uptime_seconds": uptime
    }


@app.post("/classify", response_model=PredictionOutput, tags=["Classification"])
async def classify_ticket(
    ticket: TicketInput,
    model_type: str = 'distilbert',
    use_gpt_fallback: bool = False,
    confidence_threshold: float = 0.6,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Classify a single support ticket
    
    - **title**: Ticket title
    - **description**: Ticket description
    - **model_type**: Model to use (baseline or distilbert)
    - **use_gpt_fallback**: Enable GPT for low-confidence predictions (~$0.0003/call)
    - **confidence_threshold**: Confidence threshold for triggering GPT (default: 0.6)
    
    **GPT Fallback Feature:**
    When enabled and ML confidence < threshold:
    - GPT attempts classification
    - GPT can also return "Unknown" if uncertain
    - Requires OPENAI_API_KEY environment variable
    - Costs ~$0.0003 per ticket when triggered
    
    Returns predicted category and confidence score
    """
    try:
        # Get model
        model_dict = get_model(model_type)
        
        # Prepare input
        ticket_data = {
            'title': ticket.title,
            'description': ticket.description
        }
        
        # Predict with ML model
        ml_result = predict_ticket(ticket_data, model_dict)
        
        # Check if GPT fallback should be used
        if use_gpt_fallback and ml_result['confidence'] < confidence_threshold:
            logger.info(f"ML confidence {ml_result['confidence']:.2f} < {confidence_threshold}, trying GPT fallback...")
            
            try:
                # Use hybrid classification
                hybrid_result = hybrid_classify(
                    title=ticket.title,
                    description=ticket.description,
                    ml_category=ml_result['predicted_category'],
                    ml_confidence=ml_result['confidence'],
                    ml_threshold=confidence_threshold,
                    use_gpt_fallback=True
                )
                
                # Build final result with GPT info
                result = {
                    'predicted_category': hybrid_result['category'],
                    'confidence': hybrid_result['confidence'],
                    'needs_manual_review': hybrid_result['needs_manual_review'],
                    'model_type': model_type,
                    'timestamp': datetime.now().isoformat(),
                    'classification_method': hybrid_result['method'],
                    'ml_prediction': hybrid_result['ml_prediction'],
                    'gpt_prediction': hybrid_result.get('gpt_prediction'),
                    'all_probabilities': ml_result.get('all_probabilities')
                }
                
                logger.info(f"GPT fallback result: {hybrid_result['category']} ({hybrid_result['confidence']:.2f}) - method: {hybrid_result['method']}")
                
            except Exception as gpt_error:
                logger.error(f"GPT fallback failed: {gpt_error}")
                # Fall back to ML prediction
                result = ml_result
                result['classification_method'] = 'ml'
                result['ml_prediction'] = None
                result['gpt_prediction'] = None
        else:
            # Use ML prediction
            result = ml_result
            result['classification_method'] = 'ml'
            result['ml_prediction'] = None
            result['gpt_prediction'] = None
        
        # Update stats in background
        background_tasks.add_task(update_stats, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch", tags=["Classification"])
async def classify_batch_tickets(
    batch: BatchTicketInput,
    model_type: str = 'distilbert',
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Classify multiple support tickets in batch
    
    - **tickets**: List of tickets with title and description
    - **model_type**: Model to use (baseline or distilbert)
    
    Returns list of predictions
    """
    try:
        # Get model
        model_dict = get_model(model_type)
        
        # Prepare input
        tickets_data = [
            {'title': t.title, 'description': t.description}
            for t in batch.tickets
        ]
        
        # Predict
        results = predict_batch(tickets_data, model_dict)
        
        # Update stats in background
        for result in results:
            if 'error' not in result:
                background_tasks.add_task(update_stats, result)
        
        return {
            'count': len(results),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories", tags=["Information"])
async def get_categories(model_type: str = 'distilbert'):
    """Get list of available categories"""
    try:
        model_dict = get_model(model_type)
        label_mapping = model_dict['config']['label_mapping']
        
        categories = [
            label_mapping['id_to_category'][str(i)]
            for i in range(label_mapping['num_classes'])
        ]
        
        return {
            'count': len(categories),
            'categories': categories
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Information"])
async def get_model_info(model_type: str = 'distilbert'):
    """Get model information"""
    try:
        model_dict = get_model(model_type)
        config = model_dict['config']
        
        return {
            'model_type': model_type,
            'confidence_threshold': config.get('confidence_threshold', 0.6),
            'num_classes': config['label_mapping']['num_classes'],
            'created_at': config.get('created_at', 'unknown'),
            'training_stats': config.get('training_stats', {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (dev only)
        log_level="info"
    )

