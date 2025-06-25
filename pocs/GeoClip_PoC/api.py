from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
import tempfile
import os
from PIL import Image
from geoclip import GeoCLIP
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from typing import List, Dict, Optional
import uvicorn
from datetime import datetime
import logging
from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    description="API for identifying locations from images using GeoCLIP model",
    version=settings.API_VERSION,
    docs_url=None,  # Disable default docs
    redoc_url=None  # Disable default redoc
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Initialize geocoder
geocoder = Nominatim(
    user_agent=settings.GEOCODER_USER_AGENT,
    timeout=settings.GEOCODER_TIMEOUT
)

# Initialize GeoCLIP model
try:
    model = GeoCLIP()  # Remove use_fast parameter
    logger.info("GeoCLIP model initialized successfully")
except Exception as e:
    logger.error(f"Error loading GeoCLIP model: {str(e)}")
    model = None

def get_location_name(lat: float, lon: float) -> str:
    try:
        location = geocoder.reverse(f"{lat}, {lon}", language="en")
        if location:
            return location.address
        return "Unknown location"
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logger.warning(f"Geocoding failed for coordinates {lat}, {lon}: {str(e)}")
        return "Location lookup failed"

def format_confidence(prob: float) -> float:
    return float(prob * 10000)

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API status"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None
    }

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI endpoint"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

@app.post("/predict", 
    response_model=List[Dict],
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid input"},
        500: {"description": "Internal server error"}
    }
)
async def predict_location(
    file: UploadFile = File(..., description="Image file to analyze"),
    num_predictions: Optional[int] = settings.DEFAULT_NUM_PREDICTIONS,
    min_confidence: Optional[float] = settings.MIN_CONFIDENCE_THRESHOLD
) -> List[Dict]:
    """
    Predict location from an image.
    
    Args:
        file: The image file to analyze
        num_predictions: Number of predictions to return (default: 3)
        min_confidence: Minimum confidence score to include in results (optional)
    
    Returns:
        List of predictions with location details and confidence scores
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized"
        )
    
    # Validate file
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Validate file type
    if not file.content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File type not specified. Please ensure you're sending an image file."
        )
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Only image files (jpg, jpeg, png) are supported."
        )
    
    # Validate file extension
    file_extension = os.path.splitext(file.filename)[1].lower() if file.filename else ''
    if file_extension not in ['.jpg', '.jpeg', '.png']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension: {file_extension}. Only .jpg, .jpeg, and .png files are supported."
        )
    
    try:
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            if not content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Empty file received"
                )
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Verify the file is a valid image
        try:
            with Image.open(tmp_file_path) as img:
                img.verify()  # Verify it's an image
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {str(e)}"
            )

        # Get predictions
        top_pred_gps, top_pred_prob = model.predict(tmp_file_path, top_k=num_predictions)
        
        # Process results
        predictions = []
        for i in range(num_predictions):
            lat, lon = top_pred_gps[i]
            location_name = get_location_name(lat, lon)
            confidence_score = format_confidence(top_pred_prob[i])
            
            # Skip predictions below minimum confidence if specified
            if min_confidence is not None and confidence_score < min_confidence:
                continue
                
            predictions.append({
                "prediction_number": i + 1,
                "location_name": location_name,
                "latitude": float(lat),
                "longitude": float(lon),
                "confidence_score": confidence_score
            })
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
        
        if not predictions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No predictions met the minimum confidence threshold"
            )
        
        return predictions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        # Clean up the temporary file in case of error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=settings.API_HOST, 
        port=settings.API_PORT
    ) 