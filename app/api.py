from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create FastAPI REST server
app = FastAPI(title="Salary Prediction API", version="1.0.0")

# Load model (would be loaded from config in production)
try:
    model = joblib.load("models/salary_predictor_linear_regression.pkl")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

class PredictionRequest(BaseModel):
    years_experience: float
    
class BatchPredictionRequest(BaseModel):
    data: List[float]

class PredictionResponse(BaseModel):
    predicted_salary: float
    model_version: str = "1.0.0"

@app.get("/")
def root():
    return {"message": "Salary Prediction API", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Make prediction
        prediction = model.predict([[request.years_experience]])[0]        
        logger.info(f"Prediction made for {request.years_experience} years: ${prediction:.2f}")
        
        return PredictionResponse(
            predicted_salary=round(float(prediction), 2),
            model_version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict_batch")
def predict_batch(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = model.predict(np.array(request.data).reshape(-1, 1))        
        return {
            "predictions": [float(p) for p in predictions],
            "count": len(predictions)
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)