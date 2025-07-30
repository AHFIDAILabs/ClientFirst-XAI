# This is the main FastAPI application file of the CarePredict AI application.
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import pandas as pd
import logging
from typing import List, Dict, Any, Optional

# Import your model and explanation logic
from app.model_utils import get_model, get_encoder, get_categories, get_important_features, preprocess_input
from app.explanation_engine import get_explanation

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("api_app")

app = FastAPI(
    title="HIV Client Satisfaction XAI API",
    description="API for predicting and explaining HIV client satisfaction with patient-centered care.",
    version="1.0.0"
)

# Mount static files (for HTML, CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# Load artifacts once when the app starts
try:
    model = get_model()
    label_encoder = get_encoder()
    categories = get_categories()
    important_features = get_important_features()

    if None in [model, label_encoder, categories, important_features]:
        raise RuntimeError("Failed to load one or more model artifacts at startup.")
    logger.info("All model artifacts loaded successfully for API.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load application artifacts: {e}")
    # You might want to exit or disable prediction routes if artifacts fail to load
    model, label_encoder, categories, important_features = None, None, None, None


# Pydantic model for request body validation
class ClientInput(BaseModel):
    Age: int = Field(..., ge=18, le=100, description="Age of the client")
    Employment_Grouped: str = Field(..., description="Employment status group")
    Education_Grouped: str = Field(..., description="Education level group")
    State: Optional[str] = Field(None, description="State of residence (optional)")
    HIV_Duration_Years: float = Field(..., ge=0.0, description="Years since HIV diagnosis")
    Care_Duration_Years: float = Field(..., ge=0.0, description="Years at this facility")
    Facility_Care_Dur_Years: float = Field(..., ge=0.0, description="Total years in HIV care across all facilities")
    Empathy_Score: float = Field(..., ge=1.0, le=5.0, description="Provider Empathy Score (1-5)")
    Listening_Score: float = Field(..., ge=1.0, le=5.0, description="Provider Listening Score (1-5)")
    Decision_Share_Score: float = Field(..., ge=1.0, le=5.0, description="Shared Decision-Making Score (1-5)")
    Exam_Explained: str = Field(..., description="Provider explained exams/procedures clearly (Likert scale)")
    Discuss_NextSteps: str = Field(..., description="Provider discussed the next steps in my care (Likert scale)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Age": 35,
                    "Employment_Grouped": "Employed",
                    "Education_Grouped": "Higher Education",
                    "State": "Lagos", # Example state, adjust as per your categories
                    "HIV_Duration_Years": 5.0,
                    "Care_Duration_Years": 2.0,
                    "Facility_Care_Dur_Years": 5.0,
                    "Empathy_Score": 4.0,
                    "Listening_Score": 4.0,
                    "Decision_Share_Score": 3.0,
                    "Exam_Explained": "Agree",
                    "Discuss_NextSteps": "Agree"
                }
            ]
        }
    }


# Root endpoint serving the Home page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    """Serves the HIV Client Satisfaction Dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

# Endpoint to get categories for dropdowns in UI
@app.get("/api/categories", response_model=Dict[str, List[str]])
async def get_model_categories():
    if categories is None:
        raise HTTPException(status_code=500, detail="Model categories not loaded.")
    return categories

# Prediction and Explanation Endpoint
@app.post("/api/predict_explain")
async def predict_explain(client_input: ClientInput):
    if model is None or important_features is None or categories is None:
        logger.error("Attempted prediction when model artifacts were not loaded.")
        raise HTTPException(status_code=500, detail="Application is not ready. Model artifacts failed to load.")

    try:
        # Convert Pydantic model to dict for preprocessing
        raw_inputs_dict = client_input.model_dump()

        # Preprocess input data
        input_df = preprocess_input(raw_inputs_dict, important_features, categories)
        
        # Identify categorical features for explanation engine
        categorical_features_for_explanation = [col for col in important_features if col in categories]

        # Get explanation
        explanation_result = get_explanation(model, input_df, categorical_features_for_explanation)
        
        return explanation_result

    except ValueError as ve:
        logger.error(f"Input validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Error during prediction and explanation: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {e}")