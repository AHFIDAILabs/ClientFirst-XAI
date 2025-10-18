from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

# Import pipelines
from src.ClientFirst.pipeline.predict import PredictionPipeline
from src.ClientFirst.pipeline.explanation_engine import get_explanation
from src.ClientFirst.utils.logger import logger

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Gamsuwa - HIV Client Satisfaction XAI API",
    description="AI-powered platform for predicting and explaining HIV client satisfaction",
    version="2.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Initialize prediction pipeline
try:
    prediction_pipeline = PredictionPipeline()
    logger.info("Prediction pipeline loaded successfully")
except Exception as e:
    logger.error(f"Failed to load prediction pipeline: {e}")
    prediction_pipeline = None


# Pydantic models
class ClientInput(BaseModel):
    Age: int = Field(..., ge=18, le=100, description="Age of the client")
    Employment_Grouped: str = Field(..., description="Employment status group")
    Education_Grouped: str = Field(..., description="Education level group")
    State: str = Field(..., description="State of residence")  # NOW REQUIRED
    HIV_Duration_Years: float = Field(..., ge=0.0, description="Years since HIV diagnosis")
    Care_Duration_Years: float = Field(..., ge=0.0, description="Years at this facility")
    Facility_Care_Dur_Years: float = Field(..., ge=0.0, description="Total years in HIV care")
    Empathy_Score: float = Field(..., ge=1.0, le=5.0, description="Provider Empathy Score (1-5)")
    Listening_Score: float = Field(..., ge=1.0, le=5.0, description="Provider Listening Score (1-5)")
    Decision_Share_Score: float = Field(..., ge=1.0, le=5.0, description="Shared Decision-Making Score (1-5)")
    Exam_Explained: str = Field(..., description="Provider explained exams clearly (Likert scale)")
    Discuss_NextSteps: str = Field(..., description="Provider discussed next steps (Likert scale)")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Age": 35,
                "Employment_Grouped": "Employed",
                "Education_Grouped": "Higher Education",
                "State": "Lagos",  
                "HIV_Duration_Years": 5.0,
                "Care_Duration_Years": 5.0,
                "Facility_Care_Dur_Years": 3.0,
                "Empathy_Score": 4,
                "Listening_Score": 4,
                "Decision_Share_Score": 3,
                "Exam_Explained": "Agree",
                "Discuss_NextSteps": "Agree"
            }]
        }
    }

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve home page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    """Serve dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def read_contact(request: Request):
    """Serve contact page"""
    return templates.TemplateResponse("contact.html", {"request": request})

@app.get("/faq", response_class=HTMLResponse)
async def read_faq(request: Request):
    """Serve FAQ page"""
    return templates.TemplateResponse("faq.html", {"request": request})

@app.post("/send-message")
async def send_message(
    name: str = Form(...),
    email: str = Form(...),
    message: str = Form(...)
):
    """Handle contact form submissions"""
    recipient_email = "aiteam@ahfid.org"
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    
    if not all([name, email, message, sender_email, sender_password]):
        raise HTTPException(
            status_code=400,
            detail="Missing required information or server configuration."
        )

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"New Contact Form Submission from {name}"
        
        body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        
        logger.info(f"Successfully sent email from {name} ({email})")
        return RedirectResponse(url="/contact?success=true", status_code=303)
        
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP authentication failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to send message: Authentication error."
        )
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to send message: Server error."
        )


@app.get("/api/categories", response_model=Dict[str, List[str]])
async def get_categories():
    """Get categories for UI dropdowns"""
    if prediction_pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Prediction pipeline not loaded."
        )
    
    return prediction_pipeline.get_categories()


@app.post("/api/predict_explain")
async def predict_explain(client_input: ClientInput):
    """Predict and explain client satisfaction"""
    if prediction_pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Application not ready. Prediction pipeline failed to load."
        )

    try:
        # Convert input to dict
        raw_inputs_dict = client_input.model_dump()
        
        # Preprocess input
        input_df = prediction_pipeline.preprocess_input(raw_inputs_dict)
        
        # Get explanation
        explanation_result = get_explanation(
            prediction_pipeline.model,
            input_df,
            prediction_pipeline.categorical_features
        )
        
        return explanation_result

    except ValueError as ve:
        logger.error(f"Input validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during prediction and explanation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_loaded": prediction_pipeline is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)