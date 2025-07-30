# Model Utility Functions, handling model loading and inference
import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = "model/top10_model.joblib"
ENCODER_PATH = "model/label_encoder.joblib"
CATEGORIES_PATH = "model/categories.joblib"
IMPORTANT_FEATURES_PATH = "model/important_features.joblib"


# Load artifacts once when the module is imported
try:
    _model = joblib.load(MODEL_PATH)
    _encoder = joblib.load(ENCODER_PATH)
    _categories = joblib.load(CATEGORIES_PATH)
    _important_features = joblib.load(IMPORTANT_FEATURES_PATH)
except FileNotFoundError as e:
    print(f"Error loading model artifacts: {e}. Ensure all .joblib files are in the 'model/' directory.")
    _model, _encoder, _categories, _important_features = None, None, None, None
except Exception as e:
    print(f"An unexpected error occurred while loading artifacts: {e}")
    _model, _encoder, _categories, _important_features = None, None, None, None


def get_model():
    return _model

def get_encoder():
    return _encoder

def get_categories():
    return _categories

def get_important_features():
    return _important_features

def predict_proba(data: pd.DataFrame):
    """
    Makes probability predictions using the loaded model.
    Assumes data is already preprocessed and contains the necessary features.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Cannot make predictions.")
    
    # CatBoost handles categorical features directly if specified during training.
    # Ensure the input DataFrame columns match the model's expected features.
    
    return _model.predict_proba(data)

def predict_class(data: pd.DataFrame):
    """
    Makes class predictions using the loaded model.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Cannot make predictions.")
    return _model.predict(data)

def preprocess_input(raw_inputs: dict, important_features: list, categories: dict):
    """
    Preprocesses raw input data into a DataFrame suitable for the model.
    This function replaces the preprocessing logic previously in Home.py.
    """
    likert_map = {
        'Strongly Disagree': 1, 'Disagree': 2, 'Neither Agree Or Disagree': 3,
        'Agree': 4, 'Strongly Agree': 5
    }

    final_features = {}
    
    # Calculate interaction and ratio features
    hiv_duration = raw_inputs.get('HIV_Duration_Years', 0.0)
    care_duration = raw_inputs.get('Care_Duration_Years', 0.0)
    final_features['HIV_Care_Duration_Ratio'] = hiv_duration / (care_duration + 0.1) if care_duration >= 0 else 0.0 # Avoid division by zero
    
    empathy_score = raw_inputs.get('Empathy_Score', 0.0)
    listening_score = raw_inputs.get('Listening_Score', 0.0)
    decision_share_score = raw_inputs.get('Decision_Share_Score', 0.0)
    
    final_features['Empathy_Listening_Interaction'] = empathy_score * listening_score
    final_features['Empathy_DecisionShare_Interaction'] = empathy_score * decision_share_score
    
    # Map Likert scale inputs
    final_features['Exam_Explained'] = likert_map.get(raw_inputs.get('Exam_Explained', 'Neither Agree Or Disagree'), 3)
    final_features['Discuss_NextSteps'] = likert_map.get(raw_inputs.get('Discuss_NextSteps', 'Neither Agree Or Disagree'), 3)

    # Populate the input_data dictionary with all required features for the model
    input_data = {}
    for feature in important_features:
        if feature in final_features:
            input_data[feature] = final_features[feature]
        elif feature in raw_inputs:
            input_data[feature] = raw_inputs[feature]
        else:
            # Handle missing features: assign a default value or the first category
            if feature in categories and categories[feature]:
                input_data[feature] = categories[feature][0] # Default to first category
            else:
                input_data[feature] = 0.0 # Default for numerical features

    input_df = pd.DataFrame([input_data], columns=important_features)
    
    # Ensure categorical columns have 'category' dtype if your model expects it for SHAP or other reasons
    categorical_features_in_model = [col for col in important_features if col in categories]
    for col in categorical_features_in_model:
        if col in input_df.columns:
            # Ensure categories match the training data's categories
            input_df[col] = pd.Categorical(input_df[col], categories=categories.get(col, []))

    return input_df