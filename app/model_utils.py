# Model Utility Functions, handling model loading and inference
import joblib
import pandas as pd
import numpy as np
import os

# Define paths to the model artifacts
MODEL_PATH = "model/top10_model.joblib"
ENCODER_PATH = "model/label_encoder.joblib"
CATEGORIES_PATH = "model/categories.joblib"
IMPORTANT_FEATURES_PATH = "model/important_features.joblib"


# Load artifacts once when the module is imported to optimize performance
try:
    _model = joblib.load(MODEL_PATH)
    _encoder = joblib.load(ENCODER_PATH)
    _categories = joblib.load(CATEGORIES_PATH)
    _important_features = joblib.load(IMPORTANT_FEATURES_PATH)
    print("Model artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model artifacts: {e}. Ensure all .joblib files are in the 'model/' directory.")
    _model, _encoder, _categories, _important_features = None, None, None, None
except Exception as e:
    print(f"An unexpected error occurred while loading artifacts: {e}")
    _model, _encoder, _categories, _important_features = None, None, None, None


def get_model():
    """Returns the loaded CatBoost model."""
    return _model

def get_encoder():
    """Returns the loaded LabelEncoder."""
    return _encoder

def get_categories():
    """Returns the dictionary of feature categories."""
    return _categories

def get_important_features():
    """Returns the list of important feature names."""
    return _important_features

def predict_proba(data: pd.DataFrame):
    """
    Makes probability predictions using the loaded model.
    Assumes data is already preprocessed.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Cannot make predictions.")
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
    Preprocesses raw input from the API into a DataFrame suitable for the model.
    This function handles feature engineering, mapping, and missing value imputation.
    """
    # Define the mapping for Likert scale string inputs to numerical values
    likert_map = {
        'Strongly Disagree': 1, 'Disagree': 2, 'Neither Agree Or Disagree': 3,
        'Agree': 4, 'Strongly Agree': 5
    }

    # --- 1. Feature Engineering ---
    # Create new interaction and ratio features from the raw inputs
    final_features = {}
    
    # Calculate HIV Care Duration Ratio
    hiv_duration = raw_inputs.get('HIV_Duration_Years', 0.0)
    care_duration = raw_inputs.get('Care_Duration_Years', 0.0)
    # Add a small epsilon to the denominator to prevent division by zero
    final_features['HIV_Care_Duration_Ratio'] = hiv_duration / (care_duration + 0.1) if care_duration >= 0 else 0.0
    
    # Calculate interaction scores for provider communication
    empathy_score = raw_inputs.get('Empathy_Score', 0.0)
    listening_score = raw_inputs.get('Listening_Score', 0.0)
    decision_share_score = raw_inputs.get('Decision_Share_Score', 0.0)
    
    final_features['Empathy_Listening_Interaction'] = empathy_score * listening_score
    final_features['Empathy_DecisionShare_Interaction'] = empathy_score * decision_share_score
    
    # Map Likert scale string inputs to their numerical equivalents
    final_features['Exam_Explained'] = likert_map.get(raw_inputs.get('Exam_Explained', 'Neither Agree Or Disagree'), 3)
    final_features['Discuss_NextSteps'] = likert_map.get(raw_inputs.get('Discuss_NextSteps', 'Neither Agree Or Disagree'), 3)

    # --- 2. Assemble Input Dictionary ---
    # Create a dictionary containing all features the model expects, in the correct order.
    input_data = {}
    for feature in important_features:
        if feature in final_features:
            input_data[feature] = final_features[feature]
        elif feature in raw_inputs:
            input_data[feature] = raw_inputs[feature]
        else:
            # If a feature is missing from input, decide a default value.
            # For categorical features, use the first category as default.
            # For numerical, default to 0.0.
            if feature in categories and categories[feature]:
                input_data[feature] = categories[feature][0] 
            else:
                input_data[feature] = 0.0

    # --- 3. Create and Clean DataFrame ---
    # Convert the dictionary to a pandas DataFrame with columns ordered correctly.
    input_df = pd.DataFrame([input_data], columns=important_features)
    
    # Identify which of the model's features are categorical.
    categorical_features_in_model = [col for col in important_features if col in categories]
    
    # --- 4. FIX: Handle Missing Categorical Values ---
    # This loop prevents the "Invalid type for cat_feature... NaN" error.
    for col in categorical_features_in_model:
        if col in input_df.columns:
            # Define a placeholder for missing values (e.g., if 'State' is not provided).
            placeholder = 'N/A'
            
            # Replace any NaN or None values in the column with the placeholder string.
            input_df[col] = input_df[col].fillna(placeholder)

            # Get the list of known categories for this feature from your training data.
            known_categories = categories.get(col, [])
            
            # Ensure the placeholder is part of the known categories list.
            # This is crucial because pd.Categorical will create NaNs for unknown values.
            if placeholder not in known_categories:
                # Create a mutable copy and add the placeholder.
                known_categories = list(known_categories)
                known_categories.append(placeholder)
            
            # Convert the column to the 'category' dtype with the complete list of categories.
            input_df[col] = pd.Categorical(input_df[col], categories=known_categories)

    return input_df