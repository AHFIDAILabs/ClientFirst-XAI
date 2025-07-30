# This is the API module for the core explanation logic of the CarePredict AI application.
import shap
import pandas as pd
import numpy as np
import json
import requests
import logging
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

openrouter_api_key = os.getenv("SATISFACTION_APP_KEY")
if not openrouter_api_key:
    logging.warning("SATISFACTION_APP_KEY not found in .env file. Ensure it's set in the environment.")

label_map = {
    0: 'Very Dissatisfied',
    1: 'Dissatisfied',
    2: 'Neutral',
    3: 'Satisfied',
    4: 'Very Satisfied'
}

# Rule-Based Logic (Assuming these are correct after previous fix)
def rule_empathy_listening(instance_data):
    reasons, suggestions = [], []
    if instance_data.get('Empathy_Listening_Interaction', 15) < 9:
        reasons.append("Low empathy and poor listening likely reduced satisfaction.")
        suggestions.append("Train providers to improve empathy and active listening.")
    elif instance_data.get('Empathy_Listening_Interaction', 15) > 15:
        reasons.append("Strong empathy and active listening boosted client satisfaction.")
        suggestions.append("Encourage continued focus on empathetic listening.")
    return len(reasons) > 0, reasons, suggestions

def rule_empathy_decision_share(instance_data):
    reasons, suggestions = [], []
    if instance_data.get('Empathy_DecisionShare_Interaction', 15) < 9:
        reasons.append("Lack of empathy or poor decision-sharing contributed to dissatisfaction.")
        suggestions.append("Ensure clients feel heard and included in their care planning.")
    elif instance_data.get('Empathy_DecisionShare_Interaction', 15) > 15:
        reasons.append("Clients felt supported and involved in decision-making.")
        suggestions.append("Maintain high levels of participatory care.")
    return len(reasons) > 0, reasons, suggestions

def rule_exam_next_steps(instance_data):
    reasons, suggestions = [], []
    if instance_data.get('Exam_Explained', 3) < 3:
        reasons.append("Medical exams were not clearly explained.")
        suggestions.append("Improve communication around procedures and clinical steps.")
    if instance_data.get('Discuss_NextSteps', 3) < 3:
        reasons.append("Next steps in the care journey were not well communicated.")
        suggestions.append("Ensure every client knows what to expect after each visit.")
    return len(reasons) > 0, reasons, suggestions

def rule_overall_client_satisfaction(instance_data):
    """
    Evaluates multiple new top features to generate a comprehensive picture of satisfaction.
    """
    reasons, suggestions = [], []

    # Empathy + Listening interaction
    if instance_data.get('Empathy_Listening_Interaction', 15) < 9:
        reasons.append("Low empathy and poor listening likely reduced satisfaction.")
        suggestions.append("Train providers to improve empathy and active listening.")
    elif instance_data.get('Empathy_Listening_Interaction', 15) > 15:
        reasons.append("Strong empathy and active listening boosted client satisfaction.")
        suggestions.append("Encourage continued focus on empathetic listening.")

    # Empathy + Decision-sharing interaction
    if instance_data.get('Empathy_DecisionShare_Interaction', 15) < 9:
        reasons.append("Lack of empathy or poor decision-sharing contributed to dissatisfaction.")
        suggestions.append("Ensure clients feel heard and included in their care planning.")
    elif instance_data.get('Empathy_DecisionShare_Interaction', 15) > 15:
        reasons.append("Clients felt supported and involved in decision-making.")
        suggestions.append("Maintain high levels of participatory care.")

    # Clarity of care plan and communication
    if instance_data.get('Exam_Explained', 3) < 3:
        reasons.append("Medical exams were not clearly explained.")
        suggestions.append("Improve communication around procedures and clinical steps.")

    if instance_data.get('Discuss_NextSteps', 3) < 3:
        reasons.append("Next steps in the care journey were not well communicated.")
        suggestions.append("Ensure every client knows what to expect after each visit.")

    # Structural/Contextual
    if instance_data.get('Employment_Grouped') in ['Unemployed', 'Unknown']:
        reasons.append("Client's unemployment status may affect care experience or stress levels.")
        suggestions.append("Offer counseling and support services for unemployed clients.")

    if instance_data.get('Education_Grouped') in ['None', 'Primary']:
        reasons.append("Lower education level may be linked with reduced care understanding.")
        suggestions.append("Simplify communication and use visual aids for clarity.")

    if instance_data.get('Facility_Care_Dur_Years', 0) < 1:
        reasons.append("Short duration of care at this facility may limit relationship-building.")
        suggestions.append("Strengthen early rapport and onboarding for new clients.")

    if instance_data.get('HIV_Care_Duration_Ratio', 0.0) < 0.3:
        reasons.append("Low proportion of time spent in care may affect satisfaction.")
        suggestions.append("Reinforce retention efforts and build long-term trust.")

    return len(reasons) > 0, reasons, suggestions

RULES = [
    (
        'Empathy and listening were key factors.',
        "Encourage strong provider-client communication and emotional intelligence.",
        (rule_empathy_listening, True)
    ),
    (
        'Decision-sharing and empathy influenced satisfaction.',
        "Promote client-centered decision-making practices.",
        (rule_empathy_decision_share, True)
    ),
    (
        'Exam clarity and next-step planning mattered.',
        "Make sure clients understand their exams and what comes next.",
        (rule_exam_next_steps, True)
    ),
    (
        'Overall Client Satisfaction influenced by multiple clinical and contextual factors.',
        "Address communication, education, employment, treatment duration, and participatory care.",
        (rule_overall_client_satisfaction, True)
    )
]

# Helper Functions
def enforce_categorical_dtypes(df, categorical_cols):
    """Ensures specified columns in DataFrame are of 'category' dtype."""
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

_shap_explainer_cache = {}

def get_shap_explainer(model):
    """Returns a cached SHAP TreeExplainer for the given model."""
    model_id = id(model)
    if model_id not in _shap_explainer_cache:
        _shap_explainer_cache[model_id] = shap.TreeExplainer(model)
    return _shap_explainer_cache[model_id]


def generate_ai_explanation(prediction, confidence, top_features, reasons, suggestions):
    """Generates a detailed, structured explanation using a Generative AI model."""
    if not openrouter_api_key:
        return "GenAI explanation unavailable: API key not configured."

    # --- FIX: Pre-format the strings to avoid backslash in f-string expression ---
    reasons_str = "- " + "\n- ".join(reasons) if reasons else "None."
    suggestions_str = "- " + "\n- ".join(suggestions) if suggestions else "None."

    prompt = f"""
    You are an expert AI Data Analyst for a clinical quality improvement team. Your task is to explain a client satisfaction prediction in a clear, actionable way.

    **Client Context:**
    - **Setting:** HIV Clinic
    - **Goal:** Understand drivers of client satisfaction to improve care quality.
    - **Prediction:** The model predicts this client's satisfaction level is **'{prediction}'**.
    - **Confidence:** The model is **{confidence}** confident in this prediction.

    **AI & Rule-Based Analysis Results:**
    1.  **Top Quantitative Drivers (from SHAP model analysis):**
        ```json
        {json.dumps(top_features, indent=2)}
        ```
    2.  **Qualitative Insights (from clinical rules):**
        - **Identified Issues/Reasons:** {reasons_str}
        - **System Suggestions:** {suggestions_str}

    **Your Task:** Structure your response in three distinct sections using markdown:
    ### 1. Executive Summary
    Provide a one-paragraph robust summary of the prediction and the primary reasons behind it.
    ### 2. Analysis of Drivers
    Explain *how* the top quantitative drivers and the qualitative insights connect. Translate feature names (e.g., 'Empathy_Listening_Interaction') into plain language.
    ### 3. Actionable Recommendations
    List 2-3 concrete, practical steps the clinical team can take based on this specific client's feedback.
    """
    
    headers = {"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"}
    body = {"model": "mistralai/mistral-7b-instruct:free", "messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(body), timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        logging.error(f"GenAI API request failed: {e}")
        return f"Error connecting to GenAI service: {e}."
    except (KeyError, IndexError) as e:
        logging.error(f"Unexpected GenAI API response format: {e}. Response: {response.text if 'response' in locals() else 'No response'}")
        return "Error parsing GenAI service response."
    except Exception as e:
        logging.error(f"An unexpected error occurred in GenAI explanation: {e}")
        return f"An unexpected error occurred during GenAI explanation: {e}"

# Main Explanation Pipeline
def get_explanation(model, X_instance_df: pd.DataFrame, categorical_cols: list):
    """
    Generates prediction and a full explanation for a single instance.
    This function will be called by the FastAPI endpoint.
    """
    if X_instance_df.shape[0] != 1:
        raise ValueError("Input DataFrame must contain exactly one instance for explanation.")

    instance = enforce_categorical_dtypes(X_instance_df.copy(), categorical_cols)

    # --- Prediction ---
    preds_proba = model.predict_proba(instance)[0]
    # ✅ Convert numpy.int64 to standard Python int
    pred_class = int(np.argmax(preds_proba))
    # ✅ Ensure float conversion before string formatting
    confidence = f"{round(float(np.max(preds_proba)) * 100, 1)}%"
    mapped_pred = label_map.get(pred_class, "Unknown") # Use the converted int

    # --- SHAP Value Calculation ---
    explainer = get_shap_explainer(model)
    shap_values_raw = explainer.shap_values(instance)
    expected_value_raw = explainer.expected_value

    if isinstance(shap_values_raw, list):
        # ✅ Ensure conversion to float for each shap value
        shap_values_for_class = [float(val) for val in shap_values_raw[pred_class][0]]
        # ✅ Ensure conversion to float for base value
        base_value_for_class = float(expected_value_raw[pred_class])
    else:
        # ✅ Ensure conversion to float for each shap value
        shap_values_for_class = [float(val) for val in shap_values_raw[0, :, pred_class]]
        # ✅ Ensure conversion to float for base value
        base_value_for_class = float(expected_value_raw[pred_class])

    # ✅ Ensure all values in top_shap_features are standard floats
    shap_dict = dict(zip(instance.columns, shap_values_for_class))
    top_shap_features = {k: round(float(v), 3) for k, v in sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:3]}


    # --- Rule-Based Analysis ---
    # Convert instance_data values to standard Python types if they are numpy scalars
    instance_data = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in instance.iloc[0].to_dict().items()}

    reasons, suggestions = [], []
    for _, _, rule_tuple in RULES:
        rule_fn, _ = rule_tuple
        is_triggered, rule_reasons, rule_suggestions = rule_fn(instance_data)
        if is_triggered:
            reasons.extend(rule_reasons)
            suggestions.extend(rule_suggestions)

    # --- Generative AI Synthesis ---
    genai_explanation = generate_ai_explanation(
        mapped_pred, confidence, top_shap_features, reasons, suggestions
    )

    # Final return values: ensure everything is a standard Python type
    return {
        'prediction': mapped_pred,
        'confidence': confidence,
        'top_features': top_shap_features,
        'reasons': reasons,
        'suggestions': suggestions,
        'genai_explanation': genai_explanation,
        'shap_values': shap_values_for_class, # Already converted to list of floats
        'shap_base_value': base_value_for_class, # Already converted to float
        # ✅ Convert all feature values to standard Python types
        'feature_values': [val.item() if isinstance(val, np.generic) else val for val in instance.iloc[0].values.tolist()],
        'feature_names': list(instance.columns)
    }