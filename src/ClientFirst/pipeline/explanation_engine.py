import shap
import pandas as pd
import numpy as np
import json
import requests
import os
from pathlib import Path
from dotenv import load_dotenv
from ClientFirst.utils.logger import logger

# Load environment variables
load_dotenv()

openrouter_api_key = os.getenv("SATISFACTION_APP_KEY")
if not openrouter_api_key:
    logger.warning("SATISFACTION_APP_KEY not found. GenAI explanations will be unavailable.")

# Label mapping
LABEL_MAP = {
    0: 'Very Dissatisfied',
    1: 'Dissatisfied',
    2: 'Neutral',
    3: 'Satisfied',
    4: 'Very Satisfied'
}

# Cached explainer
_shap_explainer_cache = {}


def get_shap_explainer(model):
    """Returns a cached SHAP TreeExplainer for the given model."""
    model_id = id(model)
    if model_id not in _shap_explainer_cache:
        _shap_explainer_cache[model_id] = shap.TreeExplainer(model)
        logger.info("SHAP explainer created and cached")
    return _shap_explainer_cache[model_id]


def enforce_categorical_dtypes(df, categorical_cols):
    """Ensures specified columns in DataFrame are of 'category' dtype."""
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df


class RuleBasedAnalyzer:
    """Rule-based analysis for generating qualitative insights"""
    
    @staticmethod
    def analyze_empathy_listening(instance_data):
        reasons, suggestions = [], []
        score = instance_data.get('Empathy_Listening_Interaction', 15)
        
        if score < 9:
            reasons.append("Low empathy and poor listening likely reduced satisfaction.")
            suggestions.append("Train providers to improve empathy and active listening.")
        elif score > 15:
            reasons.append("Strong empathy and active listening boosted client satisfaction.")
            suggestions.append("Encourage continued focus on empathetic listening.")
        
        return len(reasons) > 0, reasons, suggestions

    @staticmethod
    def analyze_decision_sharing(instance_data):
        reasons, suggestions = [], []
        score = instance_data.get('Empathy_DecisionShare_Interaction', 15)
        
        if score < 9:
            reasons.append("Lack of empathy or poor decision-sharing contributed to dissatisfaction.")
            suggestions.append("Ensure clients feel heard and included in their care planning.")
        elif score > 15:
            reasons.append("Clients felt supported and involved in decision-making.")
            suggestions.append("Maintain high levels of participatory care.")
        
        return len(reasons) > 0, reasons, suggestions

    @staticmethod
    def analyze_communication(instance_data):
        reasons, suggestions = [], []
        
        if instance_data.get('Exam_Explained', 3) < 3:
            reasons.append("Medical exams were not clearly explained.")
            suggestions.append("Improve communication around procedures and clinical steps.")
        
        if instance_data.get('Discuss_NextSteps', 3) < 3:
            reasons.append("Next steps in the care journey were not well communicated.")
            suggestions.append("Ensure every client knows what to expect after each visit.")
        
        return len(reasons) > 0, reasons, suggestions

    @staticmethod
    def analyze_context(instance_data):
        reasons, suggestions = [], []
        
        employment = instance_data.get('Employment_Grouped')
        if employment in ['Unemployed', 'Unknown']:
            reasons.append("Client's unemployment status may affect care experience or stress levels.")
            suggestions.append("Offer counseling and support services for unemployed clients.")
        
        education = instance_data.get('Education_Grouped')
        if education in ['None', 'Primary']:
            reasons.append("Lower education level may be linked with reduced care understanding.")
            suggestions.append("Simplify communication and use visual aids for clarity.")
        
        if instance_data.get('Facility_Care_Dur_Years', 0) < 1:
            reasons.append("Short duration of care at this facility may limit relationship-building.")
            suggestions.append("Strengthen early rapport and onboarding for new clients.")
        
        if instance_data.get('HIV_Care_Duration_Ratio', 0.0) < 0.3:
            reasons.append("Low proportion of time spent in care may affect satisfaction.")
            suggestions.append("Reinforce retention efforts and build long-term trust.")
        
        return len(reasons) > 0, reasons, suggestions

    @classmethod
    def get_all_rules(cls):
        """Returns all rule analysis functions"""
        return [
            cls.analyze_empathy_listening,
            cls.analyze_decision_sharing,
            cls.analyze_communication,
            cls.analyze_context
        ]


def generate_ai_explanation(prediction, confidence, top_features, reasons, suggestions):
    """Generates a detailed explanation using GenAI model with multiple fallbacks"""
    if not openrouter_api_key:
        logger.warning("OpenRouter API key not configured")
        return generate_intelligent_fallback(prediction, confidence, top_features, reasons, suggestions)

    reasons_str = "- " + "\n- ".join(reasons) if reasons else "None identified."
    suggestions_str = "- " + "\n- ".join(suggestions) if suggestions else "None available."

    prompt = f"""
You are an expert AI Data Analyst for a clinical quality improvement team. Your task is to explain a client satisfaction prediction in a clear, actionable way.

**Client Context:**
- **Setting:** HIV Clinic
- **Goal:** Understand drivers of client satisfaction to improve care quality.
- **Prediction:** The model predicts this client's satisfaction level is **'{prediction}'**.
- **Confidence:** The model is **{confidence}** confident in this prediction.

**AI & Rule-Based Analysis Results:**
1. **Top Quantitative Drivers (from SHAP model analysis):**
```json
    {json.dumps(top_features, indent=2)}
```
2. **Qualitative Insights (from clinical rules):**
    - **Identified Issues/Reasons:** 
    {reasons_str}
    - **System Suggestions:** 
    {suggestions_str}

**Your Task:** Structure your response in three distinct sections using markdown:
### 1. Executive Summary
Provide a one-paragraph summary of the prediction and primary reasons.

### 2. Analysis of Drivers
Explain how the top drivers and qualitative insights connect. Translate technical features into plain language.

### 3. Actionable Recommendations
List 2-3 concrete, practical steps the clinical team can take based on this specific client's feedback.
"""
    
    # Updated list of working free models (as of 2025)
    free_models = [
        "qwen/qwen-2.5-72b-instruct:free",
        "qwen/qwen-2.5-coder-32b-instruct:free",
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "google/gemma-2-9b-it:free",
        "mistralai/mistral-7b-instruct:free",
        "deepseek/deepseek-r1:free",
        "openrouter/auto:free"
    ]
    
    # Try each model in sequence
    for model in free_models:
        try:
            logger.info(f"Attempting to use model: {model}")
            
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://gamsuwa.app",
                "X-Title": "Gamsuwa HIV Client Satisfaction Platform"
            }
            
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(body),
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content'].strip()
                    if content and len(content) > 50:  # Ensure meaningful content
                        logger.info(f"Successfully generated explanation using {model}")
                        return content
                    else:
                        logger.warning(f"Model {model} returned empty or too short content")
            else:
                logger.warning(f"Model {model} returned status {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.warning(f"Model {model} timed out, trying next model...")
            continue
        except requests.exceptions.RequestException as e:
            logger.warning(f"Model {model} request failed: {e}")
            continue
        except Exception as e:
            logger.warning(f"Unexpected error with model {model}: {e}")
            continue
    
    # If all models fail, use intelligent fallback
    logger.warning("All GenAI models failed, using intelligent fallback")
    return generate_intelligent_fallback(prediction, confidence, top_features, reasons, suggestions)

def generate_intelligent_fallback(prediction, confidence, top_features, reasons, suggestions):
    """Generate an intelligent fallback explanation using available context"""
    
    # Determine satisfaction level context
    satisfaction_context = {
        'Very Satisfied': ('excellent', 'exceptional', 'highly positive'),
        'Satisfied': ('good', 'positive', 'favorable'),
        'Neutral': ('moderate', 'mixed', 'average'),
        'Dissatisfied': ('concerning', 'below expectations', 'needs improvement'),
        'Very Dissatisfied': ('critical', 'unsatisfactory', 'urgent attention required')
    }
    
    level_desc = satisfaction_context.get(prediction, ('moderate', 'mixed', 'average'))
    
    explanation = f"""### 1. Executive Summary

Our analysis predicts this client's satisfaction level as **{prediction}** with **{confidence}** confidence. This represents a **{level_desc[0]}** level of satisfaction with the healthcare services received. The prediction is based on comprehensive analysis of provider-client interactions, communication quality, and care delivery factors.

### 2. Analysis of Drivers

The key factors influencing this prediction are:

"""
    
    # Add top SHAP features with context
    for i, (feature, value) in enumerate(top_features.items(), 1):
        impact_direction = "positively" if value > 0 else "negatively"
        impact_strength = "strongly" if abs(value) > 0.2 else "moderately" if abs(value) > 0.1 else "slightly"
        
        # Humanize feature names
        feature_readable = feature.replace('_', ' ').replace('Interaction', 'interaction between').title()
        
        explanation += f"**{i}. {feature_readable}** (SHAP value: {value:.3f})\n"
        explanation += f"   - This factor {impact_strength} {impact_direction} influences the satisfaction prediction.\n"
        
        # Add context based on feature name
        if 'Empathy' in feature or 'Listening' in feature:
            explanation += f"   - {'Strong' if value > 0 else 'Weak'} provider-client rapport and active listening are reflected here.\n"
        elif 'Decision' in feature:
            explanation += f"   - Client {'felt involved' if value > 0 else 'may not have felt fully involved'} in care decisions.\n"
        elif 'Duration' in feature:
            explanation += f"   - The length of care relationship {'positively contributes' if value > 0 else 'may need attention'}.\n"
        elif 'Explained' in feature or 'NextSteps' in feature:
            explanation += f"   - Communication about procedures and care plans was {'clear and effective' if value > 0 else 'unclear or insufficient'}.\n"
        
        explanation += "\n"
    
    # Add rule-based insights
    if reasons:
        explanation += "**Clinical Rule Analysis:**\n\n"
        for reason in reasons:
            explanation += f"- {reason}\n"
        explanation += "\n"
    
    explanation += "### 3. Actionable Recommendations\n\n"
    
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            explanation += f"**{i}. {suggestion}**\n"
            
            # Add implementation hints based on suggestion content
            if 'empathy' in suggestion.lower() or 'listening' in suggestion.lower():
                explanation += "   - Consider training sessions on active listening and empathetic communication\n"
                explanation += "   - Encourage providers to spend quality time understanding client concerns\n"
            elif 'decision' in suggestion.lower():
                explanation += "   - Implement shared decision-making protocols\n"
                explanation += "   - Provide clear information about treatment options\n"
            elif 'communication' in suggestion.lower() or 'explain' in suggestion.lower():
                explanation += "   - Use plain language when explaining medical procedures\n"
                explanation += "   - Ensure clients understand their care plans before leaving\n"
            elif 'support' in suggestion.lower() or 'counseling' in suggestion.lower():
                explanation += "   - Connect clients with available support services\n"
                explanation += "   - Consider peer support programs\n"
            
            explanation += "\n"
    else:
        # Generic but useful recommendations
        if prediction in ['Dissatisfied', 'Very Dissatisfied']:
            explanation += """**1. Immediate Priority Actions**
   - Schedule follow-up conversation with this client
   - Review and address specific concerns identified
   - Assign a care coordinator for personalized support

**2. Provider Communication Enhancement**
   - Reinforce active listening and empathy in provider training
   - Implement communication quality checklists
   - Encourage extra time for patient questions

**3. System-Level Improvements**
   - Review care delivery processes for gaps
   - Strengthen patient education materials
   - Enhance provider-client relationship building
"""
        elif prediction == 'Neutral':
            explanation += """**1. Strengthen Positive Aspects**
   - Build on existing good practices
   - Encourage consistent quality across all interactions
   - Focus on relationship-building opportunities

**2. Address Identified Gaps**
   - Improve areas scoring below expectations
   - Enhance communication clarity
   - Ensure comprehensive information sharing

**3. Monitor Progress**
   - Regular satisfaction check-ins
   - Track improvements in key interaction areas
   - Gather ongoing feedback
"""
        else:  # Satisfied or Very Satisfied
            explanation += """**1. Maintain Excellence**
   - Continue current best practices
   - Recognize and replicate successful approaches
   - Share success strategies across the team

**2. Sustain Quality**
   - Regular training refreshers
   - Consistent application of patient-centered care principles
   - Monitor for any service degradation

**3. Continuous Improvement**
   - Seek opportunities to exceed expectations further
   - Stay updated on best practices
   - Foster innovation in care delivery
"""
    
    explanation += f"\n---\n\n*Note: This analysis was generated using rule-based clinical insights and machine learning explainability (SHAP values). The {confidence} confidence level indicates {'high' if float(confidence.rstrip('%')) > 80 else 'moderate' if float(confidence.rstrip('%')) > 60 else 'cautious'} certainty in this prediction.*"
    
    return explanation

def get_explanation(model, X_instance_df: pd.DataFrame, categorical_cols: list):
    """
    Generates prediction and full explanation for a single instance.
    """
    try:
        if X_instance_df.shape[0] != 1:
            raise ValueError("Input DataFrame must contain exactly one instance for explanation.")

        instance = enforce_categorical_dtypes(X_instance_df.copy(), categorical_cols)

        # Prediction
        preds_proba = model.predict_proba(instance)[0]
        pred_class = int(np.argmax(preds_proba))
        confidence = f"{round(float(np.max(preds_proba)) * 100, 1)}%"
        mapped_pred = LABEL_MAP.get(pred_class, "Unknown")

        # SHAP Values
        explainer = get_shap_explainer(model)
        shap_values_raw = explainer.shap_values(instance)
        expected_value_raw = explainer.expected_value

        if isinstance(shap_values_raw, list):
            shap_values_for_class = [float(val) for val in shap_values_raw[pred_class][0]]
            base_value_for_class = float(expected_value_raw[pred_class])
        else:
            shap_values_for_class = [float(val) for val in shap_values_raw[0, :, pred_class]]
            base_value_for_class = float(expected_value_raw[pred_class])

        shap_dict = dict(zip(instance.columns, shap_values_for_class))
        top_shap_features = {
            k: round(float(v), 3)
            for k, v in sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
        }

        # Rule-Based Analysis
        instance_data = {
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in instance.iloc[0].to_dict().items()
        }

        reasons, suggestions = [], []
        for rule_fn in RuleBasedAnalyzer.get_all_rules():
            is_triggered, rule_reasons, rule_suggestions = rule_fn(instance_data)
            if is_triggered:
                reasons.extend(rule_reasons)
                suggestions.extend(rule_suggestions)

        # GenAI Synthesis
        genai_explanation = generate_ai_explanation(
            mapped_pred, confidence, top_shap_features, reasons, suggestions
        )

        return {
            'prediction': mapped_pred,
            'confidence': confidence,
            'top_features': top_shap_features,
            'reasons': reasons,
            'suggestions': suggestions,
            'genai_explanation': genai_explanation,
            'shap_values': shap_values_for_class,
            'shap_base_value': base_value_for_class,
            'feature_values': [
                val.item() if isinstance(val, np.generic) else val
                for val in instance.iloc[0].values.tolist()
            ],
            'feature_names': list(instance.columns)
        }
        
    except Exception as e:
        logger.error(f"Error in explanation generation: {e}")
        raise e