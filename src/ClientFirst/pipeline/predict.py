import pandas as pd
import numpy as np
from pathlib import Path
from ClientFirst.utils.common import load_joblib
from ClientFirst.utils.logger import logger


class PredictionPipeline:
    def __init__(self, model_path: str = "artifacts/model_training/model.joblib"):
        """Initialize prediction pipeline with model artifacts"""
        try:
            self.model = load_joblib(Path(model_path))
            model_dir = Path(model_path).parent

            self.label_encoder = load_joblib(model_dir / "label_encoder.joblib")
            self.important_features = load_joblib(model_dir / "important_features.joblib")
            self.categorical_features = load_joblib(model_dir / "top_categorical_features.joblib")
            self.categories = load_joblib(model_dir / "categories.joblib")

            logger.info("Prediction pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing prediction pipeline: {e}")
            raise e

    def preprocess_input(self, raw_inputs: dict):
        """
        Preprocess raw input into a DataFrame suitable for model prediction.
        Handles feature engineering, mapping, and missing value imputation.
        """
        try:
            likert_map = {
                "Strongly Disagree": 1,
                "Disagree": 2,
                "Neither Agree Or Disagree": 3,
                "Agree": 4,
                "Strongly Agree": 5,
            }

            final_features = {}

            # Derived metrics
            hiv_duration = raw_inputs.get("HIV_Duration_Years", 0.0)
            care_duration = raw_inputs.get("Care_Duration_Years", 0.0)
            final_features["HIV_Care_Duration_Ratio"] = (
                hiv_duration / (care_duration + 0.1) if care_duration >= 0 else 0.0
            )

            empathy = raw_inputs.get("Empathy_Score", 0.0)
            listening = raw_inputs.get("Listening_Score", 0.0)
            decision_share = raw_inputs.get("Decision_Share_Score", 0.0)

            final_features["Empathy_Listening_Interaction"] = empathy * listening
            final_features["Empathy_DecisionShare_Interaction"] = empathy * decision_share

            # Map Likert scales
            final_features["Exam_Explained"] = likert_map.get(
                raw_inputs.get("Exam_Explained", "Neither Agree Or Disagree"), 3
            )
            final_features["Discuss_NextSteps"] = likert_map.get(
                raw_inputs.get("Discuss_NextSteps", "Neither Agree Or Disagree"), 3
            )

            # Assemble input with all expected features
            input_data = {}
            for feature in self.important_features:
                if feature in final_features:
                    input_data[feature] = final_features[feature]
                elif feature in raw_inputs:
                    value = raw_inputs[feature]
                    if feature in self.categorical_features:
                        if value is None or (isinstance(value, float) and pd.isna(value)):
                            input_data[feature] = "Unknown"
                        else:
                            input_data[feature] = str(value) if str(value).strip() else "Unknown"
                    else:
                        input_data[feature] = value if value is not None else 0.0
                else:
                    if feature in self.categories and self.categories[feature]:
                        input_data[feature] = self.categories[feature][0]
                    else:
                        input_data[feature] = 0.0

            input_df = pd.DataFrame([input_data], columns=self.important_features)

            # Handle categorical features safely
            for col in self.categorical_features:
                if col in input_df.columns:
                    input_df[col] = input_df[col].astype(str)
                    input_df[col] = input_df[col].replace(
                        ["nan", "None", "", "NaN"], "Unknown"
                    )

                    known_categories = list(self.categories.get(col, []))
                    if "Unknown" not in known_categories:
                        known_categories.append("Unknown")

                    input_df[col] = pd.Categorical(
                        input_df[col], categories=known_categories
                    )

            # Final sanity check
            input_df = input_df.fillna("Unknown")

            return input_df

        except Exception as e:
            logger.error(f"Error preprocessing input: {e}")
            raise e

    def predict(self, input_df: pd.DataFrame):
        """Make prediction and return probabilities"""
        try:
            # Final safety pass for categorical columns
            for col in self.categorical_features:
                if col in input_df.columns:
                    input_df[col] = input_df[col].astype(str)
                    input_df[col] = input_df[col].replace(
                        ["nan", "None", "", "NaN"], "Unknown"
                    )
            input_df = input_df.fillna("Unknown")

            # Debug logging
            logger.info(f"DEBUG - DataFrame dtypes: {input_df.dtypes.to_dict()}")
            logger.info(f"DEBUG - DataFrame values: {input_df.to_dict('records')[0]}")

            predictions_proba = self.model.predict_proba(input_df)[0]
            predicted_class = int(np.argmax(predictions_proba))
            confidence = float(np.max(predictions_proba))
            predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]

            return {
                "predicted_class": predicted_class,
                "predicted_label": predicted_label,
                "confidence": f"{round(confidence * 100, 1)}%",
                "probabilities": predictions_proba.tolist(),
            }

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise e

    def get_categories(self):
        """Return categories for UI dropdowns"""
        return self.categories