import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from ClientFirst.utils.logger import logger
from ClientFirst.utils.common import load_joblib, save_json
from ClientFirst.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_model(self):
        """Evaluate trained model on test data"""
        try:
            # Load model
            model = load_joblib(self.config.model_path)
            
            # Load important features, categorical features, and label encoder
            model_dir = self.config.model_path.parent
            important_features = load_joblib(model_dir / "important_features.joblib")
            categorical_features = load_joblib(model_dir / "top_categorical_features.joblib")
            label_encoder = load_joblib(model_dir / "label_encoder.joblib")
            
            logger.info(f"Important features: {important_features}")
            logger.info(f"Categorical features: {categorical_features}")
            logger.info(f"Label classes: {label_encoder.classes_}")
            
            # Load data
            df = pd.read_csv(self.config.test_data_path)
            
            if "Satisfaction" not in df.columns:
                raise ValueError("'Satisfaction' column not found in test data")
            
            # Use only important features
            X = df[important_features].copy()  # Use .copy() to avoid SettingWithCopyWarning
            y = df["Satisfaction"]
            
            # Convert categorical features to category dtype
            for col in categorical_features:
                if col in X.columns:
                    X[col] = X[col].astype('category')
            
            logger.info(f"Evaluation data shape: {X.shape}")
            
            # Make predictions (returns encoded values)
            y_pred_encoded = model.predict(X)
            
            # Decode predictions to original labels
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            
            logger.info(f"Sample predictions (encoded): {y_pred_encoded[:5]}")
            logger.info(f"Sample predictions (decoded): {y_pred[:5]}")
            logger.info(f"Sample actuals: {y.values[:5]}")
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average='weighted')
            
            # Generate classification report
            class_report = classification_report(y, y_pred, output_dict=True, zero_division=0)
            
            # Generate confusion matrix
            conf_matrix = confusion_matrix(y, y_pred, labels=label_encoder.classes_)
            
            # Create evaluation report
            report = {
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "classification_report": class_report,
                "confusion_matrix": conf_matrix.tolist(),
                "confusion_matrix_labels": label_encoder.classes_.tolist(),
                "model_info": {
                    "num_features": len(important_features),
                    "categorical_features": categorical_features,
                    "feature_names": important_features,
                    "target_classes": label_encoder.classes_.tolist()
                }
            }
            
            # Save evaluation report
            save_json(self.config.evaluation_report_path, report)
            
            logger.info(f"✅ Model Accuracy: {accuracy:.4f}")
            logger.info(f"✅ Model F1-Score: {f1:.4f}")
            logger.info(f"✅ Evaluation report saved to {self.config.evaluation_report_path}")
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise e