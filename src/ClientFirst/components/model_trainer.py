import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier
from ClientFirst.utils.logger import logger
from ClientFirst.utils.common import save_joblib
from ClientFirst.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model = None
        self.label_encoder = None
        self.important_features = None
        self.categorical_features = None

    def load_and_prepare_data(self):
        """Load data and prepare for training"""
        try:
            logger.info(f"Loading data from {self.config.data_path}")
            df = pd.read_csv(self.config.data_path)
            
            if "Satisfaction" not in df.columns:
                raise ValueError("'Satisfaction' column not found in data")
            
            X = df.drop(columns=["Satisfaction"])
            y = df["Satisfaction"]
            
            # Identify categorical features
            self.categorical_features = X.select_dtypes(include="object").columns.tolist()
            
            # Encode target
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            logger.info(f"Data loaded successfully. Shape: {X.shape}")
            logger.info(f"Target classes: {self.label_encoder.classes_}")
            
            return X, y_encoded
            
        except Exception as e:
            logger.error(f"Error loading and preparing data: {e}")
            raise e

    def train_model(self, X, y):
        """Train the model with hyperparameter tuning"""
        try:
            # Compute class weights
            classes = np.unique(y)
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y
            )
            class_weight_dict = dict(zip(classes, class_weights))
            
            logger.info(f"Class weights: {class_weight_dict}")
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                stratify=y,
                test_size=self.config.params.TRAINING.test_size,
                random_state=self.config.params.TRAINING.random_state
            )
            
            logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
            
            # CatBoost with hyperparameter tuning
            model = CatBoostClassifier(
                verbose=0,
                random_state=self.config.params.TRAINING.random_state,
                class_weights=class_weights
            )
            
            param_grid = {
                'depth': self.config.params.MODEL_PARAMS.depth,
                'learning_rate': self.config.params.MODEL_PARAMS.learning_rate,
                'iterations': self.config.params.MODEL_PARAMS.iterations,
                'l2_leaf_reg': self.config.params.MODEL_PARAMS.l2_leaf_reg,
                'border_count': self.config.params.MODEL_PARAMS.border_count
            }
            
            cv = StratifiedKFold(
                n_splits=self.config.params.TRAINING.cv_splits,
                shuffle=True,
                random_state=self.config.params.TRAINING.random_state
            )
            
            logger.info("Starting GridSearchCV...")
            grid_search = GridSearchCV(
                model,
                param_grid,
                scoring=self.config.params.TRAINING.scoring,
                cv=cv,
                n_jobs=self.config.params.TRAINING.n_jobs,
                verbose=2,
                refit=True
            )
            
            grid_search.fit(X_train, y_train, cat_features=self.categorical_features)
            
            best_model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Get top features
            importances = best_model.feature_importances_
            top_n = self.config.params.TRAINING.top_n_features
            top_indices = np.argsort(importances)[::-1][:top_n]
            self.important_features = X.columns[top_indices].tolist()
            
            logger.info(f"Top {top_n} features: {self.important_features}")
            
            # Train final model on top features
            X_top = X[self.important_features]
            top_categorical = [col for col in self.important_features if col in self.categorical_features]
            
            final_model = CatBoostClassifier(
                **grid_search.best_params_,
                verbose=0,
                random_state=self.config.params.TRAINING.random_state,
                class_weights=class_weights
            )
            
            final_model.fit(X_top, y, cat_features=top_categorical)
            
            self.model = final_model
            self.categorical_features = top_categorical
            
            logger.info("Model training completed successfully")
            
            return final_model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise e

    def save_model_artifacts(self):
        """Save trained model and related artifacts"""
        try:
            # Save model
            save_joblib(self.model, self.config.trained_model_path)
            logger.info(f"Model saved to {self.config.trained_model_path}")
            
            # Save label encoder
            encoder_path = self.config.root_dir / "label_encoder.joblib"
            save_joblib(self.label_encoder, encoder_path)
            logger.info(f"Label encoder saved to {encoder_path}")
            
            # Save important features
            save_joblib(self.important_features, self.config.important_features_path)
            logger.info(f"Important features saved to {self.config.important_features_path}")
            
            # Save categorical features
            save_joblib(self.categorical_features, self.config.top_categorical_features_path)
            logger.info(f"Categorical features saved to {self.config.top_categorical_features_path}")
            
            # Save categories for UI dropdowns
            df = pd.read_csv(self.config.data_path)
            categories = {
                col: sorted(df[col].dropna().unique().tolist())
                for col in self.categorical_features
            }
            categories_path = self.config.root_dir / "categories.joblib"
            save_joblib(categories, categories_path)
            logger.info(f"Categories saved to {categories_path}")
            
        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
            raise e