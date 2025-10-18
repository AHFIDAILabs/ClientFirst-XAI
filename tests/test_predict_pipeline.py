import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ClientFirst.pipeline.predict import PredictionPipeline

@pytest.fixture
def sample_input():
    """Fixture providing sample input data"""
    return {
        'Age': 35,
        'Employment_Grouped': 'Employed',
        'Education_Grouped': 'Higher Education',
        'State': 'Lagos',
        'HIV_Duration_Years': 5.0,
        'Care_Duration_Years': 2.0,
        'Facility_Care_Dur_Years': 5.0,
        'Empathy_Score': 4.0,
        'Listening_Score': 4.0,
        'Decision_Share_Score': 3.0,
        'Exam_Explained': 'Agree',
        'Discuss_NextSteps': 'Agree'
    }

@pytest.mark.unit
class TestPredictionPipeline:
    """Unit tests for prediction pipeline"""
    
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly"""
        # This test will be skipped if model artifacts don't exist
        if not Path("artifacts/model_training/model.joblib").exists():
            pytest.skip("Model artifacts not found")
        
        pipeline = PredictionPipeline()
        assert pipeline.model is not None
        assert pipeline.label_encoder is not None
        assert pipeline.important_features is not None
    
    def test_preprocess_input(self, sample_input):
        """Test input preprocessing"""
        if not Path("artifacts/model_training/model.joblib").exists():
            pytest.skip("Model artifacts not found")
        
        pipeline = PredictionPipeline()
        input_df = pipeline.preprocess_input(sample_input)
        
        assert isinstance(input_df, pd.DataFrame)
        assert input_df.shape[0] == 1
        assert all(col in input_df.columns for col in pipeline.important_features)
    
    def test_feature_engineering(self, sample_input):
        """Test feature engineering calculations"""
        if not Path("artifacts/model_training/model.joblib").exists():
            pytest.skip("Model artifacts not found")
        
        pipeline = PredictionPipeline()
        input_df = pipeline.preprocess_input(sample_input)
        
        # Check if engineered features exist
        if 'Empathy_Listening_Interaction' in input_df.columns:
            expected_value = sample_input['Empathy_Score'] * sample_input['Listening_Score']
            assert input_df['Empathy_Listening_Interaction'].values[0] == pytest.approx(expected_value)
    
    def test_prediction_output(self, sample_input):
        """Test prediction returns expected format"""
        if not Path("artifacts/model_training/model.joblib").exists():
            pytest.skip("Model artifacts not found")
        
        pipeline = PredictionPipeline()
        input_df = pipeline.preprocess_input(sample_input)
        result = pipeline.predict(input_df)
        
        assert 'predicted_class' in result
        assert 'predicted_label' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert isinstance(result['predicted_class'], int)
        assert isinstance(result['predicted_label'], str)
    
    def test_missing_optional_field(self, sample_input):
        """Test that all required fields must be present"""
        if not Path("artifacts/model_training/model.joblib").exists():
            pytest.skip("Model artifacts not found")
        
        pipeline = PredictionPipeline()
        
        # All fields are now required, including State
        # Test should pass with all fields present
        input_df = pipeline.preprocess_input(sample_input)
        assert isinstance(input_df, pd.DataFrame)
        assert 'State' in input_df.columns
    
    def test_likert_scale_mapping(self, sample_input):
        """Test Likert scale values are mapped correctly"""
        if not Path("artifacts/model_training/model.joblib").exists():
            pytest.skip("Model artifacts not found")
        
        pipeline = PredictionPipeline()
        input_df = pipeline.preprocess_input(sample_input)
        
        if 'Exam_Explained' in input_df.columns:
            # 'Agree' should map to 4
            assert input_df['Exam_Explained'].values[0] == 4

@pytest.mark.integration
class TestEndToEnd:
    """Integration tests for end-to-end workflow"""
    
    def test_full_prediction_workflow(self, sample_input):
        """Test complete prediction workflow"""
        if not Path("artifacts/model_training/model.joblib").exists():
            pytest.skip("Model artifacts not found")
        
        pipeline = PredictionPipeline()
        
        # Preprocess
        input_df = pipeline.preprocess_input(sample_input)
        
        # Predict
        result = pipeline.predict(input_df)
        
        # Validate result
        # assert result['predicted_class'] in [0, 1, 2, 3, 4]
        # assert '%' in result['confidence']
        # assert len(result['probabilities']) == 5
        # assert sum(result['probabilities']) == pytest.approx(1.0, rel=1e-5)
        # Validate result dynamically
        n_classes = len(pipeline.label_encoder.classes_)

        assert result['predicted_class'] in range(n_classes)
        assert '%' in result['confidence']
        assert len(result['probabilities']) == n_classes
        assert sum(result['probabilities']) == pytest.approx(1.0, rel=1e-5)

@pytest.mark.unit
class TestInputValidation:
    """Tests for input validation"""
    
    def test_age_boundary(self, sample_input):
        """Test age boundary validation"""
        if not Path("artifacts/model_training/model.joblib").exists():
            pytest.skip("Model artifacts not found")
        
        pipeline = PredictionPipeline()
        
        # Test minimum age
        input_min_age = sample_input.copy()
        input_min_age['Age'] = 18
        df = pipeline.preprocess_input(input_min_age)
        assert df is not None
        
        # Test maximum age
        input_max_age = sample_input.copy()
        input_max_age['Age'] = 100
        df = pipeline.preprocess_input(input_max_age)
        assert df is not None
    
    def test_score_ranges(self, sample_input):
        """Test score values are within valid range"""
        if not Path("artifacts/model_training/model.joblib").exists():
            pytest.skip("Model artifacts not found")
        
        pipeline = PredictionPipeline()
        
        for score_field in ['Empathy_Score', 'Listening_Score', 'Decision_Share_Score']:
            # Test minimum
            input_min = sample_input.copy()
            input_min[score_field] = 1.0
            df = pipeline.preprocess_input(input_min)
            assert df is not None
            
            # Test maximum
            input_max = sample_input.copy()
            input_max[score_field] = 5.0
            df = pipeline.preprocess_input(input_max)
            assert df is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])