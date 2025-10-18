import pytest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ClientFirst.utils.logger import logger
from src.ClientFirst.utils.common import create_directories, read_yaml
from src.ClientFirst.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH


@pytest.mark.unit
class TestProjectStructure:
    """Tests for project structure and basic setup"""
    
    def test_config_files_exist(self):
        """Test that configuration files exist"""
        assert Path("config/config.yaml").exists(), "config.yaml not found"
        assert Path("config/params.yaml").exists(), "params.yaml not found"
    
    def test_src_package_exists(self):
        """Test that src package is properly structured"""
        assert Path("src/ClientFirst").exists(), "ClientFirst package not found"
        assert Path("src/ClientFirst/__init__.py").exists(), "__init__.py not found"
    
    def test_required_directories(self):
        """Test that required directories exist or can be created"""
        required_dirs = [
            "src/ClientFirst/components",
            "src/ClientFirst/config",
            "src/ClientFirst/entity",
            "src/ClientFirst/pipeline",
            "src/ClientFirst/utils",
            "app/static/css",
            "app/static/js",
            "app/templates",
            "tests"
        ]
        
        for directory in required_dirs:
            assert Path(directory).exists(), f"Directory {directory} not found"


@pytest.mark.unit
class TestConfiguration:
    """Tests for configuration management"""
    
    def test_read_config_yaml(self):
        """Test reading configuration file"""
        try:
            config = read_yaml(CONFIG_FILE_PATH)
            assert config is not None
            assert hasattr(config, 'artifacts_root')
        except Exception as e:
            pytest.fail(f"Failed to read config.yaml: {e}")
    
    def test_read_params_yaml(self):
        """Test reading parameters file"""
        try:
            params = read_yaml(PARAMS_FILE_PATH)
            assert params is not None
            assert hasattr(params, 'MODEL_PARAMS')
        except Exception as e:
            pytest.fail(f"Failed to read params.yaml: {e}")
    
    def test_config_has_required_keys(self):
        """Test that config has all required keys"""
        config = read_yaml(CONFIG_FILE_PATH)
        required_keys = [
            'artifacts_root',
            'data_ingestion',
            'prepare_base_model',
            'model_trainer',
            'model_evaluation'
        ]
        
        for key in required_keys:
            assert hasattr(config, key), f"Config missing key: {key}"
    
    def test_params_has_model_params(self):
        """Test that params has model configuration"""
        params = read_yaml(PARAMS_FILE_PATH)
        assert hasattr(params, 'MODEL_PARAMS')
        assert hasattr(params, 'TRAINING')
        
        # Check MODEL_PARAMS contents
        model_params = params.MODEL_PARAMS
        assert hasattr(model_params, 'depth')
        assert hasattr(model_params, 'learning_rate')
        assert hasattr(model_params, 'iterations')


@pytest.mark.unit
class TestUtilities:
    """Tests for utility functions"""
    
    def test_logger_creation(self):
        """Test that logger is properly configured"""
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')
    
    def test_create_directories(self):
        """Test directory creation utility"""
        test_dir = Path("test_artifacts/temp")
        try:
            create_directories([str(test_dir)], verbose=False)
            assert test_dir.exists()
        finally:
            # Cleanup
            if test_dir.exists():
                test_dir.rmdir()
                test_dir.parent.rmdir()
    
    def test_logger_writes_to_file(self):
        """Test that logger writes to log file"""
        log_dir = Path("logs")
        log_file = log_dir / "running_logs.log"
        
        # Create log directory if it doesn't exist
        log_dir.mkdir(exist_ok=True)
        
        # Write a test message
        test_message = "Test log message from pytest"
        logger.info(test_message)
        
        # Check if log file exists
        assert log_file.exists(), "Log file not created"


@pytest.mark.unit
class TestEntityClasses:
    """Tests for entity configuration classes"""
    
    def test_data_ingestion_config_import(self):
        """Test importing DataIngestionConfig"""
        try:
            from src.ClientFirst.entity.config_entity import DataIngestionConfig
            assert DataIngestionConfig is not None
        except ImportError as e:
            pytest.fail(f"Failed to import DataIngestionConfig: {e}")
    
    def test_model_trainer_config_import(self):
        """Test importing ModelTrainerConfig"""
        try:
            from src.ClientFirst.entity.config_entity import ModelTrainerConfig
            assert ModelTrainerConfig is not None
        except ImportError as e:
            pytest.fail(f"Failed to import ModelTrainerConfig: {e}")
    
    def test_all_config_entities(self):
        """Test importing all configuration entities"""
        try:
            from src.ClientFirst.entity.config_entity import (
                DataIngestionConfig,
                PrepareBaseModelConfig,
                ModelTrainerConfig,
                ModelEvaluationConfig
            )
            assert all([
                DataIngestionConfig,
                PrepareBaseModelConfig,
                ModelTrainerConfig,
                ModelEvaluationConfig
            ])
        except ImportError as e:
            pytest.fail(f"Failed to import config entities: {e}")


@pytest.mark.unit
class TestPipelineImports:
    """Tests for pipeline module imports"""
    
    def test_import_stage_01(self):
        """Test importing stage 01 pipeline"""
        try:
            from src.ClientFirst.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
            assert DataIngestionTrainingPipeline is not None
        except ImportError as e:
            pytest.fail(f"Failed to import stage_01: {e}")
    
    def test_import_predict_pipeline(self):
        """Test importing prediction pipeline (will skip model loading)"""
        try:
            # Just test the import, not initialization
            from src.ClientFirst.pipeline import predict
            assert predict is not None
        except ImportError as e:
            pytest.fail(f"Failed to import predict module: {e}")
    
    def test_import_explanation_engine(self):
        """Test importing explanation engine"""
        try:
            from src.ClientFirst.pipeline import explanation_engine
            assert explanation_engine is not None
        except ImportError as e:
            pytest.fail(f"Failed to import explanation_engine: {e}")


@pytest.mark.unit
class TestComponentImports:
    """Tests for component module imports"""
    
    def test_import_data_ingestion(self):
        """Test importing data ingestion component"""
        try:
            from src.ClientFirst.components.data_ingestion import DataIngestion
            assert DataIngestion is not None
        except ImportError as e:
            pytest.fail(f"Failed to import DataIngestion: {e}")
    
    def test_import_model_trainer(self):
        """Test importing model trainer component"""
        try:
            from src.ClientFirst.components.model_trainer import ModelTrainer
            assert ModelTrainer is not None
        except ImportError as e:
            pytest.fail(f"Failed to import ModelTrainer: {e}")


@pytest.mark.integration
class TestDataValidation:
    """Tests for data validation (if data exists)"""
    
    def test_data_file_exists_or_skip(self):
        """Check if training data exists"""
        data_path = Path("data/processed_data.csv")
        if not data_path.exists():
            pytest.skip("Training data not found - this is expected for initial setup")
        
        import pandas as pd
        df = pd.read_csv(data_path)
        assert not df.empty, "Data file is empty"
        assert "Satisfaction" in df.columns, "Satisfaction column missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])