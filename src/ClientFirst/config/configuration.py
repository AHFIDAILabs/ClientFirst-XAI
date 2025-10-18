from ClientFirst.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from ClientFirst.utils.common import read_yaml, create_directories
from ClientFirst.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file)
        )
        
        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])
        
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            label_encoder_path=Path(config.label_encoder_path),
            categories_path=Path(config.categories_path)
        )
        
        return prepare_base_model_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        
        create_directories([config.root_dir])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            important_features_path=Path(config.important_features_path),
            top_categorical_features_path=Path(config.top_categorical_features_path),
            data_path=Path(self.config.data_ingestion.local_data_file),
            params=self.params
        )
        
        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        create_directories([config.root_dir])
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            evaluation_report_path=Path(config.evaluation_report_path),
            params=self.params
        )
        
        return model_evaluation_config