from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    label_encoder_path: Path
    categories_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_path: Path
    important_features_path: Path
    top_categorical_features_path: Path
    data_path: Path
    params: dict

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    evaluation_report_path: Path
    params: dict