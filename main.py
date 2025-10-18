from ClientFirst.utils.logger import logger
from ClientFirst.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from ClientFirst.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from ClientFirst.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from ClientFirst.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline
from pathlib import Path

def check_artifacts_exist(stage_name, artifact_paths):
    """Check if artifacts from a stage already exist"""
    all_exist = all(Path(path).exists() for path in artifact_paths)
    if all_exist:
        logger.info(f"[SKIP] {stage_name} - artifacts already exist")
        return True
    return False

# Stage 1: Data Ingestion
STAGE_NAME = "Data Ingestion Stage"
try:
    if not check_artifacts_exist(STAGE_NAME, ["artifacts/data_ingestion/data.csv"]):
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 2: Prepare Base Model
STAGE_NAME = "Prepare Base Model Stage"
try:
    if not check_artifacts_exist(STAGE_NAME, [
        "artifacts/prepare_base_model/label_encoder.joblib",
        "artifacts/prepare_base_model/categories.joblib"
    ]):
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 3: Model Training
STAGE_NAME = "Model Training Stage"
try:
    if not check_artifacts_exist(STAGE_NAME, [
        "artifacts/model_training/model.joblib",
        "artifacts/model_training/important_features.joblib"
    ]):
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_training = ModelTrainingPipeline()
        model_training.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 4: Model Evaluation
STAGE_NAME = "Model Evaluation Stage"
try:
    if not check_artifacts_exist(STAGE_NAME, [
        "artifacts/model_evaluation/evaluation_report.json"
    ]):
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_evaluation = ModelEvaluationPipeline()
        model_evaluation.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

logger.info("=" * 60)
logger.info("SUCCESS: All pipeline stages completed!")
logger.info("=" * 60)