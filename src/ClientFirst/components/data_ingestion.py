import os
import shutil
import pandas as pd
from ClientFirst.utils.logger import logger
from ClientFirst.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def copy_data(self):
        """Copy data from source to artifacts directory"""
        try:
            if not os.path.exists(self.config.source_URL):
                raise FileNotFoundError(f"Source data file not found: {self.config.source_URL}")
            
            # Copy the file to the local data file path
            shutil.copy(self.config.source_URL, self.config.local_data_file)
            logger.info(f"Data copied from {self.config.source_URL} to {self.config.local_data_file}")
            
        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            raise e

    def validate_data(self):
        """Validate the ingested data"""
        try:
            df = pd.read_csv(self.config.local_data_file)
            
            # Check if Satisfaction column exists
            if "Satisfaction" not in df.columns:
                raise ValueError("'Satisfaction' column not found in data")
            
            logger.info(f"Data validation successful. Shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in data validation: {e}")
            raise e