import os
import yaml
import json
import joblib
from box import ConfigBox
from pathlib import Path
from typing import Any
from ensure import ensure_annotations
from ClientFirst.utils.logger import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads yaml file and returns ConfigBox object.
    
    Args:
        path_to_yaml (Path): Path to yaml file
        
    Returns:
        ConfigBox: ConfigBox object
        
    Raises:
        ValueError: if yaml file is empty
        Exception: if any other error occurs
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error reading yaml file: {e}")
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Create list of directories
    
    Args:
        path_to_directories (list): List of paths of directories
        verbose (bool, optional): Ignore if multiple dirs is to be created. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save json data
    
    Args:
        path (Path): Path to json file
        data (dict): Data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load json files data
    
    Args:
        path (Path): Path to json file
        
    Returns:
        ConfigBox: Data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)


def save_joblib(data, path: Path):
    """
    Save data to joblib file
    
    Args:
        data: Data to be saved (model, encoder, etc.)
        path (Path): Path to joblib file
    """
    joblib.dump(data, path)
    logger.info(f"joblib file saved at: {path}")


def load_joblib(path: Path):
    """
    Load joblib file data
    
    Args:
        path (Path): Path to joblib file
        
    Returns:
        Data from joblib file
    """
    data = joblib.load(path)
    logger.info(f"joblib file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get size in KB
    
    Args:
        path (Path): Path of the file
        
    Returns:
        str: Size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"