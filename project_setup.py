"""
Project Setup Script
Automates the creation of directory structure and __init__.py files
"""

import os
from pathlib import Path

# Define project structure
project_structure = {
    "src/ClientFirst": [
        "components",
        "config",
        "constants",
        "entity",
        "pipeline",
        "utils"
    ],
    "app": [
        "static/css",
        "static/js",
        "static/images",
        "templates"
    ],
    "artifacts": [
        "data_ingestion",
        "prepare_base_model",
        "model_training",
        "model_evaluation"
    ],
    "config": [],
    "data": [],
    "notebooks": [],
    "tests": [],
    "logs": [],
    ".github/workflows": []
}

# Files to create with __init__.py
init_files = [
    "src/__init__.py",
    "src/ClientFirst/__init__.py",
    "src/ClientFirst/components/__init__.py",
    "src/ClientFirst/config/__init__.py",
    "src/ClientFirst/constants/__init__.py",
    "src/ClientFirst/entity/__init__.py",
    "src/ClientFirst/pipeline/__init__.py",
    "src/ClientFirst/utils/__init__.py",
    "tests/__init__.py"
]

def create_directory_structure():
    """Create all project directories"""
    print("Creating directory structure...")
    
    for base_dir, subdirs in project_structure.items():
        # Create base directory
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {base_dir}")
        
        # Create subdirectories
        for subdir in subdirs:
            full_path = Path(base_dir) / subdir
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {full_path}")

def create_init_files():
    """Create __init__.py files for Python packages"""
    print("\nCreating __init__.py files...")
    
    for init_file in init_files:
        init_path = Path(init_file)
        init_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not init_path.exists():
            init_path.touch()
            print(f"✓ Created: {init_file}")
        else:
            print(f"  Skipped (exists): {init_file}")

def create_env_example():
    """Create .env.example file"""
    print("\nCreating .env.example...")
    
    env_content = """# API Keys
SATISFACTION_APP_KEY=your_openrouter_api_key_here

# Email Configuration
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password_here

# Data Path (optional)
DATA_PATH=data/processed_data.csv

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
"""
    
    env_path = Path(".env.example")
    if not env_path.exists():
        env_path.write_text(env_content)
        print("✓ Created: .env.example")
    else:
        print("  Skipped (exists): .env.example")

def create_gitignore():
    """Create .gitignore file"""
    print("\nCreating .gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Environment Variables
.env
.env.local

# Logs
logs/
*.log

# Artifacts
artifacts/
!artifacts/.gitkeep

# Model Files
*.joblib
*.pkl
*.h5
*.pb

# Data
data/processed_data.csv
data/raw/

# Testing
.pytest_cache/
.coverage
htmlcov/
coverage.xml
*.cover

# Jupyter Notebooks
.ipynb_checkpoints/
notebooks/.ipynb_checkpoints/

# Distribution
*.tar.gz
*.whl

# Documentation
docs/_build/

# Temporary Files
*.tmp
temp/
tmp/
"""
    
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        gitignore_path.write_text(gitignore_content)
        print("✓ Created: .gitignore")
    else:
        print("  Skipped (exists): .gitignore")

def create_dockerignore():
    """Create .dockerignore file"""
    print("\nCreating .dockerignore...")
    
    dockerignore_content = """# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so

# Virtual Environment
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Git
.git/
.gitignore

# Environment
.env
.env.local

# Logs
logs/
*.log

# Testing
.pytest_cache/
.coverage
htmlcov/
tests/

# Documentation
docs/
*.md
!README.md

# CI/CD
.github/

# Notebooks
notebooks/

# Artifacts (exclude during build)
artifacts/data_ingestion/
artifacts/prepare_base_model/

# Temporary
*.tmp
temp/
tmp/
"""
    
    dockerignore_path = Path(".dockerignore")
    if not dockerignore_path.exists():
        dockerignore_path.write_text(dockerignore_content)
        print("✓ Created: .dockerignore")
    else:
        print("  Skipped (exists): .dockerignore")

def create_stage_02_pipeline():
    """Create stage 02 pipeline file"""
    print("\nCreating missing pipeline files...")
    
    stage_02_content = """from ClientFirst.config.configuration import ConfigurationManager
from ClientFirst.components.prepare_base_model import PrepareBaseModel
from ClientFirst.utils.logger import logger


STAGE_NAME = "Prepare Base Model Stage"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.prepare_encoders()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\\n\\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
"""
    
    stage_02_path = Path("src/ClientFirst/pipeline/stage_02_prepare_base_model.py")
    if not stage_02_path.exists():
        stage_02_path.write_text(stage_02_content)
        print("✓ Created: stage_02_prepare_base_model.py")

def create_prepare_base_model_component():
    """Create prepare base model component"""
    
    component_content = """import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from ClientFirst.utils.logger import logger
from ClientFirst.utils.common import save_joblib
from ClientFirst.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def prepare_encoders(self):
        \"\"\"Prepare and save label encoders and categories\"\"\"
        try:
            # This is a placeholder - actual implementation depends on data
            logger.info("Prepare base model components initialized")
            logger.info(f"Artifacts will be saved to {self.config.root_dir}")
            
        except Exception as e:
            logger.error(f"Error preparing base model: {e}")
            raise e
"""
    
    component_path = Path("src/ClientFirst/components/prepare_base_model.py")
    if not component_path.exists():
        component_path.write_text(component_content)
        print("✓ Created: prepare_base_model.py")

def create_model_evaluation_component():
    """Create model evaluation component"""
    
    eval_content = """import pandas as pd
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
        \"\"\"Evaluate trained model on test data\"\"\"
        try:
            # Load model and data
            model = load_joblib(self.config.model_path)
            df = pd.read_csv(self.config.test_data_path)
            
            X = df.drop(columns=["Satisfaction"])
            y = df["Satisfaction"]
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average='weighted')
            
            # Generate report
            report = {
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "classification_report": classification_report(y, y_pred, output_dict=True)
            }
            
            # Save evaluation report
            save_json(self.config.evaluation_report_path, report)
            
            logger.info(f"Model Accuracy: {accuracy:.4f}")
            logger.info(f"Model F1-Score: {f1:.4f}")
            logger.info(f"Evaluation report saved to {self.config.evaluation_report_path}")
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise e
"""
    
    eval_path = Path("src/ClientFirst/components/model_evaluation.py")
    if not eval_path.exists():
        eval_path.write_text(eval_content)
        print("✓ Created: model_evaluation.py")

def create_stage_04_pipeline():
    """Create stage 04 pipeline file"""
    
    stage_04_content = """from ClientFirst.config.configuration import ConfigurationManager
from ClientFirst.components.model_evaluation import ModelEvaluation
from ClientFirst.utils.logger import logger


STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate_model()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\\n\\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
"""
    
    stage_04_path = Path("src/ClientFirst/pipeline/stage_04_model_evaluation.py")
    if not stage_04_path.exists():
        stage_04_path.write_text(stage_04_content)
        print("✓ Created: stage_04_model_evaluation.py")

def main():
    """Main setup function"""
    print("=" * 60)
    print("ClientFirst-XAI Project Setup")
    print("=" * 60)
    
    # Create directory structure
    create_directory_structure()
    
    # Create __init__ files
    create_init_files()
    
    # Create configuration files
    create_env_example()
    create_gitignore()
    create_dockerignore()
    
    # Create missing component files
    create_stage_02_pipeline()
    create_prepare_base_model_component()
    create_model_evaluation_component()
    create_stage_04_pipeline()
    
    print("\n" + "=" * 60)
    print("✓ Project setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Copy .env.example to .env and configure your environment variables")
    print("2. Place your processed_data.csv in the data/ directory")
    print("3. Run: python main.py (to train the model)")
    print("4. Run: uvicorn app.main:app --reload (to start the application)")
    print("=" * 60)


if __name__ == "__main__":
    main()