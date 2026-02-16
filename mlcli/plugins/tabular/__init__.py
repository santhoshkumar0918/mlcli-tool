"""Tabular data ML plugin (default).

Generates projects for traditional ML with structured/tabular data using
scikit-learn, pandas, and xgboost.
"""

from typing import Dict, List, Any
from mlcli.plugins.base import PluginBase


class TabularPlugin(PluginBase):
    """Plugin for tabular/structured data ML projects."""
    
    name = "tabular"
    description = "Traditional ML with structured data (pandas, sklearn, xgboost)"
    dependencies = ["pandas", "scikit-learn", "xgboost"]
    
    def get_directory_structure(self) -> List[str]:
        """Return directory structure for tabular ML projects."""
        return [
            "data/raw",
            "data/processed",
            "data/external",
            "models",
            "notebooks",
            "reports/figures",
            "src",
            "tests",
            "deployments",
            "logs",
        ]
    
    def get_boilerplate_files(self) -> Dict[str, str]:
        """Return boilerplate files for tabular ML."""
        return {
            "README.md": self._get_readme(),
            ".gitignore": self._get_gitignore(),
            "requirements.txt": self.get_requirements(),
            "src/train.py": self._get_train_script(),
            "src/data_loader.py": self._get_data_loader(),
            "notebooks/01_exploratory_analysis.ipynb": self._get_notebook(),
            "tests/test_data_loader.py": self._get_test(),
        }
    
    def get_requirements(self) -> str:
        """Return requirements.txt content."""
        return """# ML Assistant CLI
mlcli

# Data processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# ML libraries
xgboost>=1.7.0

# Deployment
bentoml>=1.2.0

# Utilities
pyyaml>=6.0
rich>=13.0.0
pytest>=7.0.0
"""
    
    def get_config_template(self) -> Dict[str, Any]:
        """Return plugin-specific config."""
        return {
            "plugin": "tabular",
            "data": {
                "test_size": 0.2,
                "random_state": 42,
            },
            "model": {
                "algorithms": ["logistic_regression", "random_forest", "xgboost"],
                "cv_folds": 5,
            }
        }
    
    def _get_readme(self) -> str:
        """Get README template."""
        return """# ML Project (Tabular Data)

This project was initialized with ML Assistant CLI using the **tabular** plugin.

## Getting Started

1. Add your dataset to `data/raw/`
2. Run preprocessing: `mlcli preprocess --input data/raw/your_data.csv`
3. Train models: `mlcli train`
4. Evaluate performance: `mlcli evaluate`
5. Get suggestions: `mlcli suggest`
6. Make predictions: `mlcli predict --input new_data.csv`

## Project Structure

```
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned and preprocessed data
│   └── external/     # External datasets
├── models/           # Trained models and artifacts
├── notebooks/        # Jupyter notebooks for exploration
├── reports/          # Analysis reports and figures
├── src/              # Source code
├── tests/            # Unit tests
├── deployments/      # Deployment configurations
└── logs/             # Application logs
```

## Configuration

Edit `mlcli.yaml` to customize your ML pipeline settings.
"""
    
    def _get_gitignore(self) -> str:
        """Get .gitignore template."""
        return """# Python
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

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# ML specific
*.pkl
*.joblib
*.h5
*.hdf5
models/checkpoints/
logs/*.log
data/raw/*.csv
data/raw/*.json
data/raw/*.parquet

# Jupyter
.ipynb_checkpoints

# Testing
htmlcov/
.tox/
.coverage
.pytest_cache/

# BentoML
bentofile.yaml
bentos/
"""
    
    def _get_train_script(self) -> str:
        """Get training script template."""
        return '''"""
Starter training script for tabular ML.
Customize for your specific use case.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def load_data(filepath: str):
    """Load and prepare data."""
    df = pd.read_csv(filepath)
    return df


def train_model(X_train, y_train):
    """Train a simple model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def main():
    """Main training pipeline."""
    # Load data
    print("Loading data...")
    df = load_data("data/raw/your_data.csv")
    
    # Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save model
    joblib.dump(model, "models/model.pkl")
    print("Model saved to models/model.pkl")


if __name__ == "__main__":
    main()
'''
    
    def _get_data_loader(self) -> str:
        """Get data loader template."""
        return '''"""
Data loading and preprocessing utilities.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple


class DataLoader:
    """Handle data loading and basic preprocessing."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV file from data directory."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows from {filename}")
        return df
    
    def get_features_and_target(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Split dataframe into features and target."""
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        return X, y


# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_csv("your_data.csv")
    print(df.head())
'''
    
    def _get_notebook(self) -> str:
        """Get Jupyter notebook template."""
        return '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis\\n",
    "\\n",
    "This notebook provides a starter template for exploring your tabular dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "%matplotlib inline\\n",
    "sns.set_style('whitegrid')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "df = pd.read_csv('../data/raw/your_data.csv')\\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "df.describe()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
    
    def _get_test(self) -> str:
        """Get unit test template."""
        return '''"""
Unit tests for data loader.
Run with: pytest tests/
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import DataLoader


def test_data_loader_init():
    """Test DataLoader initialization."""
    loader = DataLoader("data/raw")
    assert loader.data_dir == Path("data/raw")


def test_get_features_and_target():
    """Test feature/target splitting."""
    loader = DataLoader()
    
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0]
    })
    
    X, y = loader.get_features_and_target(df, 'target')
    
    assert len(X.columns) == 2
    assert 'target' not in X.columns
    assert len(y) == 3
'''
