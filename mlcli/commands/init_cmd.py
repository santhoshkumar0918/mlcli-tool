"""Initialize command for ML Assistant CLI."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt

from mlcli.core.config import MLCLIConfig, save_config
from mlcli.core.exceptions import ConfigurationError
from mlcli.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()

app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Project name"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Project description"),
    plugin: str = typer.Option("tabular", "--plugin", "-p", help="Project template (tabular, chatbot)"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing configuration"),
) -> None:
    """Initialize a new ML project with configuration and directory structure."""
    
    project_dir = ctx.obj["project_dir"]
    config_file = project_dir / "mlcli.yaml"
    
    # Check if project already exists
    if config_file.exists() and not force:
        if not Confirm.ask(f"Project configuration already exists at {config_file}. Overwrite?"):
            console.print("[yellow]Initialization cancelled.[/yellow]")
            raise typer.Exit(0)
    
    # Get plugin from registry
    from mlcli.plugins.registry import PluginRegistry
    try:
        registry = PluginRegistry()
        selected_plugin = registry.get(plugin)
        console.print(f"[cyan]Using plugin:[/cyan] {selected_plugin.description}")
    except Exception as e:
        console.print(f"[red]Error loading plugin '{plugin}':[/red] {e}")
        available = registry.list_available() if 'registry' in locals() else []
        if available:
            console.print(f"[yellow]Available plugins:[/yellow] {', '.join(available)}")
        raise typer.Exit(1)
    
    # Interactive project setup
    if not name:
        name = Prompt.ask(
            "Project name", 
            default=project_dir.name,
            show_default=True
        )
    
    if not description:
        description = Prompt.ask(
            "Project description (optional)", 
            default="",
            show_default=False
        )
    
    # Create project structure using plugin
    try:
        selected_plugin.create_project(project_dir, name)
        
        # Create configuration with plugin-specific template
        config_dict = {
            "project_name": name,
            "description": description or None,
            "version": "0.1.0",
        }
        # Merge plugin-specific config
        config_dict.update(selected_plugin.get_config_template())
        
        config = MLCLIConfig(**config_dict)
        save_config(config, config_file)
        
        console.print(f"[green]✓[/green] Initialized [bold]{plugin}[/bold] project: [bold]{name}[/bold]")
        console.print(f"[green]✓[/green] Configuration saved to: {config_file}")
        console.print(f"[green]✓[/green] Project structure created in: {project_dir}")
        
        # Show next steps (plugin-aware)
        if plugin == "chatbot":
            console.print("\n[bold]Next steps:[/bold]")
            console.print("1. Copy [cyan].env.example[/cyan] to [cyan].env[/cyan] and add your OPENAI_API_KEY")
            console.print("2. Add documents to [cyan]data/knowledge_base/[/cyan]")
            console.print("3. Run [cyan]python src/app.py[/cyan] to start the chatbot")
        else:
            console.print("\n[bold]Next steps:[/bold]")
            console.print("1. Add your dataset to the [cyan]data/raw/[/cyan] directory")
            console.print("2. Run [cyan]mlcli preprocess --input data/raw/your_data.csv[/cyan]")
            console.print("3. Run [cyan]mlcli train[/cyan] to start training models")
        
    except Exception as e:
        raise ConfigurationError(f"Failed to initialize project: {e}")


def _create_project_structure(project_dir: Path) -> None:
    """Create the standard ML project directory structure."""
    
    directories = [
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
    
    for dir_path in directories:
        full_path = project_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
    
    # Create essential files
    files_to_create = {
        "README.md": _get_readme_template(),
        ".gitignore": _get_gitignore_template(),
        "requirements.txt": _get_requirements_template(),
        "src/train.py": _get_train_script_template(),
        "src/data_loader.py": _get_data_loader_template(),
        "notebooks/01_exploratory_analysis.ipynb": _get_notebook_template(),
        "tests/test_data_loader.py": _get_test_template(),
    }
    
    for filename, content in files_to_create.items():
        file_path = project_dir / filename
        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")



def _get_readme_template() -> str:
    """Get README.md template."""
    return """# ML Project

This project was initialized with ML Assistant CLI.

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


def _get_gitignore_template() -> str:
    """Get .gitignore template."""
    return """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
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

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# Environment variables
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
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
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

# BentoML
bentofile.yaml
bentos/
"""


def _get_requirements_template() -> str:
    """Get requirements.txt template."""
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
"""


def _get_train_script_template() -> str:
    """Get starter training script template."""
    return '''"""
Starter training script for ML project.
This is a boilerplate - customize it for your specific use case.
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


def _get_data_loader_template() -> str:
    """Get data loader utility template."""
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


def _get_notebook_template() -> str:
    """Get Jupyter notebook template."""
    return '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis\\n",
    "\\n",
    "This notebook provides a starter template for exploring your dataset."
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
    "# Load your dataset\\n",
    "df = pd.read_csv('../data/raw/your_data.csv')\\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Basic Statistics"
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
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualizations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Add your visualizations here\\n",
    "# Example: df['column_name'].hist(bins=30)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''


def _get_test_template() -> str:
    """Get unit test template."""
    return '''"""
Unit tests for data loader.
Run with: pytest tests/
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import DataLoader


def test_data_loader_init():
    """Test DataLoader initialization."""
    loader = DataLoader("data/raw")
    assert loader.data_dir == Path("data/raw")


def test_get_features_and_target():
    """Test feature/target splitting."""
    loader = DataLoader()
    
    # Create sample dataframe
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0]
    })
    
    X, y = loader.get_features_and_target(df, 'target')
    
    assert len(X.columns) == 2
    assert 'target' not in X.columns
    assert len(y) == 3


if __name__ == "__main__":
    pytest.main([__file__])
'''