"""Base class for MLCLI plugins."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any


class PluginBase(ABC):
    """Abstract base class for all MLCLI plugins.
    
    Plugins define domain-specific project structures and boilerplate code.
    Each plugin must implement methods to generate:
    - Directory structure
    - Boilerplate files (scripts, notebooks, configs)
    - Requirements (dependencies)
    - Configuration templates
    """
    
    # Plugin metadata (must be defined by subclasses)
    name: str
    description: str
    dependencies: List[str]
    
    @abstractmethod
    def get_directory_structure(self) -> List[str]:
        """Return list of directories to create.
        
        Example:
            ["data/raw", "data/processed", "models", "src", "notebooks"]
        """
        pass
    
    @abstractmethod
    def get_boilerplate_files(self) -> Dict[str, str]:
        """Return dict mapping file paths to their content.
        
        Returns:
            Dict[str, str]: {filepath: content}
            
        Example:
            {
                "src/train.py": "# Training script\n...",
                "README.md": "# Project\n..."
            }
        """
        pass
    
    @abstractmethod
    def get_requirements(self) -> str:
        """Return requirements.txt content.
        
        Example:
            "pandas>=2.0.0\nscikit-learn>=1.3.0"
        """
        pass
    
    @abstractmethod
    def get_config_template(self) -> Dict[str, Any]:
        """Return plugin-specific configuration for mlcli.yaml.
        
        Returns:
            Dict containing plugin-specific config options
        """
        pass
    
    def create_project(self, project_dir: Path, project_name: str) -> None:
        """Create full project structure (default implementation).
        
        This method can be overridden for custom project setup logic.
        
        Args:
            project_dir: Directory where project will be created
            project_name: Name of the project
        """
        # Create directories
        for dir_path in self.get_directory_structure():
            full_path = project_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
        
        # Create boilerplate files
        for filepath, content in self.get_boilerplate_files().items():
            file_path = project_dir / filepath
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
