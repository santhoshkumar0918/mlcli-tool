"""Image Classification plugin for PyTorch-based projects.

Generates projects for computer vision tasks using PyTorch, TorchVision, and PyTorch Lightning.
"""

from typing import Dict, List, Any
from mlcli.plugins.base import PluginBase


class ImageClassificationPlugin(PluginBase):
    """Plugin for image classification projects using PyTorch."""
    
    name = "image-classification"
    description = "Computer Vision with PyTorch & CNNs"
    dependencies = ["torch", "torchvision", "pytorch-lightning", "albumentations"]
    
    def get_directory_structure(self) -> List[str]:
        """Return directory structure for image classification projects."""
        return [
            "data/raw",
            "data/processed",
            "src/models",
            "src/data",
            "src/training",
            "notebooks",
            "tests",
            "logs",
            "checkpoints"
        ]
    
    def get_boilerplate_files(self) -> Dict[str, str]:
        """Return boilerplate files."""
        return {
            "README.md": self._get_readme(),
            ".gitignore": self._get_gitignore(),
            "requirements.txt": self.get_requirements(),
            "src/models/cnn.py": self._get_cnn_model(),
            "src/data/dataset.py": self._get_dataset(),
            "src/training/trainer.py": self._get_trainer(),
            "train.py": self._get_train_script(),
            "notebooks/01_visualize_data.ipynb": self._get_notebook(),
        }
    
    def get_requirements(self) -> str:
        """Return requirements.txt content."""
        return """# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
pytorch-lightning>=2.0.0

# Data Augmentation
albumentations>=1.3.0

# Utilities
numpy>=1.24.0
matplotlib>=3.7.0
rich>=13.0.0
pyyaml>=6.0
"""
    
    def get_config_template(self) -> Dict[str, Any]:
        """Return plugin-specific config."""
        return {
            "plugin": "image-classification",
            "model": {
                "architecture": "resnet18",
                "num_classes": 10,
                "pretrained": True
            },
            "training": {
                "batch_size": 32,
                "epochs": 10,
                "learning_rate": 0.001,
                "accelerator": "auto"  # cpu, gpu, mps
            }
        }
    
    def _get_readme(self) -> str:
        return """# Image Classification Project

Built with MLCLI image-classification plugin.

## Structure
- `src/models/`: CNN architectures (ResNet, etc.)
- `src/data/`: Dataset classes and transforms
- `src/training/`: Training loops (PyTorch Lightning)
- `checkpoints/`: Model weights

## Quick Start
1. Place images in `data/raw/` (organized by folder/class)
2. Run training: `python train.py`
3. Visualize results: `notebooks/01_visualize_data.ipynb`
"""
    
    def _get_gitignore(self) -> str:
        return """__pycache__/
*.py[cod]
.venv/
.env
checkpoints/
logs/
data/raw/
data/processed/
lightning_logs/
*.pth
"""

    def _get_cnn_model(self) -> str:
        return '''"""CNN Model definition."""
import torch
import torch.nn as nn
import torchvision.models as models

class SimpleCNN(nn.Module):
    """Simple CNN wrapper around ResNet."""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        # Use simple ResNet18 for boilerplate
        # In a real app, strict=False or weights=... would be used depending on version
        weights = 'DEFAULT' if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Replace last layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
'''

    def _get_dataset(self) -> str:
        return '''"""Dataset and DataModule."""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import pytorch_lightning as pl
from pathlib import Path

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data/raw", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        # Assumes data/raw folder has subfolders for each class
        # This is a placeholder - usually you'd split train/val
        try:
            self.train_dataset = datasets.ImageFolder(
                root=self.data_dir, 
                transform=self.transform
            )
        except Exception:
            # Fallback for empty init
            print("Warning: No data found in data/raw. Using fake data for demo.")
            self.train_dataset = datasets.FakeData(
                size=100, 
                image_size=(3, 224, 224), 
                num_classes=10, 
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
'''

    def _get_trainer(self) -> str:
        return '''"""PyTorch Lightning Module."""
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from src.models.cnn import SimpleCNN

class ClassificationTask(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleCNN(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
'''

    def _get_train_script(self) -> str:
        return '''"""Main training script."""
import pytorch_lightning as pl
from src.data.dataset import ImageDataModule
from src.training.trainer import ClassificationTask

def main():
    # 1. Setup Data
    dm = ImageDataModule(data_dir="data/raw", batch_size=32)
    
    # 2. Setup Model
    model = ClassificationTask(num_classes=10)
    
    # 3. Trainer
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        default_root_dir="checkpoints"
    )
    
    # 4. Train
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
'''

    def _get_notebook(self) -> str:
        return '''{
 "cells": [
  {
   "cell_type": "markdown",
   "source": ["# Data Visualization"]
  },
  {
   "cell_type": "code",
   "source": ["import matplotlib.pyplot as plt\\nimport torch\\nfrom torchvision.utils import make_grid"]
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
