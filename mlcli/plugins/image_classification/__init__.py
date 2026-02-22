"""Image Classification plugin for PyTorch-based projects.

Generates projects for computer vision tasks using PyTorch, TorchVision, and PyTorch Lightning.
"""

from typing import Dict, List, Any
from mlcli.plugins.base import PluginBase


class ImageClassificationPlugin(PluginBase):
    """Plugin for image classification projects using PyTorch."""
    
    name = "image-classification"
    description = "Computer Vision with PyTorch & CNNs"
    dependencies = ["torch", "torchvision", "pytorch-lightning"]
    
    def get_directory_structure(self) -> List[str]:
        """Return minimal directory structure for image classification projects."""
        return [
            "data/raw",
            "src",
        ]
    
    def get_boilerplate_files(self) -> Dict[str, str]:
        """Return essential boilerplate files."""
        return {
            "README.md": self._get_readme(),
            ".gitignore": self._get_gitignore(),
            "requirements.txt": self.get_requirements(),
            "src/model.py": self._get_model(),
            "src/data.py": self._get_data(),
            "train.py": self._get_train_script(),
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
pillow>=10.0.0
pyyaml>=6.0
rich>=13.0.0
"""
    
    def get_config_template(self) -> Dict[str, Any]:
        """Return plugin-specific config."""
        return {
            "plugin": "image-classification",
            "model": {
                "architecture": "resnet18",
                "num_classes": 10,
                "pretrained": True,
            },
            "training": {
                "batch_size": 32,
                "epochs": 10,
                "learning_rate": 0.001,
                "accelerator": "auto",
            }
        }
    
    def _get_readme(self) -> str:
        return """# Image Classification Project

Built with **ML Assistant CLI** image-classification plugin.

## Quick Start

```bash
# 1. Organize your images
# Place images in data/raw/ organized by class:
# data/raw/
#   ├── class1/
#   │   ├── img1.jpg
#   │   └── img2.jpg
#   └── class2/
#       ├── img3.jpg
#       └── img4.jpg

# 2. Update mlcli.yaml with your number of classes
# model:
#   num_classes: 2  # Change this to your actual number

# 3. Train
python train.py
```

## Project Structure

```
├── data/
│   └── raw/           # Images organized by class folders
├── src/
│   ├── model.py       # CNN model definition
│   └── data.py        # Data loading utilities
├── train.py           # Training script
└── mlcli.yaml         # Configuration
```

## Configuration

Edit `mlcli.yaml` to customize:
- Model architecture (resnet18, resnet50, efficientnet, etc.)
- Number of classes
- Batch size and learning rate
- Number of epochs

## Supported Architectures

- ResNet (18, 34, 50, 101)
- EfficientNet (b0-b7)
- MobileNet v2/v3
- Custom CNNs

## Next Steps

1. Organize your images in `data/raw/` by class
2. Update `num_classes` in `mlcli.yaml`
3. Run `python train.py`
"""
    
    def _get_gitignore(self) -> str:
        return """# Python
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
.env
.venv
env/
venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# ML artifacts
checkpoints/
lightning_logs/
*.pth
*.pt

# Data
data/processed/
"""
    
    def _get_model(self) -> str:
        return '''"""CNN Model definitions."""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Literal


class ImageClassifier(nn.Module):
    """Flexible image classifier using torchvision backbones."""
    
    ARCHITECTURES = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "efficientnet_b0": models.efficientnet_b0,
        "mobilenet_v2": models.mobilenet_v2,
    }
    
    def __init__(
        self,
        num_classes: int = 10,
        architecture: str = "resnet18",
        pretrained: bool = True,
    ):
        super().__init__()
        
        if architecture not in self.ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        weights = "DEFAULT" if pretrained else None
        self.backbone = self.ARCHITECTURES[architecture](weights=weights)
        
        self._replace_classifier(num_classes, architecture)
    
    def _replace_classifier(self, num_classes: int, architecture: str):
        """Replace the final classification layer."""
        if "resnet" in architecture:
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif "efficientnet" in architecture:
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        elif "mobilenet" in architecture:
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
'''

    def _get_data(self) -> str:
        return '''"""Data loading and augmentation."""
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import pytorch_lightning as pl


class ImageDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for image classification."""
    
    def __init__(
        self,
        data_dir: str = "data/raw",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (224, 224),
        val_split: float = 0.2,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_split = val_split
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def setup(self, stage: Optional[str] = None):
        """Load and split the dataset."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        full_dataset = datasets.ImageFolder(
            root=str(self.data_dir),
            transform=self.transform
        )
        
        self.classes = full_dataset.classes
        self.num_classes = len(self.classes)
        
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.val_dataset.dataset.transform = self.val_transform
        
        print(f"Loaded {len(full_dataset)} images across {self.num_classes} classes")
        print(f"Classes: {self.classes}")
        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
'''

    def _get_train_script(self) -> str:
        return '''"""Main training script."""
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.model import ImageClassifier
from src.data import ImageDataModule


class ClassificationTask(pl.LightningModule):
    """PyTorch Lightning module for image classification."""
    
    def __init__(
        self,
        num_classes: int = 10,
        architecture: str = "resnet18",
        learning_rate: float = 1e-3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = ImageClassifier(
            num_classes=num_classes,
            architecture=architecture,
            pretrained=pretrained,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10
        )
        
        return [optimizer], [scheduler]


def load_config():
    """Load configuration from mlcli.yaml."""
    config_path = Path("mlcli.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def main():
    config = load_config()
    
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    
    dm = ImageDataModule(
        data_dir="data/raw",
        batch_size=training_config.get("batch_size", 32),
    )
    dm.setup()
    
    task = ClassificationTask(
        num_classes=dm.num_classes,
        architecture=model_config.get("architecture", "resnet18"),
        learning_rate=training_config.get("learning_rate", 1e-3),
        pretrained=model_config.get("pretrained", True),
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="model-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
    )
    
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )
    
    trainer = pl.Trainer(
        max_epochs=training_config.get("epochs", 10),
        accelerator=training_config.get("accelerator", "auto"),
        callbacks=[checkpoint_callback, early_stop],
        logger=TensorBoardLogger("lightning_logs", name="image_classifier"),
    )
    
    trainer.fit(task, datamodule=dm)
    
    print(f"\\nBest model: {checkpoint_callback.best_model_path}")
    print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
'''
