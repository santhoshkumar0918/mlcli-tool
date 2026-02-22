"""Meta-ML Engine for MLCLI Suggestion System.

This package implements the recommendation engine that powers `mlcli suggest`.

Components:
- engine.py: Main inference engine
- knowledge_base.py: Synthetic training data generation
- training.py: Model training pipeline
- features.py: Feature extraction from artifacts

Usage:
    from mlcli.meta_ml import SuggestionEngine
    
    engine = SuggestionEngine()
    suggestions = engine.predict(data_profile, evaluation_report)
"""

from .engine import SuggestionEngine
from .knowledge_base import KnowledgeBaseGenerator

__all__ = [
    "SuggestionEngine",
    "KnowledgeBaseGenerator",
]
