"""Core ML Assistant CLI modules.

This package contains the core functionality:
- config: Configuration management
- data: Data processing and preprocessing
- models: ML model training and evaluation
- exceptions: Custom exceptions
- schemas: Pydantic schemas for artifact validation
- versioning: Artifact tracking and lineage
- telemetry: Privacy-first telemetry collection
- suggestion_model: ML-powered suggestion engine
"""

from .config import MLCLIConfig, load_config_file, save_config
from .exceptions import (
    MLCLIError,
    ConfigurationError,
    DataError,
    ModelError,
    ValidationError,
)

__all__ = [
    "MLCLIConfig",
    "load_config_file",
    "save_config",
    "MLCLIError",
    "ConfigurationError",
    "DataError",
    "ModelError",
    "ValidationError",
]
