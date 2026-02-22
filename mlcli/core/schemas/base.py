"""Base schema classes for MLCLI artifacts."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class VersionedArtifact(BaseModel):
    """Base class for all versioned MLCLI artifacts.
    
    Provides:
    - Schema versioning for migration support
    - Creation timestamp
    - MLCLI version tracking
    """
    
    schema_version: str = Field(default="1.0.0", description="Schema version for migration support")
    created_at: datetime = Field(default_factory=datetime.now, description="Artifact creation timestamp")
    mlcli_version: Optional[str] = Field(default=None, description="MLCLI version that created this artifact")
    
    model_config = {
        "json_schema_extra": {
            "description": "Base class for all versioned MLCLI artifacts",
            "examples": [{
                "schema_version": "1.0.0",
                "created_at": "2026-02-22T10:30:00",
                "mlcli_version": "0.1.0"
            }]
        },
        "extra": "forbid",
    }


class ArtifactMetadata(BaseModel):
    """Metadata for tracking artifact lineage and provenance."""
    
    artifact_id: Optional[str] = Field(default=None, description="Unique artifact identifier")
    parent_artifacts: list[str] = Field(default_factory=list, description="IDs of parent artifacts")
    checksum: Optional[str] = Field(default=None, description="SHA256 checksum of artifact content")
    tags: Dict[str, str] = Field(default_factory=dict, description="Arbitrary tags for organization")
    
    model_config = {
        "extra": "forbid",
    }
