"""Pydantic models for artifact tracking.

Defines the data structures for the artifact registry and lineage tracking.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ArtifactType(str, Enum):
    """Types of artifacts tracked by MLCLI."""
    
    DATA_PROFILE = "data_profile"
    PREPROCESSING_PIPELINE = "preprocessing_pipeline"
    TRAINING_SUMMARY = "training_summary"
    MODEL = "model"
    EVALUATION_REPORT = "evaluation_report"
    PREDICTION = "prediction"
    CONFIG = "config"
    SUGGESTIONS = "suggestions"


class ArtifactStatus(str, Enum):
    """Status of an artifact."""
    
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ArtifactEntry(BaseModel):
    """Entry for a single tracked artifact."""
    
    id: str = Field(description="Unique artifact identifier (e.g., art-001)")
    type: ArtifactType = Field(description="Type of artifact")
    status: ArtifactStatus = Field(default=ArtifactStatus.ACTIVE, description="Current status")
    
    path: str = Field(description="Relative path from project root")
    filename: str = Field(description="Original filename")
    
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    modified_at: Optional[datetime] = Field(default=None, description="Last modification timestamp")
    
    checksum: str = Field(description="SHA256 checksum")
    checksum_algorithm: str = Field(default="sha256", description="Checksum algorithm used")
    size_bytes: int = Field(ge=0, description="File size in bytes")
    
    parent_ids: List[str] = Field(default_factory=list, description="IDs of parent artifacts")
    child_ids: List[str] = Field(default_factory=list, description="IDs of child artifacts")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Artifact-specific metadata")
    tags: Dict[str, str] = Field(default_factory=dict, description="User-defined tags")
    
    mlcli_version: Optional[str] = Field(default=None, description="MLCLI version that created this")
    schema_version: str = Field(default="1.0.0", description="Schema version of this entry")
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v):
        if not v.startswith('art-'):
            raise ValueError('Artifact ID must start with "art-"')
        return v
    
    @property
    def is_active(self) -> bool:
        return self.status == ArtifactStatus.ACTIVE
    
    @property
    def has_parents(self) -> bool:
        return len(self.parent_ids) > 0
    
    @property
    def has_children(self) -> bool:
        return len(self.child_ids) > 0
    
    model_config = {
        "use_enum_values": True,
        "json_schema_extra": {
            "description": "Entry for a tracked MLCLI artifact",
            "examples": [{
                "id": "art-001",
                "type": "data_profile",
                "path": "data/processed/data_profile.json",
                "filename": "data_profile.json",
                "checksum": "a1b2c3d4e5f6...",
                "size_bytes": 2048,
                "parent_ids": [],
                "metadata": {"n_samples": 1000, "n_features": 20}
            }]
        }
    }


class ArtifactRegistry(BaseModel):
    """Registry tracking all MLCLI artifacts for a project."""
    
    version: str = Field(default="1.0.0", description="Registry schema version")
    project_name: str = Field(description="Name of the project")
    project_id: Optional[str] = Field(default=None, description="Unique project identifier")
    
    created_at: datetime = Field(default_factory=datetime.now, description="Registry creation time")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update time")
    
    artifacts: Dict[str, ArtifactEntry] = Field(
        default_factory=dict,
        description="Map of artifact ID to entry"
    )
    
    next_id_counter: int = Field(default=1, ge=1, description="Counter for generating next ID")
    
    def get_artifact(self, artifact_id: str) -> Optional[ArtifactEntry]:
        return self.artifacts.get(artifact_id)
    
    def get_artifacts_by_type(self, artifact_type: ArtifactType) -> List[ArtifactEntry]:
        return [a for a in self.artifacts.values() if a.type == artifact_type]
    
    def get_latest_artifact(self, artifact_type: ArtifactType) -> Optional[ArtifactEntry]:
        artifacts = self.get_artifacts_by_type(artifact_type)
        if not artifacts:
            return None
        return max(artifacts, key=lambda x: x.created_at)
    
    def get_lineage(self, artifact_id: str) -> List[ArtifactEntry]:
        lineage = []
        current = self.get_artifact(artifact_id)
        while current:
            lineage.append(current)
            if current.parent_ids:
                current = self.get_artifact(current.parent_ids[0])
            else:
                break
        return lineage
    
    def count_by_type(self) -> Dict[str, int]:
        counts = {}
        for artifact in self.artifacts.values():
            type_name = artifact.type if isinstance(artifact.type, str) else artifact.type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts
    
    def generate_id(self) -> str:
        artifact_id = f"art-{self.next_id_counter:04d}"
        self.next_id_counter += 1
        return artifact_id
    
    def update_timestamp(self) -> None:
        self.last_updated = datetime.now()
    
    model_config = {
        "json_schema_extra": {
            "description": "Registry for tracking all MLCLI artifacts",
            "examples": [{
                "version": "1.0.0",
                "project_name": "my-ml-project",
                "artifacts": {
                    "art-0001": {
                        "id": "art-0001",
                        "type": "data_profile",
                        "path": "data/processed/data_profile.json"
                    }
                }
            }]
        }
    }


class LineageNode(BaseModel):
    """Node in an artifact lineage graph."""
    
    artifact: ArtifactEntry
    parents: List['LineageNode'] = Field(default_factory=list)
    children: List['LineageNode'] = Field(default_factory=list)
    
    def to_dict(self, depth: int = 0, max_depth: int = 10) -> Dict[str, Any]:
        if depth >= max_depth:
            return {"id": self.artifact.id, "type": self.artifact.type, "_truncated": True}
        
        return {
            "id": self.artifact.id,
            "type": self.artifact.type,
            "path": self.artifact.path,
            "created_at": self.artifact.created_at.isoformat(),
            "parents": [p.to_dict(depth + 1, max_depth) for p in self.parents],
        }


class ArtifactDiff(BaseModel):
    """Difference between two artifacts."""
    
    artifact_id_1: str
    artifact_id_2: str
    
    same_type: bool
    same_checksum: bool
    same_size: bool
    
    size_diff_bytes: int
    time_diff_seconds: float
    
    metadata_diff: Dict[str, Any] = Field(default_factory=dict)
