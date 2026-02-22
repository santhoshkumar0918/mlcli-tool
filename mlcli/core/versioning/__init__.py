"""Artifact tracking and versioning for MLCLI.

This package provides comprehensive artifact tracking including:
- Unique ID assignment for all artifacts
- Checksum-based integrity verification
- Lineage tracking (parent-child relationships)
- Metadata storage
- Search and query capabilities

Usage:
    from mlcli.core.versioning import ArtifactTracker, ArtifactType
    
    # Initialize tracker
    tracker = ArtifactTracker(project_dir)
    
    # Register artifacts
    profile_id = tracker.register(
        artifact_type=ArtifactType.DATA_PROFILE,
        file_path="data/processed/data_profile.json",
        metadata={"n_samples": 1000}
    )
    
    model_id = tracker.register(
        artifact_type=ArtifactType.MODEL,
        file_path="models/best_model.pkl",
        parent_ids=[profile_id],
        metadata={"algorithm": "xgboost"}
    )
    
    # Query lineage
    lineage = tracker.get_lineage(model_id)
    for artifact in lineage:
        print(f"{artifact.id}: {artifact.type} -> {artifact.path}")
    
    # Verify integrity
    if tracker.verify_integrity(model_id):
        print("Model integrity verified!")
"""

from .models import (
    ArtifactEntry,
    ArtifactRegistry,
    ArtifactType,
    ArtifactStatus,
    LineageNode,
    ArtifactDiff,
)
from .artifact_tracker import ArtifactTracker
from .checksum import (
    compute_file_checksum,
    compute_dict_checksum,
    compute_string_checksum,
    verify_file_checksum,
    get_file_size,
    format_size,
)


__all__ = [
    "ArtifactTracker",
    "ArtifactEntry",
    "ArtifactRegistry",
    "ArtifactType",
    "ArtifactStatus",
    "LineageNode",
    "ArtifactDiff",
    "compute_file_checksum",
    "compute_dict_checksum",
    "compute_string_checksum",
    "verify_file_checksum",
    "get_file_size",
    "format_size",
]
