"""Artifact tracking system for MLCLI.

Provides comprehensive tracking of all MLCLI-generated artifacts including:
- Unique ID assignment
- Checksum verification
- Lineage tracking
- Metadata storage
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .models import (
    ArtifactEntry,
    ArtifactRegistry,
    ArtifactType,
    ArtifactStatus,
    LineageNode,
    ArtifactDiff,
)
from .checksum import (
    compute_file_checksum,
    compute_dict_checksum,
    get_file_size,
    format_size,
)


class ArtifactTracker:
    """Track and manage all MLCLI-generated artifacts.
    
    Provides a complete audit trail for reproducibility and debugging.
    
    Example:
        tracker = ArtifactTracker(project_dir)
        
        # Register preprocessing output
        profile_id = tracker.register(
            artifact_type=ArtifactType.DATA_PROFILE,
            file_path="data/processed/data_profile.json",
            metadata={"n_samples": 1000, "n_features": 20}
        )
        
        # Register model with lineage
        model_id = tracker.register(
            artifact_type=ArtifactType.MODEL,
            file_path="models/best_model.pkl",
            parent_ids=[profile_id],
            metadata={"algorithm": "xgboost", "accuracy": 0.92}
        )
        
        # Get lineage
        lineage = tracker.get_lineage(model_id)
    """
    
    REGISTRY_DIR = ".mlcli"
    REGISTRY_FILE = "artifact_registry.json"
    
    def __init__(self, project_dir: Path, project_name: Optional[str] = None):
        """Initialize the artifact tracker.
        
        Args:
            project_dir: Root directory of the MLCLI project
            project_name: Optional project name (defaults to directory name)
        """
        self.project_dir = Path(project_dir)
        self.registry_path = self.project_dir / self.REGISTRY_DIR / self.REGISTRY_FILE
        
        if project_name is None:
            project_name = self.project_dir.name
        
        self._registry = self._load_or_create_registry(project_name)
    
    def _load_or_create_registry(self, project_name: str) -> ArtifactRegistry:
        """Load existing registry or create new one."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                return ArtifactRegistry(**data)
            except Exception as e:
                print(f"Warning: Failed to load registry, creating new: {e}")
        
        return ArtifactRegistry(project_name=project_name)
    
    def _save_registry(self) -> None:
        """Persist registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._registry.update_timestamp()
        
        with open(self.registry_path, 'w') as f:
            json.dump(self._registry.model_dump(mode='json'), f, indent=2, default=str)
    
    @property
    def registry(self) -> ArtifactRegistry:
        """Access the registry."""
        return self._registry
    
    def register(
        self,
        artifact_type: Union[ArtifactType, str],
        file_path: Union[str, Path],
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        mlcli_version: Optional[str] = None,
    ) -> str:
        """Register a new artifact.
        
        Args:
            artifact_type: Type of artifact (enum or string)
            file_path: Path to the artifact file
            parent_ids: IDs of parent artifacts (for lineage)
            metadata: Artifact-specific metadata
            tags: User-defined tags
            mlcli_version: Version of MLCLI that created this
            
        Returns:
            The generated artifact ID
            
        Raises:
            FileNotFoundError: If file_path doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {file_path}")
        
        if isinstance(artifact_type, str):
            artifact_type = ArtifactType(artifact_type)
        
        artifact_id = self._registry.generate_id()
        
        rel_path = file_path.relative_to(self.project_dir) if file_path.is_absolute() else file_path
        
        checksum = compute_file_checksum(file_path)
        size_bytes = get_file_size(file_path)
        
        entry = ArtifactEntry(
            id=artifact_id,
            type=artifact_type,
            status=ArtifactStatus.ACTIVE,
            path=str(rel_path),
            filename=file_path.name,
            checksum=checksum,
            size_bytes=size_bytes,
            parent_ids=parent_ids or [],
            metadata=metadata or {},
            tags=tags or {},
            mlcli_version=mlcli_version,
        )
        
        self._registry.artifacts[artifact_id] = entry
        
        if parent_ids:
            for parent_id in parent_ids:
                parent = self._registry.get_artifact(parent_id)
                if parent and artifact_id not in parent.child_ids:
                    parent.child_ids.append(artifact_id)
        
        self._save_registry()
        
        return artifact_id
    
    def register_dict(
        self,
        artifact_type: Union[ArtifactType, str],
        data: Dict[str, Any],
        file_path: Union[str, Path],
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        mlcli_version: Optional[str] = None,
    ) -> str:
        """Register a dictionary as an artifact.
        
        Writes the dictionary to file_path and registers it.
        
        Args:
            artifact_type: Type of artifact
            data: Dictionary to save
            file_path: Where to save the dictionary
            parent_ids: IDs of parent artifacts
            metadata: Additional metadata
            tags: User-defined tags
            mlcli_version: MLCLI version
            
        Returns:
            The generated artifact ID
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        dict_checksum = compute_dict_checksum(data)
        if metadata is None:
            metadata = {}
        metadata["_dict_checksum"] = dict_checksum
        
        return self.register(
            artifact_type=artifact_type,
            file_path=file_path,
            parent_ids=parent_ids,
            metadata=metadata,
            tags=tags,
            mlcli_version=mlcli_version,
        )
    
    def get(self, artifact_id: str) -> Optional[ArtifactEntry]:
        """Get an artifact by ID.
        
        Args:
            artifact_id: The artifact identifier
            
        Returns:
            ArtifactEntry if found, None otherwise
        """
        return self._registry.get_artifact(artifact_id)
    
    def get_by_type(self, artifact_type: Union[ArtifactType, str]) -> List[ArtifactEntry]:
        """Get all artifacts of a given type.
        
        Args:
            artifact_type: Type to filter by
            
        Returns:
            List of matching artifacts
        """
        if isinstance(artifact_type, str):
            artifact_type = ArtifactType(artifact_type)
        return self._registry.get_artifacts_by_type(artifact_type)
    
    def get_latest(self, artifact_type: Union[ArtifactType, str]) -> Optional[ArtifactEntry]:
        """Get the most recent artifact of a given type.
        
        Args:
            artifact_type: Type to filter by
            
        Returns:
            Most recent artifact or None
        """
        if isinstance(artifact_type, str):
            artifact_type = ArtifactType(artifact_type)
        return self._registry.get_latest_artifact(artifact_type)
    
    def get_lineage(self, artifact_id: str) -> List[ArtifactEntry]:
        """Get the lineage chain for an artifact.
        
        Returns the artifact and all its ancestors, ordered from
        root (oldest) to the specified artifact (newest).
        
        Args:
            artifact_id: ID of the artifact
            
        Returns:
            List of artifacts from root to specified artifact
        """
        lineage = self._registry.get_lineage(artifact_id)
        return list(reversed(lineage))
    
    def get_lineage_tree(self, artifact_id: str) -> Optional[LineageNode]:
        """Build a lineage tree for an artifact.
        
        Args:
            artifact_id: ID of the artifact
            
        Returns:
            LineageNode tree or None if artifact not found
        """
        artifact = self.get(artifact_id)
        if not artifact:
            return None
        
        def build_node(entry: ArtifactEntry) -> LineageNode:
            node = LineageNode(artifact=entry)
            for parent_id in entry.parent_ids:
                parent = self.get(parent_id)
                if parent:
                    parent_node = build_node(parent)
                    node.parents.append(parent_node)
                    parent_node.children.append(node)
            return node
        
        return build_node(artifact)
    
    def verify_integrity(self, artifact_id: str) -> bool:
        """Verify artifact integrity by checking checksum.
        
        Args:
            artifact_id: ID of the artifact to verify
            
        Returns:
            True if checksum matches, False otherwise
        """
        artifact = self.get(artifact_id)
        if not artifact:
            return False
        
        file_path = self.project_dir / artifact.path
        if not file_path.exists():
            return False
        
        current_checksum = compute_file_checksum(file_path)
        return current_checksum == artifact.checksum
    
    def archive(self, artifact_id: str) -> bool:
        """Archive an artifact (mark as inactive).
        
        Args:
            artifact_id: ID of the artifact to archive
            
        Returns:
            True if successful, False if not found
        """
        artifact = self.get(artifact_id)
        if not artifact:
            return False
        
        artifact.status = ArtifactStatus.ARCHIVED
        artifact.modified_at = datetime.now()
        self._save_registry()
        return True
    
    def restore(self, artifact_id: str) -> bool:
        """Restore an archived artifact.
        
        Args:
            artifact_id: ID of the artifact to restore
            
        Returns:
            True if successful, False if not found
        """
        artifact = self.get(artifact_id)
        if not artifact:
            return False
        
        artifact.status = ArtifactStatus.ACTIVE
        artifact.modified_at = datetime.now()
        self._save_registry()
        return True
    
    def add_tag(self, artifact_id: str, key: str, value: str) -> bool:
        """Add a tag to an artifact.
        
        Args:
            artifact_id: ID of the artifact
            key: Tag key
            value: Tag value
            
        Returns:
            True if successful, False if not found
        """
        artifact = self.get(artifact_id)
        if not artifact:
            return False
        
        artifact.tags[key] = value
        artifact.modified_at = datetime.now()
        self._save_registry()
        return True
    
    def update_metadata(self, artifact_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for an artifact.
        
        Args:
            artifact_id: ID of the artifact
            metadata: New metadata to merge
            
        Returns:
            True if successful, False if not found
        """
        artifact = self.get(artifact_id)
        if not artifact:
            return False
        
        artifact.metadata.update(metadata)
        artifact.modified_at = datetime.now()
        self._save_registry()
        return True
    
    def diff(self, artifact_id_1: str, artifact_id_2: str) -> Optional[ArtifactDiff]:
        """Compare two artifacts.
        
        Args:
            artifact_id_1: First artifact ID
            artifact_id_2: Second artifact ID
            
        Returns:
            ArtifactDiff if both found, None otherwise
        """
        a1 = self.get(artifact_id_1)
        a2 = self.get(artifact_id_2)
        
        if not a1 or not a2:
            return None
        
        return ArtifactDiff(
            artifact_id_1=artifact_id_1,
            artifact_id_2=artifact_id_2,
            same_type=a1.type == a2.type,
            same_checksum=a1.checksum == a2.checksum,
            same_size=a1.size_bytes == a2.size_bytes,
            size_diff_bytes=abs(a1.size_bytes - a2.size_bytes),
            time_diff_seconds=abs((a1.created_at - a2.created_at).total_seconds()),
            metadata_diff={
                k: {"a1": a1.metadata.get(k), "a2": a2.metadata.get(k)}
                for k in set(a1.metadata.keys()) | set(a2.metadata.keys())
                if a1.metadata.get(k) != a2.metadata.get(k)
            }
        )
    
    def count(self) -> Dict[str, int]:
        """Get count of artifacts by type.
        
        Returns:
            Dictionary mapping type name to count
        """
        return self._registry.count_by_type()
    
    def list_all(self, status: Optional[ArtifactStatus] = None) -> List[ArtifactEntry]:
        """List all artifacts, optionally filtered by status.
        
        Args:
            status: Optional status filter
            
        Returns:
            List of artifacts
        """
        artifacts = list(self._registry.artifacts.values())
        if status:
            artifacts = [a for a in artifacts if a.status == status]
        return sorted(artifacts, key=lambda x: x.created_at, reverse=True)
    
    def search(self, query: str) -> List[ArtifactEntry]:
        """Search artifacts by path, filename, or metadata.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching artifacts
        """
        query = query.lower()
        results = []
        
        for artifact in self._registry.artifacts.values():
            if query in artifact.path.lower():
                results.append(artifact)
                continue
            if query in artifact.filename.lower():
                results.append(artifact)
                continue
            for v in artifact.metadata.values():
                if query in str(v).lower():
                    results.append(artifact)
                    break
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the registry.
        
        Returns:
            Dictionary with registry statistics
        """
        artifacts = list(self._registry.artifacts.values())
        
        total_size = sum(a.size_bytes for a in artifacts)
        
        return {
            "total_artifacts": len(artifacts),
            "active_artifacts": sum(1 for a in artifacts if a.status == ArtifactStatus.ACTIVE),
            "archived_artifacts": sum(1 for a in artifacts if a.status == ArtifactStatus.ARCHIVED),
            "total_size_bytes": total_size,
            "total_size_formatted": format_size(total_size),
            "by_type": self.count(),
            "registry_created": self._registry.created_at.isoformat(),
            "last_updated": self._registry.last_updated.isoformat(),
        }
    
    def export_lineage(self, artifact_id: str) -> Dict[str, Any]:
        """Export lineage as a JSON-serializable dictionary.
        
        Args:
            artifact_id: ID of the artifact
            
        Returns:
            Lineage tree as dictionary
        """
        tree = self.get_lineage_tree(artifact_id)
        if tree:
            return tree.to_dict()
        return {}
