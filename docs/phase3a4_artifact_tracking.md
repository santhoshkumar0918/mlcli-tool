# Phase 3A.4: Artifact Tracking

**Completed:** February 22, 2026  
**Status:** ✅ COMPLETE

---

## Summary

Implemented a comprehensive artifact tracking system that maintains an audit trail of all MLCLI-generated artifacts with unique IDs, checksums, and lineage tracking.

---

## Problem Statement

ML projects lack reproducibility due to missing tracking:

| Issue | Impact |
|-------|--------|
| No artifact IDs | Can't reference specific outputs |
| No checksums | Can't verify integrity |
| No lineage | Can't trace data→model flow |
| No metadata | Lost context about artifacts |

---

## Solution

### Architecture

```
mlcli/core/versioning/
├── __init__.py           # Public API
├── models.py             # Pydantic models
├── artifact_tracker.py   # Main tracker class
└── checksum.py           # Checksum utilities

.mlcli/
├── artifact_registry.json    # Main registry
└── telemetry/                # (Phase 3A.5)
```

### Files Created

#### 1. `checksum.py` - Checksum Utilities

```python
def compute_file_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute SHA256 checksum of a file."""
    
def compute_dict_checksum(data: dict) -> str:
    """Compute checksum of JSON-serializable dict."""
    
def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    
def format_size(size_bytes: int) -> str:
    """Human-readable size (e.g., '1.5 MB')."""
```

#### 2. `models.py` - Pydantic Models

```python
class ArtifactType(str, Enum):
    DATA_PROFILE = "data_profile"
    PREPROCESSING_PIPELINE = "preprocessing_pipeline"
    MODEL = "model"
    EVALUATION_REPORT = "evaluation_report"
    # ... more types

class ArtifactEntry(BaseModel):
    id: str                           # art-0001
    type: ArtifactType
    status: ArtifactStatus
    path: str                         # Relative path
    filename: str
    created_at: datetime
    checksum: str                     # SHA256
    size_bytes: int
    parent_ids: List[str]             # Lineage
    child_ids: List[str]
    metadata: Dict[str, Any]
    tags: Dict[str, str]

class ArtifactRegistry(BaseModel):
    version: str
    project_name: str
    artifacts: Dict[str, ArtifactEntry]
    
    def generate_id(self) -> str: ...
    def get_lineage(self, artifact_id: str) -> List[ArtifactEntry]: ...
```

#### 3. `artifact_tracker.py` - Main Tracker

```python
class ArtifactTracker:
    """Track and manage all MLCLI-generated artifacts."""
    
    def register(
        self,
        artifact_type: ArtifactType,
        file_path: Path,
        parent_ids: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Register a new artifact, return its ID."""
        
    def get(self, artifact_id: str) -> Optional[ArtifactEntry]:
        """Get artifact by ID."""
        
    def get_lineage(self, artifact_id: str) -> List[ArtifactEntry]:
        """Get full lineage chain."""
        
    def verify_integrity(self, artifact_id: str) -> bool:
        """Verify checksum matches."""
        
    def archive(self, artifact_id: str) -> bool:
        """Mark artifact as archived."""
        
    def search(self, query: str) -> List[ArtifactEntry]:
        """Search by path, filename, or metadata."""
        
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
```

---

## Usage Examples

### Basic Registration

```python
from mlcli.core.versioning import ArtifactTracker, ArtifactType

tracker = ArtifactTracker(project_dir)

# Register preprocessing output
profile_id = tracker.register(
    artifact_type=ArtifactType.DATA_PROFILE,
    file_path="data/processed/data_profile.json",
    metadata={
        "n_samples": 1000,
        "n_features": 20,
        "target_column": "target",
    }
)
print(f"Registered: {profile_id}")  # art-0001

# Register model with lineage
model_id = tracker.register(
    artifact_type=ArtifactType.MODEL,
    file_path="models/best_model.pkl",
    parent_ids=[profile_id],  # Track lineage!
    metadata={
        "algorithm": "xgboost",
        "accuracy": 0.92,
    }
)
print(f"Registered: {model_id}")  # art-0002
```

### Lineage Tracking

```python
# Get full lineage
lineage = tracker.get_lineage(model_id)
for artifact in lineage:
    print(f"{artifact.id}: {artifact.type} -> {artifact.path}")
    
# Output:
# art-0001: data_profile -> data/processed/data_profile.json
# art-0002: model -> models/best_model.pkl

# Export lineage tree
tree = tracker.export_lineage(model_id)
# Returns nested dict for visualization
```

### Integrity Verification

```python
# Verify artifact hasn't been tampered
if tracker.verify_integrity(model_id):
    print("✓ Model integrity verified")
else:
    print("⚠️ Checksum mismatch - file may have been modified")
```

### Search and Query

```python
# Get all models
models = tracker.get_by_type(ArtifactType.MODEL)

# Get latest evaluation
latest_eval = tracker.get_latest(ArtifactType.EVALUATION_REPORT)

# Search artifacts
results = tracker.search("xgboost")

# Get statistics
stats = tracker.get_stats()
# {
#     "total_artifacts": 10,
#     "active_artifacts": 8,
#     "total_size_formatted": "15.2 MB",
#     "by_type": {"model": 3, "data_profile": 2, ...}
# }
```

### Registry File Format

```json
{
  "version": "1.0.0",
  "project_name": "my-ml-project",
  "created_at": "2026-02-22T10:00:00",
  "last_updated": "2026-02-22T11:30:00",
  "artifacts": {
    "art-0001": {
      "id": "art-0001",
      "type": "data_profile",
      "status": "active",
      "path": "data/processed/data_profile.json",
      "filename": "data_profile.json",
      "created_at": "2026-02-22T10:05:00",
      "checksum": "a1b2c3d4e5f6...",
      "size_bytes": 2048,
      "parent_ids": [],
      "child_ids": ["art-0002"],
      "metadata": {"n_samples": 1000, "n_features": 20}
    },
    "art-0002": {
      "id": "art-0002",
      "type": "model",
      "path": "models/best_model.pkl",
      "parent_ids": ["art-0001"],
      "metadata": {"algorithm": "xgboost", "accuracy": 0.92}
    }
  },
  "next_id_counter": 3
}
```

---

## Features

### 1. Unique ID Generation

- Format: `art-XXXX` (e.g., `art-0001`, `art-0002`)
- Sequential, human-readable
- Guaranteed unique within project

### 2. Checksum Integrity

- SHA256 by default
- Detects file tampering
- Verifiable at any time

### 3. Lineage Tracking

- Parent-child relationships
- Full lineage chains
- Export to JSON for visualization

### 4. Metadata Storage

- Type-specific metadata
- Searchable
- Extensible via tags

### 5. Status Management

- ACTIVE - Currently in use
- ARCHIVED - Historical record
- DELETED - Marked for removal

---

## Integration Points

| Command | Artifact Types Registered |
|---------|--------------------------|
| `mlcli preprocess` | DATA_PROFILE, PREPROCESSING_PIPELINE |
| `mlcli train` | MODEL, TRAINING_SUMMARY |
| `mlcli evaluate` | EVALUATION_REPORT |
| `mlcli predict` | PREDICTION (optional) |

---

## Benefits

| Benefit | Description |
|---------|-------------|
| **Reproducibility** | Know exactly which data produced which model |
| **Debugging** | Trace errors back through lineage |
| **Compliance** | Audit trail for regulated industries |
| **Integrity** | Detect file corruption or tampering |
| **Meta-ML** | Context for suggestion engine training |

---

## Next Steps

1. ✅ Artifact Tracking - DONE
2. ⏳ Telemetry Collection (Phase 3A.5)
3. ⏳ Integrate with commands (preprocess, train, evaluate)
4. ⏳ Build CLI commands for artifact inspection

---

## Future Enhancements

- `mlcli artifacts list` - List all tracked artifacts
- `mlcli artifacts show <id>` - Show artifact details
- `mlcli artifacts lineage <id>` - Display lineage tree
- `mlcli artifacts verify` - Verify all checksums
- `mlcli artifacts diff <id1> <id2>` - Compare artifacts

---

**Author:** Senior ML/SDE Team  
**Review Date:** February 22, 2026
