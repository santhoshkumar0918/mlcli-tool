# Phase 3: AI-Powered Suggestions — Production-Grade Implementation Plan

**Author:** Senior ML/SDE Architect  
**Date:** February 19, 2026  
**Status:** ACTIVE DEVELOPMENT PLAN

---

## 🎯 Executive Summary

We will upgrade `mlcli suggest` from rule-based heuristics to a **Meta-ML Recommendation Engine** that learns from ML project telemetry. However, **we must fix critical infrastructure gaps first** to ensure scalability, reproducibility, and production-readiness.

**Key Insight:** Building a meta-ML model without data infrastructure is like training a model without validation data—it will fail in production.

---

## 🔍 Current State Analysis (Critical Gaps Identified)

### ✅ What Works
- Rule-based suggestions generate reasonable recommendations
- CLI workflow is functional end-to-end (local)
- Rich console output provides good UX

### ❌ Critical Gaps Blocking Scale

| Gap | Impact | Priority |
|-----|--------|----------|
| **No preprocessing pipeline persistence** | Predictions use wrong transforms | P0 - BLOCKER |
| **No schema validation** | JSON artifacts break silently | P0 - BLOCKER |
| **No model versioning** | Can't track which model generated which results | P0 - BLOCKER |
| **No telemetry/feedback loop** | Can't learn from user actions | P0 - BLOCKER |
| **No experiment tracking** | Can't compare model improvements | P1 - CRITICAL |
| **Hardcoded file paths** | Breaks in non-standard layouts | P1 - CRITICAL |
| **No testing for core engines** | Data/model code untested | P1 - CRITICAL |
| **No artifact versioning** | Breaking changes to JSON schema | P2 - HIGH |

**Bottom Line:** We need to build the **data platform** before building the **meta-ML brain**.

---

## 🏗️ Architecture: The Complete System

```
┌─────────────────────────────────────────────────────────────────┐
│                     MLCLI PROJECT WORKSPACE                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ARTIFACT GENERATION LAYER                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ preprocess   │  │   train      │  │  evaluate    │         │
│  │─────────────►│  │─────────────►│  │─────────────►│         │
│  │data_profile  │  │training_     │  │evaluation_   │         │
│  │  .json       │  │summary.json  │  │report.json   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                     │
│              ┌──────────────────────────┐                       │
│              │  SCHEMA VALIDATOR        │ (Pydantic V2)        │
│              │  - Version compatibility │                       │
│              │  - Type checking         │                       │
│              │  - Migration support     │                       │
│              └──────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING LAYER                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ArtifactFeatureExtractor                                │  │
│  │  - Flattens nested JSON → numeric vectors               │  │
│  │  - Handles missing fields gracefully                     │  │
│  │  - Versioned feature schemas                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│              ┌──────────────────────────┐                       │
│              │  Feature Vector (v1.0)   │                       │
│              │  [48 dimensions]         │                       │
│              │  - Data Quality (12)     │                       │
│              │  - Model Perf (16)       │                       │
│              │  - Training Meta (8)     │                       │
│              │  - Resource (12)         │                       │
│              └──────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      META-ML ENGINE CORE                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  SuggestionEngine (Multi-Label Classifier)               │  │
│  │                                                           │  │
│  │  Model: XGBoost with Platt Scaling                       │  │
│  │  Labels (15 classes):                                    │  │
│  │    0: COLLECT_MORE_DATA                                  │  │
│  │    1: HANDLE_CLASS_IMBALANCE                             │  │
│  │    2: FEATURE_ENGINEERING                                │  │
│  │    3: DIMENSIONALITY_REDUCTION                           │  │
│  │    4: HYPERPARAMETER_TUNING                              │  │
│  │    5: TRY_ENSEMBLE_MODELS                                │  │
│  │    6: REGULARIZATION_INCREASE                            │  │
│  │    7: DATA_AUGMENTATION                                  │  │
│  │    8: OUTLIER_TREATMENT                                  │  │
│  │    9: FEATURE_SCALING_CHANGE                             │  │
│  │   10: CROSS_VALIDATION_INCREASE                          │  │
│  │   11: EARLY_STOPPING                                     │  │
│  │   12: LEARNING_RATE_TUNING                               │  │
│  │   13: BATCH_SIZE_OPTIMIZATION                            │  │
│  │   14: MODEL_ARCHITECTURE_CHANGE                          │  │
│  │                                                           │  │
│  │  Output: [(label, confidence, action_code)]             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│              ┌──────────────────────────┐                       │
│              │  KNOWLEDGE BASE          │                       │
│              │  data/meta_ml/           │                       │
│              │  - synthetic_kb.jsonl    │                       │
│              │  - real_telemetry.jsonl  │                       │
│              │  - model_v1.pkl          │                       │
│              │  - feature_schema.json   │                       │
│              └──────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TELEMETRY & FEEDBACK LAYER                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  TelemetryCollector                                      │  │
│  │  - Captures user actions (suggestion → command run)      │  │
│  │  - Anonymous project fingerprint                         │  │
│  │  - Outcome tracking (accuracy delta)                     │  │
│  │  - Local-first, opt-in cloud sync                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│              ┌──────────────────────────┐                       │
│              │  .mlcli/telemetry/       │                       │
│              │  - events.jsonl          │                       │
│              │  - outcomes.jsonl        │                       │
│              └──────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONTINUOUS LEARNING                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ModelRetrainer (Runs monthly via GitHub Actions)       │  │
│  │  - Aggregates telemetry from all users                  │  │
│  │  - Trains new model version                             │  │
│  │  - Validates on holdout set                             │  │
│  │  - Deploys if metrics improve                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Phased Implementation Roadmap

### **Phase 3A: Infrastructure Foundation** (Week 1-2)
**Goal:** Fix critical gaps that block scalability and meta-ML development.

#### 3A.1: Schema Validation & Artifact Management
**Files to create:**
- `mlcli/core/schemas/` (new package)
  - `__init__.py`
  - `data_profile.py` — Pydantic model for data_profile.json
  - `training_summary.py` — Pydantic model for training_summary.json
  - `evaluation_report.py` — Pydantic model for evaluation_report.json
  - `base.py` — Base schema with versioning support

**Implementation:**
```python
# mlcli/core/schemas/base.py
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime

class VersionedArtifact(BaseModel):
    schema_version: str = Field(default="1.0.0")
    created_at: datetime = Field(default_factory=datetime.now)
    mlcli_version: str
    
    class Config:
        json_schema_extra = {
            "description": "Base class for all versioned MLCLI artifacts"
        }

# mlcli/core/schemas/data_profile.py
class DataProfileSchema(VersionedArtifact):
    # Data Shape
    n_samples: int = Field(gt=0)
    n_features: int = Field(gt=0)
    target_column: str
    
    # Data Quality Metrics
    missing_value_ratio: float = Field(ge=0.0, le=1.0)
    duplicate_ratio: float = Field(ge=0.0, le=1.0)
    
    # Feature Distribution
    numeric_features: list[str]
    categorical_features: list[str]
    high_cardinality_features: list[str]
    
    # Class Balance (classification only)
    class_distribution: dict[str, int] | None = None
    imbalance_ratio: float | None = Field(default=None, ge=1.0)
    
    # Resource Metrics
    memory_usage_mb: float
    preprocessing_time_sec: float
```

**Benefits:**
- Type-safe artifact loading
- Automatic validation
- Schema migration support
- Clear API contracts

---

#### 3A.2: Preprocessing Pipeline Persistence
**Problem:** Currently, we fit transformers during preprocessing but DON'T save them. This means predictions use raw data → WRONG!

**Files to modify:**
- `mlcli/core/data.py` — Add pipeline save/load
- `mlcli/commands/preprocess_cmd.py` — Save pipeline
- `mlcli/commands/predict_cmd.py` — Load and apply pipeline

**Implementation:**
```python
# mlcli/core/data.py
import joblib

class DataProcessor:
    def save_pipeline(self, output_dir: Path) -> None:
        """Save fitted preprocessing pipeline."""
        pipeline_path = output_dir / "preprocessing_pipeline.pkl"
        artifact = {
            "preprocessor": self.preprocessor,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "config": self.config.dict(),
            "version": "1.0.0"
        }
        joblib.dump(artifact, pipeline_path)
        logger.info(f"Pipeline saved: {pipeline_path}")
    
    @classmethod
    def load_pipeline(cls, pipeline_path: Path) -> 'DataProcessor':
        """Load fitted preprocessing pipeline."""
        artifact = joblib.load(pipeline_path)
        processor = cls(DataConfig(**artifact["config"]))
        processor.preprocessor = artifact["preprocessor"]
        processor.label_encoder = artifact["label_encoder"]
        processor.feature_names = artifact["feature_names"]
        processor.target_name = artifact["target_name"]
        return processor
```

**Impact:** Ensures predictions use EXACT same transformations as training. Critical for production.

---

#### 3A.3: Model & Artifact Versioning
**Files to create:**
- `mlcli/core/versioning/` (new package)
  - `__init__.py`
  - `artifact_tracker.py` — Track all artifacts with UUIDs
  - `model_registry.py` — Local model registry

**Implementation:**
```python
# mlcli/core/versioning/artifact_tracker.py
from pathlib import Path
import json
import hashlib
from datetime import datetime
from typing import Dict, Any

class ArtifactTracker:
    """Track all artifacts generated during ML workflow."""
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.registry_path = project_dir / ".mlcli" / "artifact_registry.json"
        self.registry = self._load_registry()
    
    def register_artifact(
        self, 
        artifact_type: str,  # "data_profile", "model", "evaluation"
        file_path: Path,
        metadata: Dict[str, Any]
    ) -> str:
        """Register an artifact and return its ID."""
        artifact_id = self._generate_id(file_path)
        
        entry = {
            "id": artifact_id,
            "type": artifact_type,
            "path": str(file_path.relative_to(self.project_dir)),
            "created_at": datetime.now().isoformat(),
            "checksum": self._compute_hash(file_path),
            "metadata": metadata,
        }
        
        self.registry["artifacts"][artifact_id] = entry
        self._save_registry()
        return artifact_id
    
    def get_lineage(self, artifact_id: str) -> Dict[str, Any]:
        """Get the full lineage of an artifact."""
        # Return all artifacts that led to this one
        pass
```

**Benefits:**
- Reproducibility: Know exactly which data/config produced which model
- Debugging: Trace errors back to source artifacts
- Compliance: Audit trail for model decisions

---

#### 3A.4: Telemetry Collection System
**Files to create:**
- `mlcli/core/telemetry/` (new package)
  - `__init__.py`
  - `collector.py` — Event collection
  - `privacy.py` — PII filtering

**Implementation:**
```python
# mlcli/core/telemetry/collector.py
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, Optional

class TelemetryCollector:
    """Privacy-first telemetry collection."""
    
    def __init__(self, project_dir: Path, enabled: bool = True):
        self.enabled = enabled
        self.telemetry_dir = project_dir / ".mlcli" / "telemetry"
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.telemetry_dir / "events.jsonl"
    
    def log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        user_action: Optional[str] = None
    ) -> None:
        """Log an event with automatic PII filtering."""
        if not self.enabled:
            return
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": self._sanitize(data),
            "user_action": user_action,
        }
        
        with open(self.events_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    def log_suggestion_shown(
        self,
        suggestions: list[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> str:
        """Log when suggestions are shown, return session ID."""
        session_id = self._generate_session_id()
        self.log_event(
            "suggestions_shown",
            {
                "session_id": session_id,
                "suggestions": suggestions,
                "context": context,
            }
        )
        return session_id
    
    def log_suggestion_acted_on(
        self,
        session_id: str,
        suggestion_index: int,
        action: str
    ) -> None:
        """Log when user acts on a suggestion."""
        self.log_event(
            "suggestion_action",
            {
                "session_id": session_id,
                "suggestion_index": suggestion_index,
                "action": action,
            },
            user_action=action
        )
```

**Usage in suggest command:**
```python
# In mlcli/commands/suggest_cmd.py
telemetry = TelemetryCollector(project_dir)
session_id = telemetry.log_suggestion_shown(suggestions, context)

# Store session_id in a hidden file for next command to pick up
(project_dir / ".mlcli" / "last_suggestion_session").write_text(session_id)
```

---

### **Phase 3B: Knowledge Base Engineering** (Week 3)
**Goal:** Create a robust synthetic dataset for training the meta-ML model.

#### 3B.1: Synthetic Data Generator
**Files to create:**
- `data/meta_ml/` (new directory)
- `mlcli/meta_ml/` (new package)
  - `__init__.py`
  - `knowledge_base.py` — KB generator
  - `simulation.py` — Simulate ML scenarios

**Strategy:** Generate 10,000+ synthetic ML project scenarios using:
1. **Literature-based patterns** (Andrew Ng's heuristics, papers)
2. **Parametric simulation** (vary n_samples, imbalance, etc.)
3. **Expert rules** (codified from your team's experience)

**Implementation:**
```python
# mlcli/meta_ml/knowledge_base.py
import numpy as np
import pandas as pd
from typing import List, Dict, Any

class KnowledgeBaseGenerator:
    """Generate synthetic ML project scenarios with expert labels."""
    
    SUGGESTION_LABELS = [
        "COLLECT_MORE_DATA",
        "HANDLE_CLASS_IMBALANCE",
        "FEATURE_ENGINEERING",
        "DIMENSIONALITY_REDUCTION",
        "HYPERPARAMETER_TUNING",
        "TRY_ENSEMBLE_MODELS",
        "REGULARIZATION_INCREASE",
        "DATA_AUGMENTATION",
        "OUTLIER_TREATMENT",
        "FEATURE_SCALING_CHANGE",
        "CROSS_VALIDATION_INCREASE",
        "EARLY_STOPPING",
        "LEARNING_RATE_TUNING",
        "BATCH_SIZE_OPTIMIZATION",
        "MODEL_ARCHITECTURE_CHANGE",
    ]
    
    def generate_scenario(self) -> Dict[str, Any]:
        """Generate one synthetic ML project scenario."""
        # Data characteristics
        n_samples = np.random.choice([10, 50, 100, 500, 1000, 5000, 10000])
        n_features = np.random.choice([5, 10, 20, 50, 100, 200])
        imbalance_ratio = np.random.choice([1.0, 1.5, 3.0, 5.0, 10.0, 20.0])
        missing_ratio = np.random.uniform(0.0, 0.5)
        
        # Model performance
        accuracy = self._simulate_accuracy(n_samples, n_features, imbalance_ratio)
        precision = accuracy + np.random.uniform(-0.1, 0.1)
        recall = accuracy + np.random.uniform(-0.15, 0.05)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        # Generate expert labels
        labels = self._apply_expert_rules(
            n_samples=n_samples,
            n_features=n_features,
            imbalance_ratio=imbalance_ratio,
            missing_ratio=missing_ratio,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
        )
        
        return {
            "features": {
                "n_samples": n_samples,
                "n_features": n_features,
                "imbalance_ratio": imbalance_ratio,
                "missing_ratio": missing_ratio,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "precision_recall_gap": abs(precision - recall),
            },
            "labels": labels,
        }
    
    def _apply_expert_rules(self, **metrics) -> List[str]:
        """Apply expert heuristics to generate suggestion labels."""
        suggestions = []
        
        # Rule 1: Small dataset
        if metrics["n_samples"] < 100:
            suggestions.append("COLLECT_MORE_DATA")
        
        # Rule 2: Class imbalance
        if metrics["imbalance_ratio"] > 3.0:
            suggestions.append("HANDLE_CLASS_IMBALANCE")
        
        # Rule 3: High dimensions
        if metrics["n_features"] > metrics["n_samples"]:
            suggestions.append("DIMENSIONALITY_REDUCTION")
        
        # Rule 4: Low accuracy with reasonable data
        if metrics["accuracy"] < 0.7 and metrics["n_samples"] > 500:
            suggestions.extend(["FEATURE_ENGINEERING", "TRY_ENSEMBLE_MODELS"])
        
        # Rule 5: Precision-recall gap (imbalance indicator)
        if metrics["precision_recall_gap"] > 0.15:
            suggestions.append("HANDLE_CLASS_IMBALANCE")
        
        # Rule 6: Moderate performance, needs tuning
        if 0.7 <= metrics["accuracy"] < 0.85:
            suggestions.append("HYPERPARAMETER_TUNING")
        
        # Rule 7: High missing values
        if metrics["missing_ratio"] > 0.3:
            suggestions.append("DATA_AUGMENTATION")
        
        return suggestions
    
    def generate_knowledge_base(self, n_scenarios: int = 10000) -> pd.DataFrame:
        """Generate complete knowledge base."""
        scenarios = [self.generate_scenario() for _ in range(n_scenarios)]
        return pd.DataFrame(scenarios)
```

**Validation Strategy:**
1. Generate 10,000 scenarios
2. Manually review 100 random samples
3. Ensure label distribution makes sense
4. Test edge cases (tiny datasets, perfect accuracy, etc.)

---

#### 3B.2: Feature Engineering Pipeline
**Files to create:**
- `mlcli/meta_ml/features.py` — Feature extraction

**Implementation:**
```python
# mlcli/meta_ml/features.py
import pandas as pd
import numpy as np
from typing import Dict, Any
from pathlib import Path

class ArtifactFeatureExtractor:
    """Extract ML-ready features from JSON artifacts."""
    
    FEATURE_VERSION = "1.0.0"
    
    def extract_from_artifacts(
        self,
        data_profile_path: Path,
        training_summary_path: Path,
        evaluation_report_path: Path
    ) -> np.ndarray:
        """Extract feature vector from all artifacts."""
        
        features = {}
        
        # Load artifacts (with validation)
        data_profile = self._load_json(data_profile_path)
        training_summary = self._load_json(training_summary_path)
        evaluation = self._load_json(evaluation_report_path)
        
        # Data Quality Features (12)
        features.update(self._extract_data_features(data_profile))
        
        # Model Performance Features (16)
        features.update(self._extract_model_features(evaluation))
        
        # Training Meta Features (8)
        features.update(self._extract_training_features(training_summary))
        
        # Resource Features (12)
        features.update(self._extract_resource_features(data_profile, training_summary))
        
        # Convert to ordered array
        return self._dict_to_array(features)
    
    def _extract_data_features(self, data_profile: Dict) -> Dict[str, float]:
        """Extract data quality features."""
        return {
            "n_samples": float(data_profile.get("original_shape", [0, 0])[0]),
            "n_features": float(data_profile.get("original_shape", [0, 0])[1]),
            "n_features_processed": float(data_profile.get("processed_shape", [0, 0])[1]),
            "missing_ratio": self._safe_divide(
                sum(data_profile.get("missing_values", {}).values()),
                data_profile.get("original_shape", [1, 1])[0]
            ),
            "duplicate_ratio": data_profile.get("duplicates", 0) / max(1, data_profile.get("original_shape", [1])[0]),
            "n_categorical": len(data_profile.get("categorical_columns", [])),
            "n_numeric": len(data_profile.get("numeric_columns", [])),
            "high_cardinality_ratio": len(data_profile.get("quality_issues", {}).get("high_cardinality_columns", [])) / max(1, len(data_profile.get("categorical_columns", []))),
            "constant_features": len(data_profile.get("quality_issues", {}).get("constant_columns", [])),
            "feature_explosion_ratio": self._safe_divide(
                data_profile.get("processed_shape", [0, 0])[1],
                data_profile.get("original_shape", [0, 1])[1]
            ),
            "samples_to_features_ratio": self._safe_divide(
                data_profile.get("original_shape", [0, 0])[0],
                data_profile.get("original_shape", [0, 1])[1]
            ),
            "memory_mb": data_profile.get("memory_usage", 0) / 1024 / 1024,
        }
    
    def _extract_model_features(self, evaluation: Dict) -> Dict[str, float]:
        """Extract model performance features."""
        metrics = evaluation.get("metrics", {})
        task_type = evaluation.get("task_type", "classification")
        
        if task_type == "classification":
            return {
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1_score": metrics.get("f1_score", 0.0),
                "roc_auc": metrics.get("roc_auc", 0.5),
                "precision_recall_gap": abs(metrics.get("precision", 0) - metrics.get("recall", 0)),
                "accuracy_f1_gap": abs(metrics.get("accuracy", 0) - metrics.get("f1_score", 0)),
                # ... more features
            }
        else:  # regression
            return {
                "r2_score": metrics.get("r2_score", 0.0),
                "mse": metrics.get("mean_squared_error", 0.0),
                "rmse": metrics.get("root_mean_squared_error", 0.0),
                "mae": metrics.get("mean_absolute_error", 0.0),
                "mape": metrics.get("mean_absolute_percentage_error", 100.0),
                # ... more features
            }
    
    @staticmethod
    def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
        """Safe division with default."""
        return a / b if b != 0 else default
```

---

### **Phase 3C: Meta-ML Engine** (Week 4-5)
**Goal:** Build and train the recommendation model.

#### 3C.1: Model Training Pipeline
**Files to create:**
- `mlcli/meta_ml/engine.py` — Core engine
- `mlcli/meta_ml/models.py` — Model definitions
- `mlcli/meta_ml/training.py` — Training loop

**Implementation:**
```python
# mlcli/meta_ml/engine.py
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
import joblib
from typing import List, Tuple, Dict

class MetaMLEngine:
    """Meta-ML recommendation engine."""
    
    def __init__(self, model_path: Path = None):
        self.model = None
        self.feature_extractor = ArtifactFeatureExtractor()
        if model_path and model_path.exists():
            self.load_model(model_path)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,  # Multi-label binary matrix
        calibrate: bool = True
    ) -> Dict[str, float]:
        """Train the meta-ML model."""
        
        # Base model: XGBoost
        base_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Multi-label wrapper
        multi_label_model = MultiOutputClassifier(base_model)
        
        # Train
        multi_label_model.fit(X, y)
        
        # Calibrate probabilities (CRITICAL for confidence scores)
        if calibrate:
            self.model = CalibratedClassifierCV(
                multi_label_model,
                method='isotonic',
                cv=5
            )
            self.model.fit(X, y)
        else:
            self.model = multi_label_model
        
        # Evaluate
        train_score = self.model.score(X, y)
        
        return {
            "train_score": train_score,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_labels": y.shape[1]
        }
    
    def predict_suggestions(
        self,
        data_profile_path: Path,
        training_summary_path: Path,
        evaluation_report_path: Path,
        top_k: int = 3,
        confidence_threshold: float = 0.3
    ) -> List[Tuple[str, float, str]]:
        """Generate top-k suggestions with confidence scores."""
        
        # Extract features
        features = self.feature_extractor.extract_from_artifacts(
            data_profile_path,
            training_summary_path,
            evaluation_report_path
        )
        
        # Predict probabilities
        probabilities = self.model.predict_proba(features.reshape(1, -1))[0]
        
        # Get top-k suggestions
        suggestions = []
        for idx, prob in enumerate(probabilities):
            if prob >= confidence_threshold:
                label = self.feature_extractor.SUGGESTION_LABELS[idx]
                action_code = self._generate_action_code(label)
                suggestions.append((label, prob, action_code))
        
        # Sort by confidence
        suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)[:top_k]
        
        return suggestions
    
    def _generate_action_code(self, label: str) -> str:
        """Generate actionable command snippet for a suggestion."""
        action_map = {
            "COLLECT_MORE_DATA": "# Add more samples to data/raw/ and re-run mlcli preprocess",
            "HANDLE_CLASS_IMBALANCE": "# Update mlcli.yaml:\ndata:\n  imbalance_strategy: smote",
            "FEATURE_ENGINEERING": "# Run: mlcli preprocess --feature-engineering advanced",
            "HYPERPARAMETER_TUNING": "# Update mlcli.yaml:\nmodel:\n  hyperparameter_tuning: true\n  cv_folds: 10",
            "TRY_ENSEMBLE_MODELS": "# Update mlcli.yaml:\nmodel:\n  algorithms: [xgboost, random_forest, gradient_boosting]",
            # ... more mappings
        }
        return action_map.get(label, "# No specific action available")
    
    def save_model(self, output_path: Path) -> None:
        """Save trained model."""
        artifact = {
            "model": self.model,
            "feature_version": self.feature_extractor.FEATURE_VERSION,
            "suggestion_labels": self.feature_extractor.SUGGESTION_LABELS,
        }
        joblib.dump(artifact, output_path)
    
    def load_model(self, model_path: Path) -> None:
        """Load trained model."""
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
```

---

#### 3C.2: Model Training Script
**Files to create:**
- `scripts/train_meta_ml.py` — Training script

```python
# scripts/train_meta_ml.py
"""
Train the meta-ML recommendation engine.
Run this script to (re)train the suggestion model.
"""
from pathlib import Path
from mlcli.meta_ml.knowledge_base import KnowledgeBaseGenerator
from mlcli.meta_ml.engine import MetaMLEngine
from mlcli.meta_ml.features import ArtifactFeatureExtractor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, precision_score, recall_score

def main():
    print("🧠 Training Meta-ML Recommendation Engine...")
    
    # Step 1: Generate knowledge base
    print("\n1️⃣ Generating synthetic knowledge base...")
    kb_generator = KnowledgeBaseGenerator()
    knowledge_base = kb_generator.generate_knowledge_base(n_scenarios=10000)
    
    # Save for inspection
    kb_path = Path("data/meta_ml/synthetic_kb.csv")
    kb_path.parent.mkdir(parents=True, exist_ok=True)
    knowledge_base.to_csv(kb_path, index=False)
    print(f"✓ Knowledge base saved: {kb_path}")
    
    # Step 2: Convert to training format
    print("\n2️⃣ Preparing training data...")
    X, y = prepare_training_data(knowledge_base)
    print(f"✓ Training data: X={X.shape}, y={y.shape}")
    
    # Step 3: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Step 4: Train model
    print("\n3️⃣ Training model...")
    engine = MetaMLEngine()
    metrics = engine.train(X_train, y_train, calibrate=True)
    print(f"✓ Training complete: {metrics}")
    
    # Step 5: Evaluate
    print("\n4️⃣ Evaluating on test set...")
    y_pred = engine.model.predict(X_test)
    
    test_metrics = {
        "hamming_loss": hamming_loss(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='samples', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='samples', zero_division=0),
    }
    print(f"✓ Test metrics: {test_metrics}")
    
    # Step 6: Save model
    print("\n5️⃣ Saving model...")
    model_path = Path("data/meta_ml/suggestion_model_v1.pkl")
    engine.save_model(model_path)
    print(f"✓ Model saved: {model_path}")
    
    print("\n✅ Meta-ML engine training complete!")
    print(f"   To use: mlcli suggest (will auto-load the model)")

if __name__ == "__main__":
    main()
```

---

### **Phase 3D: Production Integration** (Week 6)
**Goal:** Wire everything into the suggest command with graceful fallbacks.

#### 3D.1: Upgrade Suggest Command
**Files to modify:**
- `mlcli/commands/suggest_cmd.py` — Complete rewrite

**Implementation:**
```python
# mlcli/commands/suggest_cmd.py (NEW VERSION)
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn

from mlcli.meta_ml.engine import MetaMLEngine
from mlcli.core.telemetry.collector import TelemetryCollector
from mlcli.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()
app = typer.Typer()

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    use_ml: bool = typer.Option(True, "--ml/--rules", help="Use ML engine or fallback to rules"),
) -> None:
    """Get AI-powered suggestions for improving your ML pipeline."""
    
    config = ctx.obj["config"]
    project_dir = ctx.obj["project_dir"]
    
    # Initialize telemetry
    telemetry = TelemetryCollector(project_dir)
    
    # Load required artifacts
    data_profile_path = project_dir / "data" / "processed" / "data_profile.json"
    training_summary_path = project_dir / "models" / "training_summary.json"
    evaluation_report_path = project_dir / "reports" / "evaluation_report.json"
    
    # Check if artifacts exist
    missing = []
    if not data_profile_path.exists():
        missing.append("data_profile.json (run: mlcli preprocess)")
    if not training_summary_path.exists():
        missing.append("training_summary.json (run: mlcli train)")
    if not evaluation_report_path.exists():
        missing.append("evaluation_report.json (run: mlcli evaluate)")
    
    if missing:
        console.print("[red]⚠️  Missing required artifacts:[/red]")
        for item in missing:
            console.print(f"  • {item}")
        raise typer.Exit(1)
    
    # Try to use ML engine
    suggestions = None
    engine_used = "rules"
    
    if use_ml:
        try:
            console.print("[blue]🧠 Loading Meta-ML recommendation engine...[/blue]")
            model_path = Path(__file__).parent.parent.parent / "data" / "meta_ml" / "suggestion_model_v1.pkl"
            
            if model_path.exists():
                engine = MetaMLEngine(model_path)
                suggestions = engine.predict_suggestions(
                    data_profile_path,
                    training_summary_path,
                    evaluation_report_path,
                    top_k=5,
                    confidence_threshold=0.25
                )
                engine_used = "meta_ml"
                console.print("[green]✓[/green] Meta-ML engine loaded\n")
            else:
                console.print("[yellow]⚠️  Meta-ML model not found, using rule-based fallback[/yellow]\n")
        except Exception as e:
            console.print(f"[yellow]⚠️  Meta-ML engine failed ({e}), using rules[/yellow]\n")
    
    # Fallback to rules if ML failed
    if suggestions is None:
        console.print("[blue]📋 Using rule-based suggestion engine...[/blue]\n")
        from mlcli.commands.suggest_cmd_legacy import generate_rule_based_suggestions
        suggestions = generate_rule_based_suggestions(
            data_profile_path,
            training_summary_path,
            evaluation_report_path
        )
    
    # Display suggestions with confidence visualization
    _display_ml_suggestions(suggestions, engine_used)
    
    # Log to telemetry
    session_id = telemetry.log_suggestion_shown(
        [{"label": s[0], "confidence": s[1]} for s in suggestions],
        {"engine": engine_used}
    )
    
    # Save session ID for potential feedback
    (project_dir / ".mlcli" / "last_suggestion_session").write_text(session_id)
    
    console.print(f"\n[dim]Suggestions generated by: {engine_used}[/dim]")

def _display_ml_suggestions(suggestions: list, engine_type: str) -> None:
    """Display ML-powered suggestions with confidence scores."""
    
    console.print(Panel.fit(
        "🤖 [bold]AI-Powered Improvement Suggestions[/bold]",
        style="bold blue"
    ))
    
    for idx, (label, confidence, action_code) in enumerate(suggestions, 1):
        # Confidence bar
        console.print(f"\n[bold cyan]{idx}. {label.replace('_', ' ').title()}[/bold cyan]")
        
        # Visual confidence indicator
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            transient=False
        ) as progress:
            task = progress.add_task(
                f"Confidence: ",
                total=100,
                completed=confidence * 100
            )
        
        # Interpretation
        if confidence >= 0.8:
            interpretation = "[green]High confidence — Strongly recommended[/green]"
        elif confidence >= 0.5:
            interpretation = "[yellow]Medium confidence — Recommended if relevant[/yellow]"
        else:
            interpretation = "[dim]Low confidence — Consider context[/dim]"
        
        console.print(f"   {interpretation}")
        
        # Action code
        console.print(f"\n   [bold]Action:[/bold]")
        console.print(f"   [dim]{action_code}[/dim]")
```

---

## 📊 Success Metrics

### Phase 3A (Infrastructure)
- [ ] All JSON artifacts validated with Pydantic schemas
- [ ] Preprocessing pipeline save/load works in predict command
- [ ] Artifact tracker logs all generated files
- [ ] Telemetry collector records 10+ event types
- [ ] 80%+ test coverage for new core modules

### Phase 3B (Knowledge Base)
- [ ] 10,000+ synthetic scenarios generated
- [ ] Manual review of 100 samples shows >90% label accuracy
- [ ] Feature extractor handles missing fields gracefully
- [ ] Edge cases covered (empty datasets, perfect scores, etc.)

### Phase 3C (Meta-ML Engine)
- [ ] Model achieves <0.15 Hamming loss on test set
- [ ] Calibrated probabilities are well-distributed
- [ ] Top-3 suggestions match expert intuition on 20 real projects
- [ ] Model latency <100ms on typical hardware

### Phase 3D (Integration)
- [ ] Suggest command uses ML by default, graceful fallback to rules
- [ ] Confidence scores displayed with visual progress bars
- [ ] Action codes are actionable and correct
- [ ] Telemetry captures suggestion → action flow
- [ ] User feedback: 70%+ of users find suggestions helpful (survey)

---

## 🔄 Continuous Improvement Loop

```
┌──────────────────────────────────────────────────────────┐
│  Month 1: Deploy with synthetic KB                      │
│  - Collect real telemetry from users                    │
│  - Log which suggestions were helpful                   │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│  Month 2-3: Accumulate 1000+ real project traces        │
│  - Merge real data with synthetic KB                    │
│  - Weight real data higher in training                  │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│  Month 4+: Retrain quarterly                            │
│  - Use only real data (discard synthetic)               │
│  - Deploy new model via GitHub release                  │
│  - A/B test new vs old model                            │
└──────────────────────────────────────────────────────────┘
```

---

## 🎯 Why This Approach Will Scale

### 1. **Proper Separation of Concerns**
- Artifacts have schemas (type safety)
- Pipeline persistence ensures reproducibility
- Telemetry is privacy-first and opt-in

### 2. **Graceful Degradation**
- ML engine failure → fallback to rules
- Missing artifacts → clear error messages
- No model available → still functional

### 3. **Feedback-Driven Learning**
- Real user actions improve the model
- Cold start problem solved with synthetic data
- Continuous retraining keeps model fresh

### 4. **Production-Ready from Day 1**
- Versioned artifacts (audit trail)
- Calibrated confidence scores (trustworthy)
- Test coverage + CI/CD integration

### 5. **Extensible Architecture**
- New suggestion types → just add labels
- New features → extend feature extractor
- Custom domains → plugin-specific feature extractors

---

## 🚧 Known Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| **Cold start (no real data)** | Start with 10k synthetic scenarios validated by experts |
| **Model drift** | Quarterly retraining + A/B testing |
| **Poor suggestions hurt trust** | Always show confidence + fallback to rules |
| **Privacy concerns** | Local-first telemetry, explicit opt-in, PII filtering |
| **Overfitting synthetic data** | Weight real data 10x higher once available |
| **High inference latency** | Cache features, optimize model (quantization) |

---

## 💼 Development Timeline

| Phase | Duration | Blockers | Team Size |
|-------|----------|----------|-----------|
| 3A: Infrastructure | 2 weeks | None | 1-2 devs |
| 3B: Knowledge Base | 1 week | 3A complete | 1 dev + 1 ML expert |
| 3C: Meta-ML Engine | 2 weeks | 3B complete | 1-2 ML engineers |
| 3D: Integration | 1 week | 3C complete | 1 dev |
| **Total** | **6 weeks** | Sequential dependencies | 2-3 engineers |

---

## 📝 Next Immediate Actions

1. **Get Buy-in:** Review this plan with the team, align on priorities
2. **Spike 3A.2:** Prove preprocessing pipeline persistence works end-to-end (2 days)
3. **Design Review:** Schema design for all artifacts (1 day)
4. **Start 3A.1:** Implement Pydantic schemas (3 days)
5. **Parallel Track:** One engineer starts 3B.1 (KB generator) while 3A progresses

---

## 🎓 Key Learnings from Billion-Dollar Companies

### 1. **Netflix Recommendation Engine**
- Started with simple rules, gradually migrated to ML
- Always had fallback to rules (reliability)
- A/B tested everything

### 2. **Uber's ML Platform**
- Artifact versioning from day 1
- Strong schema validation prevented production bugs
- Telemetry drove 80% of model improvements

### 3. **Airbnb's ML Infra**
- "Feature stores" for consistent feature engineering
- Offline/online parity critical for reproducibility
- Invested in tooling before scaling models

**Takeaway:** Infrastructure first, fancy models second.

---

## ✅ Definition of Done

Phase 3 is "complete" when:
1. [ ] A user runs `mlcli suggest` and sees ML-powered suggestions with confidence scores
2. [ ] The command gracefully falls back to rules if ML fails
3. [ ] Telemetry captures user actions on suggestions
4. [ ] Test coverage >75% for all new modules
5. [ ] Documentation updated with examples
6. [ ] Model retraining script works end-to-end
7. [ ] 10 beta users validate suggestions are helpful

---

**This is a production-grade, scalable plan. Let's build it right.**
