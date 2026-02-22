"""Meta-ML Suggestion Engine.

Main inference engine that provides ML-powered suggestions based on
data profile and evaluation metrics.

Usage:
    from mlcli.meta_ml import SuggestionEngine
    
    engine = SuggestionEngine()
    suggestions = engine.predict(
        data_profile={"n_samples": 1000, ...},
        evaluation_report={"accuracy": 0.75, ...}
    )
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from .knowledge_base import SUGGESTION_LABELS


SUGGESTION_INFO = {
    "COLLECT_MORE_DATA": {
        "issue": "Small dataset size limits model reliability",
        "impact": "Better generalization and more reliable model performance",
        "action": "Add more samples to data/raw/ or use data augmentation techniques",
    },
    "TRY_SIMPLE_MODELS": {
        "issue": "Complex models may overfit with limited data",
        "impact": "Reduced overfitting and improved training stability",
        "action": "Use logistic regression or naive bayes instead of complex models",
    },
    "FEATURE_ENGINEERING": {
        "issue": "Current features may not capture all predictive patterns",
        "impact": "Unlock hidden patterns and improve accuracy",
        "action": "Create interaction features or domain-specific features",
    },
    "FEATURE_SELECTION": {
        "issue": "High dimensionality relative to sample count",
        "impact": "Faster training, better interpretability, reduced overfitting",
        "action": "Use SelectKBest or feature importance to reduce dimensions",
    },
    "DIMENSIONALITY_REDUCTION": {
        "issue": "Too many features tracking similar information",
        "impact": "Reduced training time and filtered noise",
        "action": "Apply PCA to reduce feature space",
    },
    "SMOTE_IMBALANCE": {
        "issue": "Significant class imbalance detected in dataset",
        "impact": "Balanced performance across all classes",
        "action": "Update mlcli.yaml: data.imbalance_strategy: smote",
    },
    "CLASS_WEIGHTS": {
        "issue": "Model training may be biased towards majority class",
        "impact": "Fair treatment of minority classes during training",
        "action": "Update mlcli.yaml: model.class_weight: balanced",
    },
    "STRATIFIED_SAMPLING": {
        "issue": "Class distribution may not be preserved in splits",
        "impact": "Representative evaluation across class distribution",
        "action": "Update mlcli.yaml: data.stratify: true",
    },
    "HYPERPARAMETER_TUNING": {
        "issue": "Sub-optimal model configuration detected",
        "impact": "Maximum model performance for your dataset",
        "action": "Update mlcli.yaml: model.hyperparameter_tuning: true",
    },
    "TRY_ENSEMBLE_MODELS": {
        "issue": "Individual models hitting performance ceiling",
        "impact": "More robust predictions and improved accuracy",
        "action": "Update mlcli.yaml: model.algorithms: [random_forest, xgboost]",
    },
    "REGULARIZATION": {
        "issue": "Signs of overfitting detected",
        "impact": "Better generalization on unseen data",
        "action": "Add L1/L2 regularization or increase dropout",
    },
    "HANDLE_MISSING_VALUES": {
        "issue": "High percentage of missing values detected",
        "impact": "More reliable training and reduced bias",
        "action": "Improve imputation strategy or remove high-missing columns",
    },
    "DATA_AUGMENTATION": {
        "issue": "Limited data variety may affect generalization",
        "impact": "Improved model generalization",
        "action": "Use SMOTE or synthetic data generation techniques",
    },
    "OUTLIER_TREATMENT": {
        "issue": "Outliers may be affecting model training",
        "impact": "More stable and reliable model",
        "action": "Apply robust scaling or remove outlier samples",
    },
    "CROSS_VALIDATION": {
        "issue": "Model evaluation may not be reliable",
        "impact": "More reliable model evaluation metrics",
        "action": "Increase cv_folds in mlcli.yaml for reliable evaluation",
    },
    "EARLY_STOPPING": {
        "issue": "Training may be continuing too long",
        "impact": "Prevented overfitting and faster training",
        "action": "Add early stopping to training configuration",
    },
    "LEARNING_RATE_TUNING": {
        "issue": "Learning rate may need adjustment",
        "impact": "Better convergence and improved accuracy",
        "action": "Try learning rates: [0.001, 0.01, 0.1]",
    },
    "BATCH_SIZE_OPTIMIZATION": {
        "issue": "Batch size may affect training stability",
        "impact": "More stable training and better gradients",
        "action": "Experiment with batch sizes: [16, 32, 64]",
    },
    "MODEL_ARCHITECTURE_CHANGE": {
        "issue": "Current model may not be optimal for this task",
        "impact": "Better suited model for your problem",
        "action": "Try different architectures suited for your data type",
    },
}

FEATURE_NAMES = [
    "n_samples",
    "n_features",
    "missing_pct_max",
    "imbalance_ratio",
    "accuracy",
    "f1_score",
    "precision_recall_gap",
]


class SuggestionEngine:
    """ML-powered suggestion engine.
    
    Uses a trained multi-label classifier to suggest ML improvements
    based on data profile and evaluation metrics.
    
    Features:
    - ML-powered predictions with confidence scores
    - Graceful fallback to rule-based suggestions
    - Human-readable issue/impact/action descriptions
    
    Example:
        engine = SuggestionEngine()
        
        # Load a trained model (optional)
        engine.load_model("suggestion_model_v2.pkl")
        
        # Get suggestions
        suggestions = engine.predict(
            data_profile={"n_samples": 500, "n_features": 100, ...},
            evaluation_report={"accuracy": 0.72, "f1_score": 0.70, ...}
        )
        
        for s in suggestions:
            print(f"{s['suggestion']}: {s['confidence']:.1%}")
            print(f"  Action: {s['action']}")
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the suggestion engine.
        
        Args:
            model_path: Optional path to trained model file
        """
        self.model = None
        self.mlb = None
        self.feature_names = FEATURE_NAMES
        self.labels = SUGGESTION_LABELS
        self.version = "unknown"
        self.model_path = model_path
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: Union[str, Path]) -> bool:
        """Load a trained model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            return False
        
        try:
            if HAS_JOBLIB:
                artifacts = joblib.load(model_path)
            else:
                import pickle
                with open(model_path, 'rb') as f:
                    artifacts = pickle.load(f)
            
            self.model = artifacts.get('model')
            self.mlb = artifacts.get('mlb')
            
            if 'feature_names' in artifacts:
                self.feature_names = artifacts['feature_names']
            if 'suggestion_labels' in artifacts:
                self.labels = artifacts['suggestion_labels']
            if 'version' in artifacts:
                self.version = artifacts['version']
            
            self.model_path = model_path
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load model: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if the engine is ready for ML predictions."""
        return self.model is not None and self.mlb is not None
    
    def extract_features(
        self,
        data_profile: Dict[str, Any],
        evaluation_report: Dict[str, Any]
    ) -> np.ndarray:
        """Extract feature vector from data profile and evaluation report.
        
        Args:
            data_profile: Data profile dictionary
            evaluation_report: Evaluation report dictionary
            
        Returns:
            7-dimensional feature vector
        """
        shape = data_profile.get("shape", data_profile.get("original_shape", [0, 0]))
        n_samples = float(shape[0]) if shape else 0.0
        n_features = float(shape[1]) if len(shape) > 1 else 0.0
        
        missing_pct = data_profile.get("missing_percentage", data_profile.get("missing_pct", {}))
        if missing_pct:
            max_missing = max(missing_pct.values())
            if isinstance(max_missing, (int, float)) and max_missing > 1:
                missing_pct_max = max_missing / 100.0
            else:
                missing_pct_max = float(max_missing)
        else:
            missing_pct_max = 0.0
        
        target_info = data_profile.get("target_info", {})
        value_counts = target_info.get("value_counts", {})
        if value_counts:
            counts = list(value_counts.values())
            if counts and min(counts) > 0:
                imbalance_ratio = float(max(counts)) / float(min(counts))
            else:
                imbalance_ratio = 1.0
        else:
            imbalance_ratio = data_profile.get("imbalance_ratio", 1.0)
        
        metrics = evaluation_report.get("metrics", {})
        accuracy = float(metrics.get("accuracy", metrics.get("Accuracy", 0.0)))
        f1_score = float(metrics.get("f1_score", metrics.get("F1 Score", 0.0)))
        
        precision = float(metrics.get("precision", metrics.get("Precision", 0.0)))
        recall = float(metrics.get("recall", metrics.get("Recall", 0.0)))
        precision_recall_gap = abs(precision - recall)
        
        return np.array([[
            n_samples,
            n_features,
            missing_pct_max,
            imbalance_ratio,
            accuracy,
            f1_score,
            precision_recall_gap,
        ]])
    
    def predict(
        self,
        data_profile: Dict[str, Any],
        evaluation_report: Dict[str, Any],
        top_k: int = 5,
        confidence_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Generate suggestions based on data profile and evaluation.
        
        Args:
            data_profile: Data profile dictionary
            evaluation_report: Evaluation report dictionary
            top_k: Maximum number of suggestions to return
            confidence_threshold: Minimum confidence to include
            
        Returns:
            List of suggestion dictionaries
        """
        features = self.extract_features(data_profile, evaluation_report)
        
        if self.is_ready():
            suggestions = self._predict_ml(features, top_k, confidence_threshold)
        else:
            suggestions = self._predict_rules(features, top_k)
        
        return suggestions
    
    def _predict_ml(
        self,
        features: np.ndarray,
        top_k: int,
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """Use ML model for predictions."""
        y_probs = self.model.predict_proba(features)
        
        if isinstance(y_probs, list):
            probs = [p[0][1] if p.shape[1] == 2 else p[0][0] for p in y_probs]
        else:
            probs = y_probs[0]
        
        classes = self.mlb.classes_ if hasattr(self.mlb, 'classes_') else self.labels
        
        suggestions = []
        for i, class_name in enumerate(classes):
            if i < len(probs) and probs[i] > confidence_threshold:
                info = SUGGESTION_INFO.get(str(class_name), {})
                suggestions.append({
                    "suggestion": str(class_name),
                    "confidence": float(probs[i]),
                    "issue": info.get("issue", "Performance improvement opportunity"),
                    "impact": info.get("impact", "Improves overall pipeline robustness"),
                    "action": info.get("action", "Apply recommended ML technique"),
                    "source": "ml",
                })
        
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        return suggestions[:top_k]
    
    def _predict_rules(
        self,
        features: np.ndarray,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Fallback rule-based predictions."""
        n_samples, n_features, missing_pct_max, imbalance_ratio, accuracy, f1_score, precision_recall_gap = features[0]
        
        suggestions = []
        
        if n_samples < 100:
            suggestions.append(("COLLECT_MORE_DATA", 0.9))
            suggestions.append(("TRY_SIMPLE_MODELS", 0.85))
        elif n_samples < 500:
            suggestions.append(("DATA_AUGMENTATION", 0.7))
        
        if imbalance_ratio > 10:
            suggestions.append(("SMOTE_IMBALANCE", 0.85))
            suggestions.append(("CLASS_WEIGHTS", 0.8))
        elif imbalance_ratio > 5:
            suggestions.append(("SMOTE_IMBALANCE", 0.75))
        elif imbalance_ratio > 3:
            suggestions.append(("STRATIFIED_SAMPLING", 0.65))
        
        if n_features > 100:
            suggestions.append(("FEATURE_SELECTION", 0.8))
            suggestions.append(("DIMENSIONALITY_REDUCTION", 0.75))
        elif n_features > 50 and n_samples < 5000:
            suggestions.append(("DIMENSIONALITY_REDUCTION", 0.7))
        
        if accuracy < 0.6:
            suggestions.append(("FEATURE_ENGINEERING", 0.85))
            suggestions.append(("TRY_ENSEMBLE_MODELS", 0.8))
            suggestions.append(("HYPERPARAMETER_TUNING", 0.75))
        elif accuracy < 0.7:
            suggestions.append(("FEATURE_ENGINEERING", 0.8))
            suggestions.append(("HYPERPARAMETER_TUNING", 0.75))
        elif accuracy < 0.85:
            suggestions.append(("HYPERPARAMETER_TUNING", 0.7))
        
        if precision_recall_gap > 0.15:
            suggestions.append(("SMOTE_IMBALANCE", 0.75))
        
        if missing_pct_max > 0.3:
            suggestions.append(("HANDLE_MISSING_VALUES", 0.8))
        if missing_pct_max > 0.5:
            suggestions.append(("OUTLIER_TREATMENT", 0.7))
        
        if not suggestions:
            suggestions.append(("HYPERPARAMETER_TUNING", 0.5))
        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for label, confidence in suggestions[:top_k]:
            info = SUGGESTION_INFO.get(label, {})
            result.append({
                "suggestion": label,
                "confidence": confidence,
                "issue": info.get("issue", "Performance improvement opportunity"),
                "impact": info.get("impact", "Improves overall pipeline robustness"),
                "action": info.get("action", "Apply recommended ML technique"),
                "source": "rules",
            })
        
        return result
    
    def get_suggestion_info(self, suggestion: str) -> Dict[str, str]:
        """Get detailed information about a suggestion.
        
        Args:
            suggestion: Suggestion name
            
        Returns:
            Dictionary with issue, impact, and action
        """
        return SUGGESTION_INFO.get(suggestion, {
            "issue": "Performance improvement opportunity",
            "impact": "Improves overall pipeline robustness",
            "action": "Apply recommended ML technique",
        })
