"""Inference engine for the suggestion meta-model.

This module provides the MLSuggestionEngine class that loads trained models
and generates ML improvement suggestions based on data profiles and evaluation reports.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


SUGGESTION_LABELS = [
    "COLLECT_MORE_DATA",
    "TRY_SIMPLE_MODELS",
    "FEATURE_ENGINEERING",
    "SMOTE_IMBALANCE",
    "CLASS_WEIGHTS",
    "STRATIFIED_SAMPLING",
    "FEATURE_SELECTION",
    "DIMENSIONALITY_REDUCTION",
    "HYPERPARAMETER_TUNING",
    "TRY_ENSEMBLE_MODELS",
    "REGULARIZATION",
    "HANDLE_MISSING_VALUES",
    "DATA_AUGMENTATION",
    "OUTLIER_TREATMENT",
    "CROSS_VALIDATION",
    "EARLY_STOPPING",
    "LEARNING_RATE_TUNING",
    "BATCH_SIZE_OPTIMIZATION",
    "MODEL_ARCHITECTURE_CHANGE",
]

FEATURE_NAMES = [
    "n_samples",
    "n_features",
    "missing_pct_max",
    "imbalance_ratio",
    "accuracy",
    "f1_score",
    "precision_recall_gap",
]


def extract_features_from_reports(
    data_profile: Dict[str, Any], 
    evaluation_report: Dict[str, Any]
) -> np.ndarray:
    """Extract feature vector from data profile and evaluation report.
    
    Args:
        data_profile: Dictionary containing data profile information
        evaluation_report: Dictionary containing evaluation metrics
        
    Returns:
        7-dimensional feature vector as numpy array
    """
    shape = data_profile.get("shape", data_profile.get("original_shape", [0, 0]))
    n_samples = float(shape[0]) if shape else 0.0
    n_features = float(shape[1]) if len(shape) > 1 else 0.0
    
    missing_pct = data_profile.get("missing_percentage", data_profile.get("missing_pct", {}))
    missing_max = max(missing_pct.values()) if missing_pct else 0.0
    if isinstance(list(missing_pct.values())[0] if missing_pct else 0, float):
        missing_pct_max = missing_max / 100.0 if missing_max > 1 else missing_max
    else:
        missing_pct_max = missing_max / 100.0 if missing_max > 1 else missing_max
    
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
    accuracy = float(metrics.get("Accuracy", metrics.get("accuracy", 0.0)))
    f1_score = float(metrics.get("F1 Score", metrics.get("f1_score", 0.0)))
    
    precision = float(metrics.get("Precision", metrics.get("precision", 0.0)))
    recall = float(metrics.get("Recall", metrics.get("recall", 0.0)))
    precision_recall_gap = abs(precision - recall)
    
    features = np.array([
        n_samples,
        n_features,
        missing_pct_max,
        imbalance_ratio,
        accuracy,
        f1_score,
        precision_recall_gap,
    ]).reshape(1, -1)
    
    return features


class MLSuggestionEngine:
    """ML-powered suggestion generator.
    
    Loads a trained multi-label classifier and generates suggestions
    based on data profile and evaluation report inputs.
    
    Example:
        engine = MLSuggestionEngine()
        suggestions = engine.get_suggestions(data_profile, evaluation_report)
        for s in suggestions:
            print(f"{s['suggestion']}: {s['confidence']:.1%}")
    """
    
    ISSUE_MAPPING = {
        "COLLECT_MORE_DATA": "Small dataset size limits model reliability",
        "TRY_SIMPLE_MODELS": "Complex models may overfit with limited data",
        "FEATURE_ENGINEERING": "Current features may not capture all predictive patterns",
        "SMOTE_IMBALANCE": "Significant class imbalance detected in dataset",
        "CLASS_WEIGHTS": "Model training may be biased towards majority class",
        "STRATIFIED_SAMPLING": "Class distribution may not be preserved in splits",
        "FEATURE_SELECTION": "High dimensionality relative to sample count",
        "DIMENSIONALITY_REDUCTION": "Too many features tracking similar information",
        "REGULARIZATION": "Signs of overfitting detected",
        "HYPERPARAMETER_TUNING": "Sub-optimal model configuration detected",
        "TRY_ENSEMBLE_MODELS": "Individual models hitting performance ceiling",
        "HANDLE_MISSING_VALUES": "High percentage of missing values detected",
        "DATA_AUGMENTATION": "Limited data variety may affect generalization",
        "OUTLIER_TREATMENT": "Outliers may be affecting model training",
        "CROSS_VALIDATION": "Model evaluation may not be reliable",
        "EARLY_STOPPING": "Training may be continuing too long",
        "LEARNING_RATE_TUNING": "Learning rate may need adjustment",
        "BATCH_SIZE_OPTIMIZATION": "Batch size may affect training stability",
        "MODEL_ARCHITECTURE_CHANGE": "Current model may not be optimal for this task",
    }
    
    IMPACT_MAPPING = {
        "COLLECT_MORE_DATA": "Better generalization and more reliable model performance",
        "TRY_SIMPLE_MODELS": "Reduced overfitting and improved training stability",
        "FEATURE_ENGINEERING": "Unlock hidden patterns and improve accuracy",
        "SMOTE_IMBALANCE": "Balanced performance across all classes",
        "CLASS_WEIGHTS": "Fair treatment of minority classes during training",
        "STRATIFIED_SAMPLING": "Representative evaluation across class distribution",
        "FEATURE_SELECTION": "Faster training, better interpretability, reduced overfitting",
        "DIMENSIONALITY_REDUCTION": "Reduced training time and filtered noise",
        "REGULARIZATION": "Better generalization on unseen data",
        "HYPERPARAMETER_TUNING": "Maximum model performance for your dataset",
        "TRY_ENSEMBLE_MODELS": "More robust predictions and improved accuracy",
        "HANDLE_MISSING_VALUES": "More reliable training and reduced bias",
        "DATA_AUGMENTATION": "Improved model generalization",
        "OUTLIER_TREATMENT": "More stable and reliable model",
        "CROSS_VALIDATION": "More reliable model evaluation metrics",
        "EARLY_STOPPING": "Prevented overfitting and faster training",
        "LEARNING_RATE_TUNING": "Better convergence and improved accuracy",
        "BATCH_SIZE_OPTIMIZATION": "More stable training and better gradients",
        "MODEL_ARCHITECTURE_CHANGE": "Better suited model for your problem",
    }
    
    ACTION_MAPPING = {
        "COLLECT_MORE_DATA": "Add more samples to data/raw/ or use data augmentation",
        "TRY_SIMPLE_MODELS": "Use logistic regression or naive bayes instead of complex models",
        "FEATURE_ENGINEERING": "Create interaction features or domain-specific features",
        "SMOTE_IMBALANCE": "Update mlcli.yaml: data.imbalance_strategy: smote",
        "CLASS_WEIGHTS": "Update mlcli.yaml: model.class_weight: balanced",
        "STRATIFIED_SAMPLING": "Update mlcli.yaml: data.stratify: true",
        "FEATURE_SELECTION": "Use SelectKBest or feature importance to reduce dimensions",
        "DIMENSIONALITY_REDUCTION": "Apply PCA to reduce feature space",
        "REGULARIZATION": "Add L1/L2 regularization or increase dropout",
        "HYPERPARAMETER_TUNING": "Update mlcli.yaml: model.hyperparameter_tuning: true",
        "TRY_ENSEMBLE_MODELS": "Update mlcli.yaml: model.algorithms: [random_forest, xgboost]",
        "HANDLE_MISSING_VALUES": "Improve imputation strategy or remove high-missing columns",
        "DATA_AUGMENTATION": "Use SMOTE or synthetic data generation techniques",
        "OUTLIER_TREATMENT": "Apply robust scaling or remove outlier samples",
        "CROSS_VALIDATION": "Increase cv_folds in mlcli.yaml for reliable evaluation",
        "EARLY_STOPPING": "Add early stopping to training configuration",
        "LEARNING_RATE_TUNING": "Try learning rates: [0.001, 0.01, 0.1]",
        "BATCH_SIZE_OPTIMIZATION": "Experiment with batch sizes: [16, 32, 64]",
        "MODEL_ARCHITECTURE_CHANGE": "Try different architectures suited for your data type",
    }
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the suggestion engine.
        
        Args:
            model_path: Path to the trained model file. If None, uses default location.
        """
        self.model = None
        self.mlb = None
        self.feature_names = FEATURE_NAMES
        self.labels = SUGGESTION_LABELS
        self.version = "unknown"
        
        if model_path is None:
            model_path = Path(__file__).parent / "data" / "suggestion_model_v2.pkl"
        
        if not model_path.exists():
            model_path = Path(__file__).parent / "data" / "suggestion_model.pkl"
        
        if not model_path.exists():
            return
        
        try:
            if HAS_JOBLIB:
                artifacts = joblib.load(model_path)
            else:
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
                
        except Exception as e:
            print(f"Warning: Failed to load suggestion model: {e}")
            self.model = None
            self.mlb = None
    
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self.model is not None
    
    def get_suggestions(
        self, 
        data_profile: Dict[str, Any], 
        evaluation_report: Dict[str, Any],
        confidence_threshold: float = 0.3,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate ML improvement suggestions based on reports.
        
        Args:
            data_profile: Dictionary containing data profile information
            evaluation_report: Dictionary containing evaluation metrics
            confidence_threshold: Minimum confidence to include a suggestion
            top_k: Maximum number of suggestions to return
            
        Returns:
            List of suggestion dictionaries with confidence scores
        """
        if self.model is None or self.mlb is None:
            return []
        
        features = extract_features_from_reports(data_profile, evaluation_report)
        
        y_probs = self.model.predict_proba(features)
        
        if isinstance(y_probs, list):
            probs = [p[0][1] if p.shape[1] == 2 else p[0][0] for p in y_probs]
        else:
            probs = y_probs[0]
        
        classes = self.mlb.classes_ if hasattr(self.mlb, 'classes_') else self.labels
        
        suggestions = []
        for i, class_name in enumerate(classes):
            if i < len(probs) and probs[i] > confidence_threshold:
                suggestions.append({
                    "suggestion": str(class_name),
                    "confidence": float(probs[i]),
                    "issue": self._map_to_issue(class_name),
                    "impact": self._map_to_impact(class_name),
                    "action": self._map_to_action(class_name),
                })
        
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return suggestions[:top_k]
    
    def _map_to_issue(self, suggestion: str) -> str:
        return self.ISSUE_MAPPING.get(suggestion, "Performance improvement opportunity identified")
    
    def _map_to_impact(self, suggestion: str) -> str:
        return self.IMPACT_MAPPING.get(suggestion, "Improves overall pipeline robustness")
    
    def _map_to_action(self, suggestion: str) -> str:
        return self.ACTION_MAPPING.get(suggestion, "Apply recommended ML technique")
