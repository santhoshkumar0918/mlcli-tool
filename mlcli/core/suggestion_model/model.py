"""Inference engine for the suggestion meta-model."""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from .features import extract_features_from_reports

class MLSuggestionEngine:
    """ML-powered suggestion generator."""
    
    def __init__(self, model_path: Path = None):
        if model_path is None:
            model_path = Path(__file__).parent / "data" / "suggestion_model.pkl"
            
        if not model_path.exists():
            self.model = None
            self.mlb = None
            return
            
        with open(model_path, 'rb') as f:
            artifacts = pickle.load(f)
            self.model = artifacts['model']
            self.mlb = artifacts['mlb']
            
    def get_suggestions(self, data_profile: Dict[str, Any], evaluation_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict relevant suggestions based on reports."""
        
        if self.model is None or self.mlb is None:
            return []
            
        # Extract features
        features = extract_features_from_reports(data_profile, evaluation_report)
        
        # Get probabilities for each label
        # model is a MultiOutputClassifier, predict_proba returns a list of arrays
        y_probs = self.model.predict_proba(features)
        
        # Extract probabilities for the 'Positive' class (index 1) for each output
        # y_probs[i] is shape (1, 2) where y_probs[i][0][1] is prob of class i being active
        probs = [p[0][1] for p in y_probs]
        
        # Get class names
        classes = self.mlb.classes_
        
        # Format results
        suggestions = []
        for i, class_name in enumerate(classes):
            if probs[i] > 0.3: # Threshold
                suggestions.append({
                    "suggestion": class_name,
                    "confidence": float(probs[i]),
                    "issue": self._map_to_issue(class_name),
                    "impact": self._map_to_impact(class_name)
                })
                
        # Sort by confidence
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return suggestions
        
    def _map_to_issue(self, suggestion: str) -> str:
        mapping = {
            "COLLECT_MORE_DATA": "Small dataset size",
            "TRY_SIMPLE_MODELS": "Model might be too complex for data size",
            "FEATURE_ENGINEERING": "Limited informative features",
            "SMOTE_IMBALANCE": "Significant class imbalance detected",
            "CLASS_WEIGHTS": "Imbalanced class distribution",
            "STRATIFIED_SAMPLING": "Potential bias in data splits",
            "FEATURE_SELECTION": "High dimensionality relative to samples",
            "DIMENSIONALITY_REDUCTION": "Too many features tracking similar information",
            "REGULARIZATION": "Evidence of overfitting",
            "HYPERPARAMETER_TUNING": "Sub-optimal model performance",
            "TRY_ENSEMBLE_MODELS": "Individual models hitting performance ceiling",
            "XGBOOST_UPGRADE": "Linear/Forest models underperforming",
            "HANDLE_MISSING_VALUES": "High percentage of missing data",
            "DATA_CLEANING": "Data quality issues detected",
            "FEATURE_IMPUTATION": "Missing values impacting model accuracy"
        }
        return mapping.get(suggestion, "Performance bottleneck identified")
        
    def _map_to_impact(self, suggestion: str) -> str:
        mapping = {
            "COLLECT_MORE_DATA": "Better generalization and higher accuracy",
            "TRY_SIMPLE_MODELS": "Reduced overfitting and improved stability",
            "FEATURE_ENGINEERING": "Unlocks hidden patterns in data",
            "SMOTE_IMBALANCE": "Improves sensitivity to minority classes",
            "CLASS_WEIGHTS": "Better balance across all evaluation metrics",
            "STRATIFIED_SAMPLING": "More reliable evaluation results",
            "FEATURE_SELECTION": "Faster training and better interpretability",
            "DIMENSIONALITY_REDUCTION": "Filters noise and speeds up inference",
            "REGULARIZATION": "Better performance on unseen data",
            "HYPERPARAMETER_TUNING": "Squeezes maximum performance from current model",
            "TRY_ENSEMBLE_MODELS": "More robust and accurate predictions",
            "XGBOOST_UPGRADE": "Significant jump in predictive power",
            "HANDLE_MISSING_VALUES": "Reduces bias from incomplete samples",
            "DATA_CLEANING": "Improves signal-to-noise ratio",
            "FEATURE_IMPUTATION": "Prevents data loss during training"
        }
        return mapping.get(suggestion, "Improves overall pipeline robustness")
