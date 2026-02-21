"""Feature extraction for the suggestion meta-model."""

import numpy as np
from typing import Dict, Any, List

def extract_features_from_reports(data_profile: Dict[str, Any], evaluation_report: Dict[str, Any]) -> np.ndarray:
    """
    Extract a numeric feature vector from MLCLI project reports.
    
    Features:
    1. n_samples
    2. n_features
    3. missing_pct_max
    4. imbalance_ratio
    5. accuracy
    6. f1_score
    7. precision_recall_gap
    """
    
    # 1. n_samples
    n_samples = data_profile.get("shape", [0, 0])[0]
    
    # 2. n_features
    n_features = data_profile.get("shape", [0, 0])[1]
    
    # 3. missing_pct_max
    missing_pcts = data_profile.get("missing_percentage", {})
    missing_pct_max = max(missing_pcts.values()) if missing_pcts else 0.0
    
    # 4. imbalance_ratio
    # Simplified: max_class_count / min_class_count
    imbalance_ratio = 1.0
    target_info = data_profile.get("target_info", {})
    if target_info and "value_counts" in target_info:
        counts = list(target_info["value_counts"].values())
        if counts and min(counts) > 0:
            imbalance_ratio = max(counts) / min(counts)
            
    # 5. accuracy
    metrics = evaluation_report.get("metrics", {})
    # Note: evaluation_report might have different structure, checking common keys
    accuracy = metrics.get("Accuracy", 0.0)
    if not accuracy: # try lowercase
        accuracy = metrics.get("accuracy", 0.0)
        
    # 6. f1_score
    f1_score = metrics.get("F1 Score", 0.0)
    if not f1_score:
        f1_score = metrics.get("f1_score", 0.0)
        
    # 7. precision_recall_gap
    precision = metrics.get("Precision", metrics.get("precision", 0.0))
    recall = metrics.get("Recall", metrics.get("recall", 0.0))
    pr_gap = abs(precision - recall)
    
    features = [
        float(n_samples),
        float(n_features),
        float(missing_pct_max),
        float(imbalance_ratio),
        float(accuracy),
        float(f1_score),
        float(pr_gap)
    ]
    
    return np.array(features).reshape(1, -1)

FEATURE_NAMES = [
    "n_samples",
    "n_features",
    "missing_pct_max",
    "imbalance_ratio",
    "accuracy",
    "f1_score",
    "precision_recall_gap"
]
