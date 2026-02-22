"""Knowledge Base Generator for Meta-ML Training.

Generates synthetic ML scenarios using expert heuristics derived from:
- Andrew Ng's ML advice
- Production ML best practices
- Common ML pitfalls

Usage:
    from mlcli.meta_ml import KnowledgeBaseGenerator
    
    generator = KnowledgeBaseGenerator(seed=42)
    scenarios = generator.generate_knowledge_base(n_scenarios=10000)
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


SUGGESTION_LABELS = [
    "COLLECT_MORE_DATA",
    "TRY_SIMPLE_MODELS",
    "FEATURE_ENGINEERING",
    "FEATURE_SELECTION",
    "DIMENSIONALITY_REDUCTION",
    "SMOTE_IMBALANCE",
    "CLASS_WEIGHTS",
    "STRATIFIED_SAMPLING",
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


class KnowledgeBaseGenerator:
    """Generate synthetic ML scenarios with expert-labeled suggestions.
    
    Uses domain knowledge to create realistic training scenarios for
    the Meta-ML suggestion engine.
    
    Example:
        generator = KnowledgeBaseGenerator(seed=42)
        scenarios = generator.generate_knowledge_base(10000)
        
        for scenario in scenarios[:5]:
            print(f"Features: {scenario['features']}")
            print(f"Labels: {scenario['labels']}")
    """
    
    SAMPLE_SIZES = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000]
    FEATURE_COUNTS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    IMBALANCE_RATIOS = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_scenario(self) -> Dict[str, Any]:
        """Generate a single synthetic ML scenario.
        
        Returns:
            Dictionary with 'features' and 'labels' keys
        """
        n_samples = random.choice(self.SAMPLE_SIZES)
        n_features = random.choice(self.FEATURE_COUNTS)
        missing_pct_max = round(random.uniform(0, 0.7), 3)
        imbalance_ratio = random.choice(self.IMBALANCE_RATIOS)
        
        base_accuracy = self._simulate_base_accuracy(n_samples, n_features, imbalance_ratio)
        noise = random.gauss(0, 0.05)
        accuracy = max(0.1, min(0.99, base_accuracy + noise))
        
        f1_base = accuracy - random.uniform(0, 0.1)
        f1_score = max(0.1, min(accuracy, f1_base))
        
        precision = min(0.99, f1_score + random.uniform(-0.1, 0.1))
        recall = min(0.99, f1_score + random.uniform(-0.1, 0.1))
        precision_recall_gap = abs(precision - recall)
        
        labels = self._apply_expert_rules(
            n_samples=n_samples,
            n_features=n_features,
            missing_pct_max=missing_pct_max,
            imbalance_ratio=imbalance_ratio,
            accuracy=accuracy,
            f1_score=f1_score,
            precision_recall_gap=precision_recall_gap,
        )
        
        return {
            "features": {
                "n_samples": n_samples,
                "n_features": n_features,
                "missing_pct_max": missing_pct_max,
                "imbalance_ratio": imbalance_ratio,
                "accuracy": round(accuracy, 4),
                "f1_score": round(f1_score, 4),
                "precision_recall_gap": round(precision_recall_gap, 4),
            },
            "labels": labels,
        }
    
    def _simulate_base_accuracy(
        self,
        n_samples: int,
        n_features: int,
        imbalance_ratio: float
    ) -> float:
        """Simulate realistic base accuracy based on data characteristics."""
        base = 0.75
        
        if n_samples < 100:
            base -= 0.15
        elif n_samples < 500:
            base -= 0.05
        elif n_samples > 10000:
            base += 0.05
        
        if n_features > 100:
            base -= 0.05
        if n_features > n_samples:
            base -= 0.10
        
        if imbalance_ratio > 10:
            base -= 0.10
        elif imbalance_ratio > 5:
            base -= 0.05
        
        return base + random.gauss(0, 0.03)
    
    def _apply_expert_rules(
        self,
        n_samples: int,
        n_features: int,
        missing_pct_max: float,
        imbalance_ratio: float,
        accuracy: float,
        f1_score: float,
        precision_recall_gap: float,
    ) -> List[str]:
        """Apply expert heuristics to generate suggestion labels."""
        suggestions = []
        
        if n_samples < 100:
            suggestions.append("COLLECT_MORE_DATA")
            suggestions.append("TRY_SIMPLE_MODELS")
        elif n_samples < 500:
            suggestions.append("DATA_AUGMENTATION")
        
        if imbalance_ratio > 5:
            suggestions.append("SMOTE_IMBALANCE")
        if imbalance_ratio > 10:
            suggestions.append("CLASS_WEIGHTS")
        if imbalance_ratio > 3:
            suggestions.append("STRATIFIED_SAMPLING")
        
        if n_features > 50 and n_samples < 5000:
            suggestions.append("DIMENSIONALITY_REDUCTION")
        if n_features > 100:
            suggestions.append("FEATURE_SELECTION")
        if n_features > n_samples:
            suggestions.append("DIMENSIONALITY_REDUCTION")
        
        if accuracy < 0.7:
            suggestions.append("FEATURE_ENGINEERING")
            suggestions.append("HYPERPARAMETER_TUNING")
        if accuracy < 0.6:
            suggestions.append("TRY_ENSEMBLE_MODELS")
            suggestions.append("MODEL_ARCHITECTURE_CHANGE")
        
        if 0.7 <= accuracy < 0.85:
            suggestions.append("HYPERPARAMETER_TUNING")
        
        if accuracy > 0.95:
            suggestions.append("CROSS_VALIDATION")
        
        if precision_recall_gap > 0.15:
            suggestions.append("SMOTE_IMBALANCE")
            suggestions.append("CLASS_WEIGHTS")
        
        if f1_score < accuracy - 0.1:
            suggestions.append("FEATURE_ENGINEERING")
        
        if missing_pct_max > 0.3:
            suggestions.append("HANDLE_MISSING_VALUES")
        if missing_pct_max > 0.5:
            suggestions.append("OUTLIER_TREATMENT")
        
        if n_samples > 1000 and accuracy < 0.8:
            suggestions.append("EARLY_STOPPING")
            suggestions.append("REGULARIZATION")
        
        if 0.6 <= accuracy < 0.75:
            suggestions.append("LEARNING_RATE_TUNING")
        
        if n_samples > 5000 and n_features < 50:
            suggestions.append("BATCH_SIZE_OPTIMIZATION")
        
        if not suggestions:
            suggestions.append("HYPERPARAMETER_TUNING")
        
        return list(set(suggestions))
    
    def generate_knowledge_base(
        self,
        n_scenarios: int = 10000,
        output_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """Generate a complete knowledge base.
        
        Args:
            n_scenarios: Number of scenarios to generate
            output_path: Optional path to save the knowledge base
            
        Returns:
            List of scenario dictionaries
        """
        scenarios = []
        
        for _ in range(n_scenarios):
            scenario = self.generate_scenario()
            scenarios.append(scenario)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            kb_data = {
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "n_scenarios": n_scenarios,
                "suggestion_labels": SUGGESTION_LABELS,
                "scenarios": scenarios,
            }
            
            with open(output_path, "w") as f:
                json.dump(kb_data, f, indent=2)
        
        return scenarios
    
    def get_label_distribution(self, scenarios: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get the distribution of labels in the scenarios.
        
        Args:
            scenarios: List of scenario dictionaries
            
        Returns:
            Dictionary mapping label to count
        """
        counts = {label: 0 for label in SUGGESTION_LABELS}
        
        for scenario in scenarios:
            for label in scenario["labels"]:
                counts[label] = counts.get(label, 0) + 1
        
        return counts
    
    def generate_edge_cases(self) -> List[Dict[str, Any]]:
        """Generate edge case scenarios for robustness.
        
        Returns:
            List of edge case scenarios
        """
        edge_cases = []
        
        edge_cases.append({
            "features": {
                "n_samples": 10,
                "n_features": 2,
                "missing_pct_max": 0.0,
                "imbalance_ratio": 1.0,
                "accuracy": 0.5,
                "f1_score": 0.33,
                "precision_recall_gap": 0.0,
            },
            "labels": ["COLLECT_MORE_DATA", "TRY_SIMPLE_MODELS"],
        })
        
        edge_cases.append({
            "features": {
                "n_samples": 100000,
                "n_features": 1000,
                "missing_pct_max": 0.0,
                "imbalance_ratio": 1.0,
                "accuracy": 0.99,
                "f1_score": 0.99,
                "precision_recall_gap": 0.0,
            },
            "labels": ["CROSS_VALIDATION"],
        })
        
        edge_cases.append({
            "features": {
                "n_samples": 100,
                "n_features": 500,
                "missing_pct_max": 0.7,
                "imbalance_ratio": 100.0,
                "accuracy": 0.55,
                "f1_score": 0.3,
                "precision_recall_gap": 0.4,
            },
            "labels": [
                "COLLECT_MORE_DATA",
                "DIMENSIONALITY_REDUCTION",
                "HANDLE_MISSING_VALUES",
                "SMOTE_IMBALANCE",
                "CLASS_WEIGHTS",
                "FEATURE_ENGINEERING",
            ],
        })
        
        edge_cases.append({
            "features": {
                "n_samples": 5000,
                "n_features": 20,
                "missing_pct_max": 0.0,
                "imbalance_ratio": 1.0,
                "accuracy": 0.92,
                "f1_score": 0.91,
                "precision_recall_gap": 0.02,
            },
            "labels": ["HYPERPARAMETER_TUNING", "CROSS_VALIDATION"],
        })
        
        return edge_cases
