"""Training script for Meta-ML Suggestion Engine.

This script generates synthetic training data based on ML expert heuristics
and trains a multi-label classifier for generating ML improvement suggestions.

Usage:
    python -m mlcli.core.suggestion_model.train [--output-dir PATH] [--n-samples N]

The resulting model will be used by `mlcli suggest` to provide
AI-powered recommendations.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer


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


class KnowledgeBaseGenerator:
    """Generate synthetic ML scenarios with expert-labeled suggestions.
    
    Uses domain knowledge to create realistic training scenarios:
    - Andrew Ng's ML advice
    - Common ML pitfalls
    - Production ML best practices
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_scenario(self) -> Dict[str, Any]:
        n_samples = random.choice([20, 50, 100, 500, 1000, 5000, 10000, 50000, 100000])
        n_features = random.choice([2, 5, 10, 20, 50, 100, 200, 500])
        missing_pct_max = round(random.uniform(0, 0.7), 3)
        imbalance_ratio = random.choice([1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0])
        base_accuracy = self._simulate_base_accuracy(n_samples, n_features, imbalance_ratio)
        noise = random.gauss(0, 0.05)
        accuracy = max(0.1, min(0.99, base_accuracy + noise))
        f1_base = accuracy - random.uniform(0, 0.1)
        f1_score = max(0.1, min(accuracy, f1_base))
        precision = min(0.99, f1_score + random.uniform(-0.1, 0.1))
        recall = min(0.99, f1_score + random.uniform(-0.1, 0.1))
        precision_recall_gap = abs(precision - recall)
        suggestions = self._apply_expert_rules(
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
                "accuracy": accuracy,
                "f1_score": f1_score,
                "precision_recall_gap": precision_recall_gap,
            },
            "suggestions": suggestions,
        }
    
    def _simulate_base_accuracy(self, n_samples: int, n_features: int, imbalance_ratio: float) -> float:
        base = 0.75
        if n_samples < 100:
            base -= 0.15
        elif n_samples < 500:
            base -= 0.05
        elif n_samples > 10000:
            base += 0.05
        
        if n_features > 100:
            base -= 0.05
        
        if imbalance_ratio > 10:
            base -= 0.1
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
        
        if len(suggestions) == 0:
            suggestions.append("HYPERPARAMETER_TUNING")
        
        return list(set(suggestions))
    
    def generate_knowledge_base(self, n_scenarios: int = 10000) -> List[Dict[str, Any]]:
        scenarios = []
        for _ in range(n_scenarios):
            scenario = self.generate_scenario()
            scenarios.append(scenario)
        return scenarios


def extract_features(scenario: Dict[str, Any]) -> List[float]:
    features = scenario["features"]
    return [
        float(features["n_samples"]),
        float(features["n_features"]),
        float(features["missing_pct_max"]),
        float(features["imbalance_ratio"]),
        float(features["accuracy"]),
        float(features["f1_score"]),
        float(features["precision_recall_gap"]),
    ]


def train_model(
    scenarios: List[Dict[str, Any]],
    output_dir: Path,
    model_version: str = "v2",
) -> Dict[str, Any]:
    print(f"Preparing training data from {len(scenarios)} scenarios...")
    
    X = np.array([extract_features(s) for s in scenarios])
    
    mlb = MultiLabelBinarizer(classes=SUGGESTION_LABELS)
    y_lists = [s["suggestions"] for s in scenarios]
    y = mlb.fit_transform(y_lists)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label matrix shape: {y.shape}")
    print(f"Labels per sample (avg): {y.sum(axis=1).mean():.2f}")
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training multi-label classifier...")
    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    
    model = MultiOutputClassifier(base_model, n_jobs=-1)
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Train score: {train_score:.4f}")
    print(f"Test score: {test_score:.4f}")
    
    y_pred = model.predict(X_test)
    
    from sklearn.metrics import hamming_loss, f1_score as multilabel_f1
    hamming = hamming_loss(y_test, y_pred)
    f1 = multilabel_f1(y_test, y_pred, average="samples")
    
    print(f"Hamming loss: {hamming:.4f}")
    print(f"Sample F1 score: {f1:.4f}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"suggestion_model_{model_version}.pkl"
    
    artifacts = {
        "model": model,
        "mlb": mlb,
        "feature_names": [
            "n_samples",
            "n_features", 
            "missing_pct_max",
            "imbalance_ratio",
            "accuracy",
            "f1_score",
            "precision_recall_gap",
        ],
        "suggestion_labels": SUGGESTION_LABELS,
        "version": model_version,
        "created_at": datetime.now().isoformat(),
        "metrics": {
            "train_score": train_score,
            "test_score": test_score,
            "hamming_loss": hamming,
            "f1_score": f1,
        }
    }
    
    joblib.dump(artifacts, model_path, compress=3)
    print(f"Model saved to: {model_path}")
    
    kb_path = output_dir / "knowledge_base.json"
    with open(kb_path, "w") as f:
        json.dump(scenarios[:1000], f, indent=2)
    print(f"Sample knowledge base saved to: {kb_path}")
    
    return {
        "model_path": str(model_path),
        "knowledge_base_path": str(kb_path),
        "metrics": {
            "train_score": train_score,
            "test_score": test_score,
            "hamming_loss": hamming,
            "f1_score": f1,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Train Meta-ML Suggestion Model")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Output directory for model and data",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v2",
        help="Model version identifier",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Meta-ML Suggestion Engine Training")
    print("=" * 60)
    print()
    
    print(f"Generating {args.n_samples} training scenarios...")
    generator = KnowledgeBaseGenerator(seed=args.seed)
    scenarios = generator.generate_knowledge_base(args.n_samples)
    
    label_counts = {}
    for s in scenarios:
        for label in s["suggestions"]:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count} ({count/args.n_samples*100:.1f}%)")
    
    print()
    results = train_model(
        scenarios=scenarios,
        output_dir=args.output_dir,
        model_version=args.model_version,
    )
    
    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model: {results['model_path']}")
    print(f"Metrics:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
