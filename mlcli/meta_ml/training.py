"""Model training pipeline for Meta-ML Suggestion Engine.

Usage:
    python -m mlcli.meta_ml.training --n-samples 10000 --output-dir data/meta_ml
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .knowledge_base import KnowledgeBaseGenerator, SUGGESTION_LABELS


def prepare_training_data(scenarios: List[Dict[str, Any]]) -> tuple:
    """Convert scenarios to training matrices.
    
    Args:
        scenarios: List of scenario dictionaries
        
    Returns:
        Tuple of (X, y) numpy arrays
    """
    X = []
    y = []
    
    for scenario in scenarios:
        features = scenario["features"]
        labels = scenario["labels"]
        
        feature_vector = [
            float(features["n_samples"]),
            float(features["n_features"]),
            float(features["missing_pct_max"]),
            float(features["imbalance_ratio"]),
            float(features["accuracy"]),
            float(features["f1_score"]),
            float(features["precision_recall_gap"]),
        ]
        
        label_vector = [1 if label in labels else 0 for label in SUGGESTION_LABELS]
        
        X.append(feature_vector)
        y.append(label_vector)
    
    return np.array(X), np.array(y)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "random_forest",
    calibrate: bool = True,
) -> Dict[str, Any]:
    """Train the Meta-ML model.
    
    Args:
        X: Feature matrix
        y: Label matrix
        model_type: Type of model to train
        calibrate: Whether to calibrate probabilities
        
    Returns:
        Dictionary with model and metrics
    """
    from sklearn.model_selection import train_test_split
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import hamming_loss, f1_score
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training data: X={X_train.shape}, y={y_train.shape}")
    print(f"Test data: X={X_test.shape}, y={y_test.shape}")
    
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            base_model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            )
        except ImportError:
            print("XGBoost not available, using RandomForest")
            from sklearn.ensemble import RandomForestClassifier
            base_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
    else:
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    
    model = MultiOutputClassifier(base_model, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    hamming = hamming_loss(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="samples")
    
    print(f"\nTraining Results:")
    print(f"  Train score: {train_score:.4f}")
    print(f"  Test score: {test_score:.4f}")
    print(f"  Hamming loss: {hamming:.4f}")
    print(f"  F1 score: {f1:.4f}")
    
    mlb = MultiLabelBinarizer(classes=SUGGESTION_LABELS)
    mlb.fit([SUGGESTION_LABELS])
    
    return {
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
        "version": "v2",
        "model_type": model_type,
        "metrics": {
            "train_score": train_score,
            "test_score": test_score,
            "hamming_loss": hamming,
            "f1_score": f1,
        }
    }


def save_model(artifacts: Dict[str, Any], output_path: Path) -> None:
    """Save model artifacts to disk.
    
    Args:
        artifacts: Dictionary with model and metadata
        output_path: Path to save the model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import joblib
        joblib.dump(artifacts, output_path, compress=3)
    except ImportError:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(artifacts, f)
    
    print(f"\nModel saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Meta-ML Suggestion Model")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/meta_ml"),
        help="Output directory",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost"],
        help="Model type",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Meta-ML Suggestion Engine Training")
    print("=" * 60)
    
    print(f"\nGenerating {args.n_samples} training scenarios...")
    generator = KnowledgeBaseGenerator(seed=args.seed)
    scenarios = generator.generate_knowledge_base(args.n_samples)
    
    label_dist = generator.get_label_distribution(scenarios)
    print("\nLabel distribution:")
    for label, count in sorted(label_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {label}: {count} ({count/args.n_samples*100:.1f}%)")
    
    print("\nPreparing training data...")
    X, y = prepare_training_data(scenarios)
    
    print("\nTraining model...")
    artifacts = train_model(X, y, model_type=args.model_type)
    
    model_path = args.output_dir / "suggestion_model_v2.pkl"
    save_model(artifacts, model_path)
    
    kb_path = args.output_dir / "knowledge_base.json"
    generator.generate_knowledge_base(min(1000, args.n_samples), kb_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Knowledge base: {kb_path}")
    print(f"Metrics: {artifacts['metrics']}")


if __name__ == "__main__":
    main()
