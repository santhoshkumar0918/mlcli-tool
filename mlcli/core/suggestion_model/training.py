"""Training script for the suggestion meta-model."""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

def train_suggestion_model(data_path: Path, output_path: Path):
    """Train the suggestion engine model."""
    
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    # Prepare features
    X = []
    y_raw = []
    
    for item in data:
        features = item['features']
        X.append([
            features['n_samples'],
            features['n_features'],
            features['missing_pct_max'],
            features['imbalance_ratio'],
            features['accuracy'],
            features['f1_score'],
            features['precision_recall_gap']
        ])
        y_raw.append(item['suggestions'])
        
    X = np.array(X)
    
    # Binarize labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y_raw)
    
    # Train multi-label classifier
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    model = MultiOutputClassifier(base_model)
    model.fit(X, y)
    
    # Save model and binarizer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'mlb': mlb
        }, f)
        
    print(f"Suggestion model trained and saved to {output_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    data_file = base_dir / "data" / "training_data.json"
    model_file = base_dir / "data" / "suggestion_model.pkl"
    train_suggestion_model(data_file, model_file)
