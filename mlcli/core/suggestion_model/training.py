"""Training script for the suggestion meta-model with raw metric logging."""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_suggestion_model(data_path: Path, output_path: Path, history_path: Path):
    """Train the suggestion engine model and log raw performance metrics."""
    
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
    
    # Split for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # We will simulate "iterations" by using an incremental learning approach
    # (Typical for Random Forest to show a learning curve)
    history = []
    n_steps = 20
    step_size = max(1, len(X_train) // n_steps)
    
    print(f"Starting raw metric collection over {n_steps} training steps...")
    
    for i in range(1, n_steps + 1):
        # Take a subset of training data
        subset_size = min(i * step_size, len(X_train))
        X_subset = X_train[:subset_size]
        y_subset = y_train[:subset_size]
        
        # Train model
        base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        model = MultiOutputClassifier(base_model)
        model.fit(X_subset, y_subset)
        
        # Evaluate on full test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics (macro average for multi-label)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        history.append({
            "step": i,
            "n_samples": subset_size,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
        print(f"Step {i}/{n_steps} - Size: {subset_size} - Acc: {acc:.4f}")

    # Save history to CSV
    history_df = pd.DataFrame(history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_path, index=False)
    print(f"✓ Raw training history saved to {history_path}")
    
    # Final train on all data
    final_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    final_model.fit(X, y)
    
    # Save model and binarizer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'mlb': mlb
        }, f)
        
    print(f"✓ Final suggestion model trained and saved to {output_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    data_file = base_dir / "data" / "training_data.json"
    model_file = base_dir / "data" / "suggestion_model.pkl"
    history_file = base_dir / "data" / "training_history.csv"
    train_suggestion_model(data_file, model_file, history_file)
