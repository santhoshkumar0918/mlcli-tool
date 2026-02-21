# MLCLI: AI-Powered Suggestions (Phase 3) - Self-Service Guide

This guide explains how the AI Suggestion system (Meta-Model) works and how you can manage or extend it on your own.

---

## 🏗️ 1. Architecture Overview

The system uses a **Multi-Label Classifier** to analyze your ML pipeline's state and suggest improvements.

### Component Map:
- **`mlcli/core/suggestion_model/data/training_data.json`**: The "Experience Base". Contains examples of system states mapped to expert solutions.
- **`mlcli/core/suggestion_model/features.py`**: The "Translator". Converts JSON reports into numeric vectors (n_samples, accuracy, etc.).
- **`mlcli/core/suggestion_model/training.py`**: The "Teacher". Trains a Random Forest model on the data.
- **`mlcli/core/suggestion_model/model.py`**: The "Brain". Loads the trained model and performs inference (prediction) with confidence scores.
- **`mlcli/commands/suggest_cmd.py`**: The "Interface". Calls the brain and displays results in the terminal.

---

## 🚀 2. How to Add New Expert Knowledge

If you find a new pattern (e.g., "Models always fail when feature count > 500"), you can teach it to the CLI:

1. **Edit `training_data.json`**:
   Add a new entry to the list:
   ```json
   {
     "features": {
       "n_samples": 1000,
       "n_features": 600,
       "missing_pct_max": 0.05,
       "imbalance_ratio": 1.1,
       "accuracy": 0.6,
       "f1_score": 0.58,
       "precision_recall_gap": 0.02
     },
     "suggestions": ["FEATURE_SELECTION", "DIMENSIONALITY_REDUCTION"]
   }
   ```

2. **Run the trainer**:
   ```bash
   python3 mlcli/core/suggestion_model/training.py
   ```
   *Note: Ensure you use the virtual environment's python.*

---

## 🧠 3. How to Add a New Suggestion Type

If you want the CLI to suggest something entirely new (e.g., "GPU_ACCELERATION"):

1. **Update `training_data.json`**: Use the new string in your "suggestions" lists.
2. **Update `model.py`**:
   In the `MLSuggestionEngine` class, update the mapping methods:
   - `_map_to_issue()`: Add a human-readable description of the problem.
   - `_map_to_impact()`: Add a human-readable description of why this helps.
3. **Retrain**: Run the `training.py` script as shown above.

---

## 📊 4. The Workflow Technical Flow

When you run `mlcli suggest`:
1. It looks for `data_profile.json` and `evaluation_report.json`.
2. `features.py` extracts 7 key numbers (Samples, Features, Imbalance, etc.).
3. `suggestion_model.pkl` is loaded. This is a "MultiOutputClassifier" because one state can have multiple suggestions.
4. The system calculates a probability for **every** possible suggestion.
5. If the probability is **> 30%**, it is displayed as a "Match" in your terminal.

---

## 🛠️ 5. Maintenance Tips

- **Confidence Scores**: If the CLI is giving too many suggestions, increase the threshold in `model.py` (line 46 of `get_suggestions`).
- **Feature Names**: If you add a new numeric metric to the reports, add it to `features.py` so the model can see it.

---

## 📜 Summary of Files You Can Edit:
- `training_data.json`: Add knowledge here.
- `model.py`: Add human-readable mapping text here.
- `training.py`: Change the ML algorithm (e.g., use XGBoost instead of Random Forest) here.
