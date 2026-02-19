# MLCLI: Technical Demo & Onboarding Guide

Welcome to the **ML Assistant CLI** project! This guide is designed to help you and your teammates run a full end-to-end demo and understand the core workflow of our framework.

---

## Step 0: Initial Installation

For new teammates who have just cloned the repository:

```bash
# Clone the repository (if not already done)
git clone https://github.com/santhoshkumar0918/mlcli-tool.git
cd mlcli-tool

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -e .
```

---

## Step 1: Environment Activation

Before running any commands, ensure your virtual environment is active.

```bash
# Navigate to the tool directory
cd /path/to/mlcli-tool

# Activate the environment
source .venv/bin/activate
```

---

## Step 2: Project Initialization

We use a Plugin-Based system to scaffold different project types.

### Option A: Traditional Tabular ML
```bash
# Initialize a new project in a directory
mlcli --project-dir my_tabular_project init --name "Churn Prediction" --plugin tabular
```

### Option B: Chatbot (LLM/RAG)
```bash
mlcli --project-dir my_chatbot init --name "KnowledgeBot" --plugin chatbot
```

### Terminal Simulation: mlcli init
```
[cyan]Using plugin:[/cyan] LangChain-based chatbot with RAG (Retrieval-Augmented Generation)

[green]✓[/green] Initialized [bold]chatbot[/bold] project: [bold]KnowledgeBot[/bold]
[green]✓[/green] Configuration saved to: my_chatbot/mlcli.yaml
[green]✓[/green] Project structure created in: my_chatbot

Next steps:
1. Copy .env.example to .env and add your OPENAI_API_KEY
2. Add documents to data/knowledge_base/
3. Run python src/app.py to start the chatbot
```

---

## Step 3: The Tabular ML Workflow

If you initialized `my_tabular_project`, run the following flow:

```bash
cd my_tabular_project
```

### 1. Data Preprocessing
Cleans your dataset, handles missing values, and generates a data profile.
```bash
# Add a sample.csv to data/raw/ first
mlcli preprocess --input data/raw/sample.csv --target label
```

**What it does:**
- Loads and analyzes your data
- Handles missing values (imputation)
- Scales numerical features (StandardScaler)
- Encodes categorical features (OneHotEncoder)
- Splits into train/test sets
- **Saves preprocessing pipeline** for consistent predictions

**Output artifacts:**
- `data/processed/train.csv` - Training data
- `data/processed/test.csv` - Test data
- `data/processed/data_profile.json` - Data quality report
- `data/processed/preprocessing_pipeline.pkl` - **Fitted pipeline (NEW in Phase 3A)**

### 2. Model Training
Trains baseline models (logistic regression, random forest) and saves the best one.
```bash
mlcli train
```

### Terminal Simulation: mlcli train
```
[19:40:19] INFO     Training random_forest...                                                                                     
[19:40:37] INFO     Training complete. Best model: logistic_regression (score: 0.5667)                                            
⠧ Training complete!
                   Model Training Results                    
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Model                  ┃ CV Score ┃ Val Score ┃ Status    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│ logistic_regression ⭐ │ 0.5667   │ 0.5000    │ ✓ Success │
│ random_forest          │ 0.5667   │ 0.5000    │ ✓ Success │
└────────────────────────┴──────────┴───────────┴───────────┘

Training Summary:
• Task Type: Classification
• Models Trained: 2
• Best Model: logistic_regression
• Best Score: 0.5667
• Cross-Validation: 5 folds
• Scoring Metric: accuracy
```

**What it does:**
- Auto-detects task type (classification vs regression)
- Trains multiple algorithms in parallel
- Performs hyperparameter tuning (GridSearchCV)
- Evaluates with cross-validation
- Saves best model

**Output artifacts:**
- `models/best_model.pkl` - Best performing model
- `models/all_models/*.pkl` - All trained models
- `models/training_summary.json` - Training metrics

### 3. Evaluation
Generates a detailed performance report with metrics.
```bash
mlcli evaluate
```

### Terminal Simulation: mlcli evaluate
```
         Model Performance Metrics          
┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric    ┃ Value  ┃ Interpretation      ┃
┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ Accuracy  │ 0.5000 │ Poor                │
│ Precision │ 0.2500 │ Low precision       │
│ Recall    │ 0.5000 │ Low recall          │
│ F1 Score  │ 0.3333 │ Poor balance        │
└───────────┴────────┴─────────────────────┘
[19:41:46] INFO     Evaluation report saved to: reports/evaluation_report.json
```

**What it does:**
- Loads test data and best model
- Computes comprehensive metrics
- Generates confusion matrix (classification)
- Creates visualization plots
- Provides human-readable interpretations

**Output artifacts:**
- `reports/evaluation_report.json` - Full metrics
- `reports/figures/confusion_matrix.png` - Visual analysis
- `reports/figures/roc_curve.png` - ROC curve (classification)

### 4. AI-Powered Suggestions
Analyses your results and gives you technical advice on how to improve.
```bash
mlcli suggest
```

### Terminal Simulation: mlcli suggest
```
🤖 Loading Meta-ML recommendation engine...
✓ Meta-ML engine loaded

🤖 AI-Powered Improvement Suggestions

1. Data Size
   Confidence: ████████████████████ 92%
   High confidence — Strongly recommended
   
   Action:
   # Add more samples to data/raw/ and re-run mlcli preprocess

2. Model Performance
   Confidence: ███████████████░░░░░ 78%
   Medium confidence — Recommended if relevant
   
   Action:
   # Update mlcli.yaml:
   model:
     algorithms: [xgboost, random_forest, gradient_boosting]

Top Priority Actions:
1. Data Size (High)
   Issue: Small dataset (20 samples)
   Suggestion: Consider collecting more data or using data augmentation techniques
   Impact: More data typically leads to better model performance

2. Model Performance (High)
   Issue: Low classification performance (0.567)
   Suggestion: Try advanced algorithms (XGBoost, Neural Networks) or ensemble methods
   Impact: Significant improvement in model accuracy
```

**What it does:**
- Loads all project artifacts (data profile, training summary, evaluation)
- Uses Meta-ML engine (if available) or rule-based heuristics
- Generates actionable recommendations with confidence scores
- Prioritizes suggestions by impact

**Output artifacts:**
- `reports/suggestions.json` - All recommendations
- `.mlcli/telemetry/events.jsonl` - Anonymized usage data (Phase 3A.4)

### 5. Prediction (Inference)
Run your trained model on new, unseen data.
```bash
mlcli predict --input data/raw/new_data.csv --output predictions/results.csv
```

### Terminal Simulation: mlcli predict
```
Loading model from: models/best_model.pkl
Loading preprocessing pipeline from: data/processed/preprocessing_pipeline.pkl
✓ Preprocessing pipeline loaded: data/processed/preprocessing_pipeline.pkl
  Features: 10 | Target: label

[19:51:01] INFO     ✓ Preprocessing pipeline loaded                    
Making predictions...

📊 Prediction Summary
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric            ┃ Value   ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Input File        │ new_data.csv │
│ Output File       │ predictions/results.csv │
│ Total Predictions │ 100     │
│ Task Type         │ Classification │
└───────────────────┴─────────┘

         Prediction Distribution          
┏━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Class ┃ Count ┃ Percentage ┃
┡━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ 0     │ 45    │ 45.0%      │
│ 1     │ 55    │ 55.0%      │
└───────┴───────┴────────────┘

✓ Predictions saved to: predictions/results.csv

Next steps:
1. Review the predictions in the output file
2. Use mlcli evaluate if you have ground truth labels
3. Consider mlcli package for deployment if satisfied with results
```

**What it does:**
- **Loads saved preprocessing pipeline** (ensures same transformations as training)
- Applies transformations to new data
- Generates predictions
- Includes confidence scores (classification with `--probabilities`)
- Handles large datasets with batching

**Output artifacts:**
- `predictions/results.csv` - Predictions with original data
- `predictions/results.csv` (with `--probabilities`) - Includes confidence scores

---

## Step 4: The Chatbot (RAG) Workflow

For projects initialized with the chatbot plugin:

```bash
cd my_chatbot

# Setup environment
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your_key_here

# Add your knowledge base documents
cp your_docs.txt data/knowledge_base/

# Start the chatbot
python src/app.py
```

This launches a Gradio interface at `http://localhost:7860` where you can:
- Ask questions about your documents
- Get AI-powered responses with source citations
- Test RAG retrieval quality

---

## Teammate Quickstart Summary

```bash
# 1. Setup (once)
cd mlcli-tool
source .venv/bin/activate
pip install -e .

# 2. Initialize project
mlcli init --plugin tabular

# 3. Full ML Pipeline
mlcli preprocess --input data.csv --target label
mlcli train
mlcli evaluate
mlcli suggest
mlcli predict --input new_data.csv

# 4. View results
cat reports/evaluation_report.json
cat reports/suggestions.json
cat predictions/predictions_new_data.csv
```

**TIP:** Use the `-h` flag on any command to see detailed help!

```bash
mlcli --help
mlcli init --help
mlcli preprocess --help
mlcli train --help
```

---

## Advanced Usage

### Custom Configuration

Edit `mlcli.yaml` to customize behavior:

```yaml
project_name: my-ml-project
version: 0.1.0

data:
  target_column: label
  test_size: 0.2
  missing_value_strategy: mean  # auto, drop, mean, median, mode
  scaling_strategy: standard     # standard, minmax, robust, none
  encoding_strategy: onehot      # onehot, label, target

model:
  algorithms:
    - logistic_regression
    - random_forest
    - xgboost
  hyperparameter_tuning: true
  cv_folds: 5
  scoring_metric: accuracy       # accuracy, f1, precision, recall
  random_state: 42

deployment:
  provider: bentocloud           # Coming in Phase 2
  scaling_min: 1
  scaling_max: 3
```

### Batch Predictions

For large datasets:

```bash
mlcli predict --input large_data.csv --batch-size 10000
```

### Include Probabilities

For classification tasks:

```bash
mlcli predict --input data.csv --probabilities
```

Output includes confidence scores:
```csv
feature1,feature2,prediction,probability_class_0,probability_class_1,confidence
5.1,3.5,1,0.23,0.77,0.77
```

---

## Phase 3A Improvements (Current)

### ✅ What's New

1. **Preprocessing Pipeline Persistence**
   - Pipeline now saved to `preprocessing_pipeline.pkl`
   - Predictions use exact same transformations as training
   - Eliminates train/test skew in production

2. **Improved Error Messages**
   - Clear guidance when pipeline is missing
   - Helpful hints for next steps
   - Better debugging information

3. **Reproducibility**
   - All artifacts versioned with timestamps
   - Config embedded in pipeline
   - Metadata for validation

### 🚧 Coming Soon (Phase 3B-D)

- **Schema Validation**: Pydantic models for all JSON artifacts
- **Artifact Tracking**: Full lineage tracking with checksums
- **Telemetry System**: Privacy-first usage analytics
- **Meta-ML Engine**: Learned recommendation model
- **Confidence Scores**: Calibrated probabilities for suggestions

---

## Troubleshooting

### "Preprocessing pipeline not found"

**Problem:** Running `mlcli predict` before preprocessing.

**Solution:**
```bash
mlcli preprocess --input data/raw/data.csv --target label
mlcli predict --input new_data.csv
```

### "Target column not specified"

**Problem:** No target column in config or command.

**Solution:**
```bash
# Option 1: Use flag
mlcli preprocess --input data.csv --target your_target_column

# Option 2: Edit mlcli.yaml
data:
  target_column: your_target_column
```

### "Model not found"

**Problem:** Running `mlcli predict` or `mlcli evaluate` before training.

**Solution:**
```bash
mlcli train
mlcli evaluate
mlcli predict --input new_data.csv
```

### Virtual Environment Not Activated

**Symptoms:** `mlcli: command not found`

**Solution:**
```bash
cd mlcli-tool
source .venv/bin/activate
```

---

## Project Structure Overview

After running the full pipeline, your project looks like:

```
my-ml-project/
├── mlcli.yaml                  # Configuration
├── data/
│   ├── raw/
│   │   └── sample.csv          # Original data
│   ├── processed/
│   │   ├── train.csv           # Training split
│   │   ├── test.csv            # Test split
│   │   ├── data_profile.json   # Data analysis
│   │   └── preprocessing_pipeline.pkl  # ⭐ Fitted pipeline
│   └── external/
├── models/
│   ├── best_model.pkl          # Best model
│   ├── training_summary.json   # Training metrics
│   └── all_models/
│       ├── logistic_regression.pkl
│       └── random_forest.pkl
├── reports/
│   ├── evaluation_report.json  # Full metrics
│   ├── suggestions.json        # AI recommendations
│   └── figures/
│       ├── confusion_matrix.png
│       └── roc_curve.png
├── predictions/
│   └── predictions_new_data.csv
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
└── src/
    ├── train.py
    └── data_loader.py
```

---

## Contributing

Found a bug or have a feature request? Please:

1. Check existing issues: https://github.com/santhoshkumar0918/mlcli-tool/issues
2. Open a new issue with detailed description
3. For bugs, include:
   - Command that failed
   - Error message
   - `mlcli.yaml` content
   - Python version

---

## Next Steps

- **Explore the codebase:** `mlcli/core/`, `mlcli/commands/`, `mlcli/plugins/`
- **Read the product doc:** `docs/product_document.md`
- **Check the Phase 3 plan:** `docs/phase3_meta_ml_plan.md`
- **Read immediate actions:** `docs/IMMEDIATE_ACTIONS.md`

**Happy ML-ing! 🚀**
