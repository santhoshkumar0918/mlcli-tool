# mlcli Demo Guide

A step-by-step walkthrough to run a complete ML workflow from scratch.

---

## Prerequisites

> ⚠️ **All commands must be run from the project root:**
> ```bash
> cd /home/santhoshkumar/Desktop/learn/mlcli-tool
> ```

```bash
# 1. Go to the project root (IMPORTANT — not a subfolder like demo_artifacts/)
cd /home/santhoshkumar/Desktop/learn/mlcli-tool

# 2. Create and activate virtual environment (fish shell)
python -m venv .venv
source .venv/bin/activate.fish

# 3. Install the package (must be in the folder that contains pyproject.toml)
pip install -e .
```

---

## Step 0 — Confirm the CLI works

```bash
mlcli --help
```

Expected output: lists all 10 commands (init, preprocess, train, evaluate, suggest, predict, package, deploy, monitor, rollback).

---

## Step 1 — Initialize a new project

> `mlcli init --name <project-name>` creates a new subfolder with the project name in the current directory.

```bash
mlcli init --name my-demo-project --description "Demo ML project" --plugin tabular
```

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--name` | `-n` | Project name | prompted |
| `--description` | `-d` | Short description | optional |
| `--plugin` | `-p` | Template: `tabular`, `chatbot`, `image-classification` | `tabular` |
| `--force` | `-f` | Overwrite existing config | false |

This creates a project folder with:
```
my-demo-project/
├── mlcli.yaml          ← main config
├── README.md
├── requirements.txt
├── data/raw/
├── data/processed/
├── models/
├── notebooks/
├── reports/
└── src/
```

---

## Step 2 — Prepare your data

> ⚠️ After `mlcli init`, **`cd` into your project folder first**, then run all remaining commands from there:
> ```bash
> cd my-demo-project
> ```

Drop your CSV into `data/raw/`. For a quick test, run this command (copy only the line starting with `python`, not the surrounding text):

```bash
python -c "open('data/raw/data.csv','w').write('age,income,education,employed,churn\n25,50000,Bachelors,yes,1\n45,80000,Masters,no,0\n35,60000,Bachelors,yes,1\n29,48000,HS,no,0\n52,120000,PhD,yes,1\n41,72000,Masters,yes,0\n30,55000,Bachelors,no,1\n38,70000,Masters,yes,0\n48,90000,PhD,no,0\n33,52000,HS,yes,1\n')"
```

Verify it was created:
```bash
ls data/raw/
head data/raw/data.csv
```

---

## Step 3 — Preprocess the data

```bash
mlcli preprocess \
  --input  data/raw/data.csv \
  --target churn \
  --output data/processed
```

| Flag | Short | Description | Required |
|------|-------|-------------|----------|
| `--input` | `-i` | Path to input CSV | ✅ yes |
| `--output` | `-o` | Output directory | no (uses config) |
| `--target` | `-t` | Target column name | no (uses config) |
| `--analyze-only` | | Only profile data, skip preprocessing | no |

**Artifacts saved in `data/processed/`:**
- `train.csv` — training split
- `test.csv` — test split
- `preprocessor.pkl` — fitted sklearn ColumnTransformer
- `preprocessing_pipeline.pkl` — full pipeline bundle (features + metadata)
- `data_profile.json` — dataset analysis report
- `feature_names.json` — feature list

**Analyze-only mode (no artifacts written):**
```bash
mlcli preprocess -i data/raw/data.csv -t churn --analyze-only
```

---

## Step 4 — Train models

```bash
mlcli train \
  --train-data data/processed/train.csv \
  --test-data  data/processed/test.csv \
  --target     churn \
  --output     models
```

| Flag | Short | Description |
|------|-------|-------------|
| `--train-data` | `-t` | Training CSV path |
| `--test-data` | | Test CSV path |
| `--target` | | Target column name |
| `--output` | `-o` | Directory to save model files |

**Trains multiple algorithms and compares:**
- Logistic Regression
- Random Forest
- Support Vector Machine (SVC)
- K-Nearest Neighbors
- Gaussian Naive Bayes
- XGBoost

**Artifacts saved in `models/`:**
- `best_model.pkl` — best model by CV score
- `all_models/` — individual models
- `training_summary.json` — scores and metadata

**With verbose output:**
```bash
mlcli --verbose train -t data/processed/train.csv --target churn -o models
```

---

## Step 5 — Evaluate the model

```bash
mlcli evaluate \
  --test-data data/processed/test.csv \
  --model     models/best_model.pkl \
  --target    churn \
  --output    reports
```

| Flag | Short | Description |
|------|-------|-------------|
| `--test-data` | `-t` | Test CSV path |
| `--model` | `-m` | Trained model `.pkl` file |
| `--target` | | Target column name |
| `--output` | `-o` | Directory for evaluation reports |

**Artifacts saved in `reports/`:**
- `evaluation_report.json` — accuracy, precision, recall, F1, ROC-AUC
- `figures/confusion_matrix.png`
- `figures/roc_curve.png`

---

## Step 6 — Get AI-powered suggestions

```bash
mlcli suggest \
  --data-profile data/processed/data_profile.json \
  --evaluation   reports/evaluation_report.json \
  --training     models/training_summary.json \
  --top-k        5
```

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--data-profile` | | Path to `data_profile.json` | optional |
| `--evaluation` | | Path to `evaluation_report.json` | optional |
| `--training` | | Path to `training_summary.json` | optional |
| `--ml` / `--rules` | | Use ML engine or rule-based fallback | `--ml` |
| `--top-k` | `-k` | Number of suggestions to display | `5` |

**Example output:**
```
Suggestions:
  1. Try StandardScaler on numeric features (confidence: 0.91)
  2. Increase n_estimators for RandomForest (confidence: 0.87)
  3. Apply SMOTE to handle class imbalance (confidence: 0.83)
  ...
```

---

## Step 7 — Make predictions on new data

```bash
mlcli predict \
  --input data/raw/data.csv \
  --model models/best_model.pkl \
  --output reports/predictions.csv
```

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--input` | `-i` | CSV file to predict on | ✅ required |
| `--output` | `-o` | Output CSV file | auto-named |
| `--model` | `-m` | Model `.pkl` file | uses config |
| `--probabilities` | `-p` | Include class probabilities | false |
| `--batch-size` | `-b` | Rows per batch (large files) | `1000` |

**With probabilities:**
```bash
mlcli predict -i new_records.csv -o predictions.csv --probabilities
```

**Large dataset with batching:**
```bash
mlcli predict -i big_file.csv -o out.csv --batch-size 500
```

---

## Step 8 — Package the model (Phase 2)

```bash
mlcli package main
```

Packages the best model as a BentoML service for deployment.  
> Note: Requires BentoML installed. Currently in Phase 2 (stub).

---

## Step 9 — Deploy (Phase 2)

```bash
mlcli deploy main
```

Deploys the packaged BentoML service to a cloud provider.  
> Note: Currently in Phase 2 (stub).

---

## Step 10 — Monitor (Phase 2)

```bash
mlcli monitor main
```

View live metrics and logs for the deployed model.  
> Note: Currently in Phase 2 (stub).

---

## Step 11 — Rollback (Phase 2)

```bash
mlcli rollback main
```

Roll back to the previous deployed version.  
> Note: Currently in Phase 2 (stub).

---

## Global Options (apply to any command)

```bash
mlcli --verbose   <command>     # Enable detailed log output
mlcli --config    my.yaml <cmd> # Use a custom config file
mlcli --project-dir /path <cmd> # Override the project root directory
```

---

## Full end-to-end workflow (copy-paste ready)

> Run each line **one at a time** in your terminal. Do NOT copy the backtick fences.

```bash
# 1. Go to the mlcli-tool repo root and activate the environment
cd /home/santhoshkumar/Desktop/learn/mlcli-tool
source .venv/bin/activate.fish

# 2. Init a new project (creates my-demo-project/ folder here)
mlcli init -n my-demo-project -p tabular

# 3. cd INTO the new project folder — all commands below run from here
cd my-demo-project

# 4. Create sample data (single line, fish-safe)
python -c "open('data/raw/data.csv','w').write('age,income,education,employed,churn\n25,50000,Bachelors,yes,1\n45,80000,Masters,no,0\n35,60000,Bachelors,yes,1\n29,48000,HS,no,0\n52,120000,PhD,yes,1\n41,72000,Masters,yes,0\n30,55000,Bachelors,no,1\n38,70000,Masters,yes,0\n48,90000,PhD,no,0\n33,52000,HS,yes,1\n')"

# 5. Preprocess
mlcli preprocess -i data/raw/data.csv -t churn -o data/processed

# 6. Train
mlcli train -t data/processed/train.csv --test-data data/processed/test.csv --target churn -o models

# 7. Evaluate
mlcli evaluate -t data/processed/test.csv -m models/best_model.pkl --target churn -o reports

# 8. Suggest
mlcli suggest --data-profile data/processed/data_profile.json --evaluation reports/evaluation_report.json --training models/training_summary.json -k 5

# 9. Predict
mlcli predict -i data/raw/data.csv -m models/best_model.pkl -o reports/predictions.csv --probabilities
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `mlcli: command not found` | Run `source .venv/bin/activate.fish` first |
| `No module named 'click'` | Run `pip install --force-reinstall click` |
| `No module named 'packaging'` | Run `pip install --force-reinstall packaging` |
| `Pipeline not found` error in predict | Run `preprocess` step first to generate `preprocessing_pipeline.pkl` |
| `Target column not found` | Check that `--target` matches the exact column name in your CSV |
| `mlcli.yaml not found` | Run `mlcli init` first, or use `--config` to point to your yaml |
