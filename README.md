# ML Project (Tabular Data)

This project was initialized with **ML Assistant CLI** using the **tabular** plugin.

## Quick Start

```bash
# 1. Add your dataset
cp /path/to/your/data.csv data/raw/

# 2. Preprocess data
mlcli preprocess --input data/raw/data.csv --target target_column

# 3. Train models
mlcli train

# 4. Evaluate performance
mlcli evaluate

# 5. Get AI suggestions
mlcli suggest

# 6. Make predictions
mlcli predict --input new_data.csv --output predictions.csv
```

## Project Structure

```
├── data/
│   └── raw/           # Your raw datasets
├── mlcli.yaml         # ML pipeline configuration
└── README.md
```

Additional directories are created automatically:
- `data/processed/` - Preprocessed data
- `models/` - Trained models
- `reports/` - Evaluation reports

## Configuration

Edit `mlcli.yaml` to customize:
- Target column name
- Model algorithms to try
- Preprocessing strategies
- Hyperparameter tuning settings

## Supported Algorithms

- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
- Support Vector Machine

## Next Steps

1. Add your dataset to `data/raw/`
2. Edit `mlcli.yaml` to set your target column
3. Run `mlcli preprocess` to prepare your data
