# ML Project

This project was initialized with ML Assistant CLI.

## Getting Started

1. Add your dataset to `data/raw/`
2. Run preprocessing: `mlcli preprocess --input data/raw/your_data.csv`
3. Train models: `mlcli train`
4. Evaluate performance: `mlcli evaluate`
5. Get suggestions: `mlcli suggest`
6. Make predictions: `mlcli predict --input new_data.csv`

## Project Structure

```
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned and preprocessed data
│   └── external/     # External datasets
├── models/           # Trained models and artifacts
├── notebooks/        # Jupyter notebooks for exploration
├── reports/          # Analysis reports and figures
├── src/              # Source code
├── tests/            # Unit tests
├── deployments/      # Deployment configurations
└── logs/             # Application logs
```

## Configuration

Edit `mlcli.yaml` to customize your ML pipeline settings.
