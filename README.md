# ML Assistant CLI

**From dataset to deployed API in minutes**

ML Assistant CLI is a developer-first command-line tool that unifies the entire ML lifecycle - from data preprocessing to cloud deployment - with AI-guided suggestions and one-click deployments.

## Features

- **End-to-end ML workflow** in a single CLI
- **AI-powered suggestions** using Meta-ML recommendation engine
- **Artifact tracking** with lineage and integrity verification
- **Privacy-first telemetry** for continuous improvement
- **BentoML integration** for reproducible model packaging
- **Multi-cloud deployment** (BentoCloud, Azure ML, AWS SageMaker)
- **Production-ready** with monitoring, rollbacks, and traffic management
- **Beginner-friendly** with sensible defaults and clear guidance

## Quick Start

### Installation

```bash
# Install from PyPI
pip install ml-assistant-cli

# Or with cloud support
pip install ml-assistant-cli[cloud]

# Verify installation
mlcli --help
```

### Initialize a new ML project

```bash
mlcli init --name my-ml-project
cd my-ml-project
```

### Process your data

```bash
# Add your dataset to data/raw/
mlcli preprocess --input data/raw/your_data.csv --target target_column
```

### Train models

```bash
mlcli train
```

### Evaluate and get AI suggestions

```bash
mlcli evaluate
mlcli suggest              # ML-powered suggestions
mlcli suggest --rules      # Rule-based fallback
mlcli suggest --top-k 10   # Show top 10 suggestions
```

### Make predictions

```bash
mlcli predict --input new_data.csv --output predictions.csv
```

### Deploy to cloud

```bash
mlcli package
mlcli deploy --provider bentocloud
mlcli monitor
```

## AI-Powered Suggestions

The `mlcli suggest` command uses a Meta-ML recommendation engine trained on expert ML heuristics:

- **19 suggestion types** covering data quality, model performance, and deployment
- **Confidence scores** with visualization
- **Actionable recommendations** with specific commands
- **Graceful fallback** to rule-based suggestions

### Suggestion Types

| Category | Suggestions |
|----------|-------------|
| Data | COLLECT_MORE_DATA, FEATURE_ENGINEERING, FEATURE_SELECTION, DIMENSIONALITY_REDUCTION |
| Imbalance | SMOTE_IMBALANCE, CLASS_WEIGHTS, STRATIFIED_SAMPLING |
| Performance | HYPERPARAMETER_TUNING, TRY_ENSEMBLE_MODELS, REGULARIZATION |
| Quality | HANDLE_MISSING_VALUES, OUTLIER_TREATMENT, DATA_AUGMENTATION |
| Training | CROSS_VALIDATION, EARLY_STOPPING, LEARNING_RATE_TUNING |

### Train the Suggestion Model

```bash
# Generate training data and train model
python -m mlcli.meta_ml.training --n-samples 10000
```

## Project Structure

```
my-ml-project/
├── data/
│   └── raw/           # Your datasets
├── models/            # Trained models
├── reports/           # Evaluation reports
├── .mlcli/            # MLCLI metadata
│   ├── artifact_registry.json    # Artifact tracking
│   └── telemetry/               # Event logs
├── mlcli.yaml         # Configuration
└── README.md
```

## Configuration

Customize your ML pipeline in `mlcli.yaml`:

```yaml
project_name: my-ml-project
description: My awesome ML project

data:
  target_column: target
  test_size: 0.2
  missing_value_strategy: auto
  scaling_strategy: standard
  imbalance_strategy: auto  # smote, class_weights, none

model:
  algorithms: [logistic_regression, random_forest, xgboost]
  hyperparameter_tuning: true
  cv_folds: 5
  class_weight: balanced

deployment:
  provider: bentocloud
  scaling_min: 1
  scaling_max: 3
  instance_type: cpu.2
```

## Artifact Tracking

MLCLI tracks all generated artifacts with unique IDs, checksums, and lineage:

```python
from mlcli.core.versioning import ArtifactTracker, ArtifactType

tracker = ArtifactTracker(project_dir)

# Register artifacts
profile_id = tracker.register(
    artifact_type=ArtifactType.DATA_PROFILE,
    file_path="data/processed/data_profile.json",
    metadata={"n_samples": 1000}
)

# Get lineage
lineage = tracker.get_lineage(model_id)
```

## Roadmap

### Phase 1: Local MVP

- [x] Project initialization
- [x] Data preprocessing and analysis
- [x] Model training with hyperparameter optimization
- [x] Model evaluation and metrics
- [x] AI-guided suggestions
- [x] Batch predictions
- [ ] BentoML packaging

### Phase 2: Cloud MVP

- [ ] BentoCloud deployment
- [ ] Model monitoring
- [ ] Deployment rollbacks

### Phase 3: Multi-Cloud

- [ ] Azure ML integration
- [ ] AWS SageMaker support
- [ ] Advanced deployment strategies
- [ ] CI/CD integration

## Architecture

```
mlcli/
├── commands/          # CLI commands
├── core/
│   ├── config.py      # Configuration management
│   ├── data.py        # Data processing
│   ├── models.py      # Model training
│   ├── schemas/       # Pydantic validation schemas
│   ├── versioning/    # Artifact tracking
│   ├── telemetry/     # Privacy-first telemetry
│   └── suggestion_model/  # ML suggestion engine
├── meta_ml/           # Meta-ML recommendation engine
└── plugins/           # Plugin system (tabular, chatbot, image)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- Documentation: https://mlcli.readthedocs.io
- Issue Tracker: https://github.com/mlcli/mlcli/issues
- Discussions: https://github.com/mlcli/mlcli/discussions
