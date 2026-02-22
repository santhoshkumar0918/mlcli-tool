# MLCLI

<div align="center">

**The ML Framework for Production**

[![PyPI version](https://badge.fury.io/py/ml-assistant-cli.svg)](https://badge.fury.io/py/ml-assistant-cli)
[![Python](https://img.shields.io/pypi/pyversions/ml-assistant-cli.svg)](https://pypi.org/project/ml-assistant-cli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Documentation](https://mlcli.readthedocs.io) | [Quick Start](#quick-start) | [Examples](#examples) | [Contributing](CONTRIBUTING.md)

</div>

---

## Overview

MLCLI is a production-grade ML framework that automates the entire machine learning lifecycle. From raw data to deployed API in minutes, not weeks.

| Feature | MLCLI | Traditional |
|---------|-------|-------------|
| Time to production | Minutes | Weeks |
| ML expertise required | Minimal | Advanced |
| Deployment ready | Yes | Manual setup |
| AI suggestions | Built-in | None |
| Artifact tracking | Automatic | Manual |

---

## Quick Start

### Install

```bash
pip install ml-assistant-cli
```

### Initialize

```bash
mlcli init my-project
cd my-project
```

### Train & Deploy

```bash
# Add your data to data/raw/
mlcli preprocess --input data/raw/data.csv --target label
mlcli train
mlcli evaluate
mlcli suggest
mlcli predict --input new_data.csv
```

---

## Features

### AI-Powered Suggestions

MLCLI analyzes your data and models, providing actionable recommendations:

```bash
$ mlcli suggest

1. HYPERPARAMETER_TUNING (78% confidence)
   Issue: Sub-optimal model configuration detected
   Action: Enable hyperparameter_tuning in mlcli.yaml

2. SMOTE_IMBALANCE (72% confidence)
   Issue: Class imbalance ratio of 8.5:1 detected
   Action: Set imbalance_strategy: smote in mlcli.yaml
```

### Plugin System

| Plugin | Use Case | Framework |
|--------|----------|-----------|
| `tabular` | Structured data | scikit-learn, XGBoost |
| `chatbot` | RAG applications | LangChain, OpenAI |
| `image-classification` | Computer vision | PyTorch, Lightning |

```bash
mlcli init --plugin chatbot my-chatbot
mlcli init --plugin image-classification my-cv-project
```

### Artifact Tracking

Every artifact is tracked with lineage and integrity:

```
.mlcli/
├── artifact_registry.json    # Full audit trail
└── telemetry/                # Usage analytics
```

---

## Configuration

Simple YAML configuration:

```yaml
# mlcli.yaml
project_name: my-project

data:
  target_column: label
  test_size: 0.2

model:
  algorithms: [xgboost, random_forest]
  hyperparameter_tuning: true
  cv_folds: 5

deployment:
  provider: bentocloud
  instance_type: cpu.2
```

---

## Examples

### Tabular Classification

```bash
mlcli init --plugin tabular classifier
cd classifier

cp ~/data.csv data/raw/
mlcli preprocess --input data/raw/data.csv --target label
mlcli train
mlcli evaluate
mlcli suggest
```

### RAG Chatbot

```bash
mlcli init --plugin chatbot assistant
cd assistant

cp ~/docs/* data/knowledge_base/
echo "OPENAI_API_KEY=sk-..." > .env
python src/app.py
```

### Image Classification

```bash
mlcli init --plugin image-classification classifier
cd classifier

# data/raw/class1/*.jpg
# data/raw/class2/*.jpg
python train.py
```

---

## Architecture

```
mlcli/
├── core/
│   ├── schemas/        # Pydantic validation
│   ├── versioning/     # Artifact tracking
│   ├── telemetry/      # Usage analytics
│   └── suggestion_model/   # Meta-ML engine
├── meta_ml/            # Recommendation system
├── plugins/            # Extensible plugins
└── commands/           # CLI commands
```

---

## API Reference

| Command | Description |
|---------|-------------|
| `mlcli init` | Initialize new project |
| `mlcli preprocess` | Process and validate data |
| `mlcli train` | Train ML models |
| `mlcli evaluate` | Evaluate model performance |
| `mlcli suggest` | Get AI-powered suggestions |
| `mlcli predict` | Make predictions |
| `mlcli package` | Package for deployment |
| `mlcli deploy` | Deploy to cloud |

---

## Roadmap

### v0.2.0 (Current)
- [x] AI-powered suggestions
- [x] Artifact tracking
- [x] Telemetry collection
- [x] Plugin system

### v0.3.0
- [ ] BentoML packaging
- [ ] Cloud deployment
- [ ] Model monitoring

### v0.4.0
- [ ] Azure ML integration
- [ ] AWS SageMaker integration
- [ ] Auto-scaling

---

## Contributing

```bash
git clone https://github.com/mlcli/mlcli.git
cd mlcli
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for ML Engineers, by ML Engineers**

[GitHub](https://github.com/mlcli/mlcli) | [PyPI](https://pypi.org/project/ml-assistant-cli/) | [Documentation](https://mlcli.readthedocs.io)

</div>
