# ML Assistant CLI 🚀

**From dataset to deployed API in minutes**

ML Assistant CLI is a developer-first command-line tool that unifies the entire ML lifecycle - from data preprocessing to cloud deployment - with AI-guided suggestions and one-click deployments.

## ✨ Features

- **End-to-end ML workflow** in a single CLI
- **AI-guided suggestions** for data quality and model improvements
- **BentoML integration** for reproducible model packaging
- **Multi-cloud deployment** (BentoCloud, Azure ML, AWS SageMaker HyperPod)
- **Production-ready** with monitoring, rollbacks, and traffic management
- **Beginner-friendly** with sensible defaults and clear guidance

## 🚀 Quick Start

### Installation

```bash
pip install mlcli
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

### Evaluate and get suggestions

```bash
mlcli evaluate
mlcli suggest
```

### Make predictions

```bash
mlcli predict --input new_data.csv --output predictions.csv
```

### Deploy to cloud (coming soon)

```bash
mlcli package
mlcli deploy --provider bentocloud
mlcli monitor
```

## 📁 Project Structure

```
my-ml-project/
├── data/
│   ├── raw/          # Original datasets
│   ├── processed/    # Cleaned data
│   └── external/     # External datasets
├── models/           # Trained models
├── reports/          # Analysis reports
├── deployments/      # Deployment configs
├── mlcli.yaml       # Configuration
└── README.md
```

## ⚙️ Configuration

Customize your ML pipeline in `mlcli.yaml`:

```yaml
project_name: my-ml-project
description: My awesome ML project

data:
  target_column: target
  test_size: 0.2
  missing_value_strategy: auto
  scaling_strategy: standard

model:
  algorithms: [logistic_regression, random_forest, xgboost]
  hyperparameter_tuning: true
  cv_folds: 5

deployment:
  provider: bentocloud
  scaling_min: 1
  scaling_max: 3
  instance_type: cpu.2
```

## 🎯 Roadmap

### Phase 1: Local MVP ✅

- [x] Project initialization
- [x] Data preprocessing and analysis
- [ ] Model training with hyperparameter optimization
- [ ] Model evaluation and metrics
- [ ] AI-guided suggestions
- [ ] Batch predictions
- [ ] BentoML packaging

### Phase 2: Cloud MVP

- [ ] BentoCloud deployment
- [ ] Model monitoring
- [ ] Deployment rollbacks

### Phase 3: Multi-Cloud

- [ ] Azure ML integration
- [ ] AWS SageMaker HyperPod support
- [ ] Advanced deployment strategies
- [ ] CI/CD integration

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🆘 Support

- 📖 [Documentation](https://mlcli.readthedocs.io)
- 🐛 [Issue Tracker](https://github.com/mlcli/mlcli/issues)
- 💬 [Discussions](https://github.com/mlcli/mlcli/discussions)

---

**Built with ❤️ for the ML community**
