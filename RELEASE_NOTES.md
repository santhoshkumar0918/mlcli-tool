# Release Notes - v0.3.0

**Release Date:** February 22, 2026

## Highlights

This release introduces **AI-Powered Suggestions** - a Meta-ML recommendation engine that analyzes your ML pipeline and provides actionable improvement suggestions.

## New Features

### AI-Powered Suggestions (Phase 3)

- **Meta-ML Engine**: Trained recommendation system with 19 suggestion types
- **Confidence Scores**: Visual confidence bars for each suggestion
- **Actionable Recommendations**: Specific commands and config changes
- **Graceful Fallback**: Rule-based suggestions when ML model unavailable

### Artifact Tracking (Phase 3A.4)

- **Unique IDs**: Every artifact gets `art-XXXX` identifier
- **Lineage Tracking**: Parent-child relationships between artifacts
- **Integrity Verification**: SHA256 checksums for all files
- **Audit Trail**: Full history in `.mlcli/artifact_registry.json`

### Pydantic Schemas (Phase 3A.2)

- **Type Safety**: All JSON artifacts validated with Pydantic v2
- **Version Migration**: Schema versioning for future compatibility
- **Better Errors**: Clear validation error messages

### Telemetry (Phase 3A.5)

- **Privacy-First**: Local-only by default
- **PII Filtering**: Automatic removal of sensitive data
- **Suggestion Tracking**: Track which suggestions are acted upon

### Plugin System Improvements

- **Cleaner Projects**: Minimal boilerplate, only essential files
- **Better Templates**: Improved README, requirements, and configs
- **3 Plugins**: tabular, chatbot, image-classification

## Installation

```bash
pip install ml-assistant-cli==0.3.0
```

## Quick Start

```bash
# Initialize
mlcli init my-project
cd my-project

# Add data and train
mlcli preprocess --input data/raw/data.csv --target label
mlcli train
mlcli evaluate

# Get AI suggestions
mlcli suggest
```

## Breaking Changes

- Minimum Python version: 3.9 (was 3.8)
- Generated project structure simplified (fewer empty directories)
- Dependencies updated (removed bentoml from core, added joblib)

## Migration Guide

If you have existing projects:

1. Update MLCLI: `pip install --upgrade ml-assistant-cli`
2. Projects will continue to work without changes
3. Run `mlcli suggest` to see new AI-powered recommendations

## Known Issues

- BentoML packaging not yet implemented (planned for v0.4.0)
- Cloud deployment pending (planned for v0.3.1)

## Contributors

- MLCLI Team

## Next Release (v0.4.0)

- BentoML packaging
- Cloud deployment (BentoCloud, Azure ML, AWS SageMaker)
- Model monitoring
- Auto-scaling
