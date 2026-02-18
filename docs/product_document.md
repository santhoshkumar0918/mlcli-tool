<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# ML Assistant CLI — Full Product Document

Date: August 12, 2025

## Overview

ML Assistant CLI is a single, developer-first command-line tool that takes a project from raw dataset to a deployed, monitored API endpoint with a few simple commands, combining preprocessing, training, evaluation, packaging, deployment, monitoring, and rollback into one cohesive experience, with AI-guided suggestions along the way. It packages models as BentoML services for portability and integrates with cloud providers for managed online endpoints and traffic-safe rollouts.[^1][^2][^3][^4][^5][^6]

## Problem and Vision

ML workflows are fragmented across many tools for data prep, model training, packaging, and deployment, creating friction and steep learning curves for beginners and practitioners; ML Assistant CLI unifies this lifecycle under one opinionated, easy CLI with reproducible artifacts and multi-cloud deployment paths.[^3][^4][^6]

## User Personas

- Beginner ML student: wants local-first learning, sensible defaults, clear hints, and optional local serving.[^6]
- Applied ML engineer: needs quick dataset→endpoint, packaging, one-command deploy, logs, and simple rollback.[^2][^4][^3]
- MLOps/platform engineer: needs standardized packaging, versioning, CI/CD-friendly CLI, and cloud governance alignment.[^4][^7][^5][^1][^2]


## Key Capabilities

- **Domain-Adaptive Initialization**: Project scaffolding via a modular **Plugin System** (`--plugin tabular`, `chatbot`, `image-classification`).
- **End-to-end workflow**: Unified CLI for `init`, `preprocess`, `train`, `evaluate`, `suggest`, `predict`, `package`, `deploy`, `monitor`, and `rollback`.
- **AI-Powered Recommendations**: 
  - *Current*: Rule-based data quality and hyperparameter nudges.
  - *Future*: Meta-ML inference engine for personalized engineering advice.
- **Unified Packaging (BentoML)**: Automated Bento artifact and OCI image creation for zero-config reproducibility.
- **Multi-Cloud Deployment**:
    - **BentoCloud**: Serverless deployment with auto-scaling and rolling updates.
    - **Azure ML**: Managed online endpoints with blue/green traffic splitting.
    - **AWS SageMaker**: Orchestrated job submission and cluster management via SageMaker HyperPod.


## Architecture

- **CLI Core (Python/Typer)**: Command routing, configuration management, and interactive terminal UI via `rich`.
- **Plugin System**:
    - **`PluginBase`**: Abstract interface for domain-specific logic.
    - **`PluginRegistry`**: Dynamic discovery of built-in and custom templates.
    - **Domain Plugins**: `Tabular` (Sklearn), `Chatbot` (LangChain), `ImageClassification` (PyTorch).
- **Core Operations**: Modular engines for Data Processing, Model Training, and Evaluation.
- **Packaging Engine**: Wraps model artifacts into BentoML services using generated `bentofile.yaml`.
- **Provider Adapters**: Thin wrappers for cloud-native SDKs (BentoCloud, Azure ML, AWS SageMaker).
- **State Registry**: Local tracking of model versions, container tags, and endpoint traffic statuses.


## User Flows

- Local flow
    - mlcli init → preprocess → train → evaluate → suggest → predict → package for local Bento serving.[^6][^1][^3]
- Cloud flow (BentoCloud)
    - mlcli package → mlcli deploy --provider=bentocloud → mlcli monitor → mlcli rollback, leveraging bentoml deploy flags for scaling/strategy/env/secrets and wait/timeout.[^9][^2][^3]
- Cloud flow (Azure ML)
    - mlcli package → mlcli deploy --provider=azureml (create endpoint, first deployment, 100% traffic) → monitor logs → optional second deployment and traffic split; rollback by restoring traffic mapping.[^5][^4]
- Cloud flow (AWS HyperPod)
    - Prereqs: HyperPod CLI installed and cluster connected → mlcli package → mlcli deploy --provider=hyperpod (submit job/deploy) → monitor logs and describe; redeploy prior image/config to roll back.[^13][^11][^7][^10]


## MVP (College — Local Only)

- Scope: init, preprocess, train, evaluate, predict, suggest; local artifacts (model.pkl, metrics.json, pipeline.yaml, data_profile.json) and optional Bento local serving via bento build/serve.[^8][^1][^6]
- Technicals:
    - CLI: Python + Typer/Click with simple config files.
    - Preprocess: schema inference, missing value handling, encoding, scaling; data profile output.
    - Train: logistic regression, random forest, optional XGBoost; light grid/random search.
    - Evaluate: accuracy, precision, recall, F1, AUC, confusion matrix; next-steps hints.
    - Package: generate a Bento service and run bento build for consistent local serving.[^1][^8]


## Cloud-Enabled MVP

- Goal: extend local MVP with one-click deploy to a primary provider (BentoCloud recommended for speed) with monitor/rollback.[^2][^3][^9]
- Features:
    - package: produce Bento with tag/version suitable for cloud deployment.[^3][^1]
    - deploy: call provider CLI/SDK; return endpoint URL/status; accept scaling/instance/strategy/env/secrets via flags or config file.[^9][^4][^5][^2]
    - monitor: provider logs/status wrappers.[^12][^7][^10][^4][^5]
    - rollback: bind to provider-native traffic/version semantics (BentoCloud strategies, Azure traffic mapping, HyperPod redeploy).[^7][^10][^4][^5][^2]
- Why BentoCloud first: single bentoml deploy command handles build→push→deploy automatically, with real-time status and robust config options.[^2][^3][^9]


## Full Product (Multi-Cloud, Production-grade)

- Multi-provider abstraction: BentoCloud, Azure ML managed online endpoints, AWS SageMaker HyperPod; consistent CLI UX across providers.[^10][^4][^7][^2]
- Advanced deployment:
    - Config-driven deploys via YAML/JSON files, secrets and env injection, rollout strategies (rolling, recreate, controlled), min/max replicas, instance types.[^4][^5][^9][^2]
    - Traffic management and safe rollouts (blue/green, canary) with explicit traffic split adjustments and rollback.[^5][^9][^4][^2]
- Governance and security:
    - Auth flows: bentoml cloud login; Azure ML workspace auth via SDK/az; AWS IAM and HyperPod cluster context.[^7][^10][^4][^2]
    - Secrets: pass via BentoCloud flags/config or Azure Key Vault/managed identity; ensure least-privilege IAM on AWS.[^9][^4][^7]
- Observability:
    - Logs and status via provider CLIs/APIs: bentoml cloud describe/logs; az ml online-deployment logs/show; hyperpod get-log.[^12][^10][^4][^7]
    - Local dev: Azure’s --local flag to run endpoints in Docker for rapid testing before cloud promote.[^5]
- CI/CD readiness:
    - Non-interactive deployments with config files and tokens; environment checks; version pinning for Bento builds; reproducible deploy specs.[^4][^2][^9]


## Command Surface

- init: Scaffolds project using `--plugin` strategy (tabular, chatbot, image-classification).
- preprocess: Data cleaning and profiling; emits `data_profile.json`.
- train: Plugin-aware model training with hyperparameter tuning; saves `model.pkl`.
- evaluate: Detailed metrics, diagnostics, and visual reports.
- suggest:AI-guided recommendations (Rule-based → Meta-ML).
- predict: Batch inference for new datasets.
- package: Containerizes project via BentoML versioned tags.
- deploy: Multi-provider deployment management (scaling, rollout strategies).
- monitor: Real-time logs and health status from cloud endpoints.
- rollback: Reversion to previous stable traffic/version configurations.


## Example Commands

- Local:
    - mlcli init → mlcli preprocess --input data.csv → mlcli train --model random_forest → mlcli evaluate --test test.csv → mlcli suggest → mlcli predict --input new.csv --output predictions.csv.[^6]
    - mlcli package → serve locally with Bento using bento build/serve.[^1][^3][^8]
- BentoCloud:
    - mlcli package → mlcli deploy --provider=bentocloud --name my-svc --scaling-min 1 --scaling-max 3 --instance-type cpu.2 --strategy RollingUpdate --wait → mlcli monitor → mlcli rollback --version v1.[^3][^9][^2]
- Azure ML:
    - mlcli package → mlcli deploy --provider=azureml (create endpoint, blue deployment, 100% traffic) → mlcli monitor → mlcli deploy --provider=azureml --slot green and traffic-split 10/90 → mlcli rollback to blue.[^5][^4]
- HyperPod:
    - hyperpod connect-cluster … → mlcli package → mlcli deploy --provider=hyperpod → mlcli monitor (get logs, list pods) → mlcli rollback by redeploying previous image.[^11][^10][^7]


## Technical Execution Plan

- **Phase 1: Modular Foundation (ACTIVE)**: 
    - ✅ Plugin Core (Base/Registry).
    - ✅ Tabular, Chatbot, and Image Plugins.
    - ✅ Boilerplate generation for domain-specific ML.
- **Phase 2: Cloud Integration (UP NEXT)**:
    - Implement `package` command via BentoML.
    - Initial `deploy` support for BentoCloud.
- **Phase 3: Intelligence & Scale**:
    - Transition to Meta-ML suggestion engine.
    - Full multi-provider cloud support (Azure/AWS).


## Risks and Mitigations

- Auth/config friction: add preflight checks for tokens, workspace, cluster; clear error messages with remediation steps.[^7][^10][^2][^4]
- Packaging drift: standard Bento service templates, version pinning, and capture deploy args into tracked config files for reproducibility.[^8][^1][^9]
- Rollback complexity: local registry to map versions→deployments; provider-native traffic/version rollbacks to keep operations safe and fast.[^2][^4][^5]
- Provider changes: isolate provider adapters and maintain contract tests against CLIs/SDKs.[^10][^4][^7][^2]


## What Makes It Stand Out

An opinionated CLI that unifies the full ML lifecycle with AI-guided help, reproducible Bento packaging, and first-class multi-cloud deployment integrations, delivering beginner-friendly UX and production-ready rollouts with traffic control and logs from day one.[^1][^6][^8][^3][^4][^7][^5][^2]

<div style="text-align: center">⁂</div>

[^1]: https://docs.bentoml.com/en/latest/reference/bentoml/cli.html

[^2]: https://docs.bentoml.com/en/latest/reference/bentocloud/bentocloud-cli.html

[^3]: https://docs.bentoml.com/en/latest/scale-with-bentocloud/deployment/create-deployments.html

[^4]: https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-deploy-model?view=azureml-api-2

[^5]: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2

[^6]: https://docs.bentoml.com

[^7]: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks-hyperpod-cli-reference.html

[^8]: https://pypi.org/project/bentoml/

[^9]: https://docs.bentoml.com/en/latest/scale-with-bentocloud/deployment/configure-deployments.html

[^10]: https://github.com/aws/sagemaker-hyperpod-cli

[^11]: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html

[^12]: https://docs.bentoml.com/en/latest/scale-with-bentocloud/deployment/manage-deployments.html

[^13]: https://docs.aws.amazon.com/sagemaker/latest/dg/nova-hp-train.html

[^14]: https://docs.zenml.io/stacks/stack-components/model-deployers/bentoml

[^15]: https://www.youtube.com/watch?v=XSuDno7gpQ8

[^16]: https://github.com/bentoml/aws-lambda-deploy

[^17]: https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-pipeline-python-sdk?view=azureml-api-2

[^18]: https://docs.aws.amazon.com/sagemaker/latest/dg/smcluster-getting-started-slurm-cli.html

[^19]: https://learn.microsoft.com/en-us/azure/machine-learning/migrate-to-v2-deploy-endpoints?view=azureml-api-2

[^20]: https://docs.azure.cn/en-us/machine-learning/how-to-deploy-automl-endpoint?view=azureml-api-2
