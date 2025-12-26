# Goalie: Complete Usage Guide

This document provides detailed examples of using the Goalie prediction framework.

## Table of Contents
1. [Setup](#setup)
2. [Data Ingestion](#data-ingestion)
3. [Feature Engineering](#feature-engineering)
4. [Model Training](#model-training)
5. [Analysis and Reporting](#analysis-and-reporting)
6. [Advanced Usage](#advanced-usage)

## Setup

### Installation
```bash
# Clone the repository
git clone https://github.com/HenryTheHamster/goalie.git
cd goalie

# Install dependencies
pip install -e .

# Set up API key
export API_FOOTBALL_KEY="your-api-key-here"
# Or create .env file
echo "API_FOOTBALL_KEY=your-api-key-here" > .env
```

### Verify Installation
```bash
python -m src.cli --help
```

## Quick Start

Run the complete demo pipeline with synthetic data:
```bash
bash scripts/demo.sh
```

## Data Ingestion

### Option 1: Use API-Football (Real Data)

First, verify your API connection:
```bash
python -m src.cli health-check
```

Ingest fixtures and teams:
```bash
# Ingest all configured leagues
python -m src.cli ingest

# Ingest specific league
python -m src.cli ingest --league-id 39  # Premier League
```

### Option 2: Use Synthetic Data (For Testing)

```bash
python scripts/generate_test_data.py
```

## Pipeline Workflow

### 1. Canonicalize Data
```bash
python -m src.cli canonicalize
```

### 2. Engineer Features
```bash
python -m src.cli features
```

### 3. Train Models
```bash
python -m src.cli train --run-id my_experiment
```

### 4. Analyze Results
```bash
python -m src.cli report
```

### 5. View Report
```bash
cat artifacts/runs/my_experiment/report.md
```

## Key Features

### Leakage Protection
- All features computed from past matches only
- Odds disabled by default (requires explicit flag)
- Temporal ordering strictly enforced

### Model Outputs
- Home/away goal predictions
- Goal probability distributions
- Over/Under probabilities (1.5, 2.5, 3.5)
- Confidence scores

### Evaluation Metrics
- MAE, RMSE for point predictions
- Log-likelihood for distributional fit
- Brier score for probabilistic accuracy
- Diagnostic P&L backtest

For detailed usage, see the full documentation.
