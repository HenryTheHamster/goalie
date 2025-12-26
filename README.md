# Goalie: AI/ML Football Goals Prediction Framework

End-to-end, reproducible AI/ML prediction framework for football (soccer) goals, focused on English Premier League (EPL) and Championship.

## Features

- **Target Predictions**: Home/away goal distributions and Over/Under probabilities
- **Rolling Pre-match Predictions**: Fixed snapshot cutoffs at T-7d, T-3d, T-24h, T-6h, T-1h
- **Layered Features**: Elo-style attack/defence ratings, EWMA rolling stats, rolling averages
- **Multiple Models**: Statistical baseline (Poisson), ML challenger (XGBoost), Bayesian model
- **Comprehensive Evaluation**: Likelihood/fit, calibration, market utility metrics, P&L backtests
- **Robust Storage**: Parquet snapshots, DuckDB query layer, per-run Markdown reports
- **Data Source**: API-Football (DIRECT auth) with local mirroring

## Hard Rules

### 1. NO TARGET LEAKAGE
- No feature may use post-cutoff information relative to snapshot time
- Odds data NOT used unless explicitly enabled with `--feature-group odds`
- Pipeline fails with clear error if leakage detected

### 2. INTERPRETABILITY FIRST
- Every feature has stable name, description, logged in feature dictionary
- Feature-group ablation support (enable/disable feature blocks)

### 3. API-FOOTBALL DATA QUALITY
- Training uses only final results: status in {FT, AET, PEN}
- Postponed/abandoned/suspended matches excluded from training
- Deduplicate by fixture_id; raw data is append-only
- Enforce UTC timestamps and league/season consistency

## Project Structure

```
goalie/
├── README.md
├── pyproject.toml
├── src/
│   ├── config/          # Configuration management
│   ├── ingest/          # API-Football data ingestion
│   ├── data/            # Data canonicalization
│   ├── features/        # Feature engineering
│   ├── models/          # Statistical, ML, Bayesian models
│   ├── eval/            # Evaluation metrics
│   ├── reports/         # Report generation
│   └── cli.py           # Command-line interface
├── scripts/             # Utility scripts
├── tests/               # Unit and integration tests
└── artifacts/
    ├── parquet/         # Data storage
    │   ├── raw/         # Raw API data (append-only)
    │   ├── canonical/   # Canonicalized data
    │   ├── features/    # Feature datasets
    │   ├── predictions/ # Model predictions
    │   └── metrics/     # Evaluation metrics
    └── runs/            # Per-run reports and manifests
```

## Installation

### Prerequisites
- Python 3.9+
- API-Football API key (DIRECT auth, not RapidAPI)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/HenryTheHamster/goalie.git
cd goalie
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up your API key:
```bash
export API_FOOTBALL_KEY="your-api-key-here"
# Or create .env file with: API_FOOTBALL_KEY=your-api-key-here
```

## Usage

### Quick Start

Run the complete demo pipeline with synthetic data:
```bash
bash scripts/demo.sh
```

### Step-by-Step Workflow

#### 1. Check API Connection
```bash
python -m src.cli health-check
```

#### 2. Ingest Data from API-Football
```bash
# Ingest fixtures and teams for all configured leagues
python -m src.cli ingest

# Ingest specific league
python -m src.cli ingest --league-id 39  # Premier League

# Ingest only fixtures
python -m src.cli ingest --no-teams

# Include leagues catalog
python -m src.cli ingest --leagues
```

#### 3. Canonicalize Data
Transform raw API data into training-ready format:
```bash
python -m src.cli canonicalize
```

#### 4. Engineer Features
Generate features with leakage protection:
```bash
python -m src.cli features
```

#### 5. Train Models
Train Poisson baseline and XGBoost models:
```bash
python -m src.cli train --run-id my_run
```

This command will:
- Train both baseline and XGBoost models
- Evaluate on test set
- Generate predictions with Over/Under probabilities
- Create a Markdown report with all metrics
- Save predictions to Parquet files

#### 6. Generate Analysis Report
Query and analyze results using DuckDB:
```bash
python -m src.cli report
```

#### 7. List Available Data
```bash
python -m src.cli list-data
```

### Using Synthetic Data for Testing

Generate synthetic test data without API calls:
```bash
python scripts/generate_test_data.py
```

This creates realistic synthetic match data for development and testing.

## Configuration

The framework uses sensible defaults for:
- **Leagues**: Premier League (ID: 39) and Championship (ID: 40)
- **Seasons**: 2022, 2023, 2024
- **Snapshot Cutoffs**: T-7d, T-3d, T-24h, T-6h, T-1h
- **Feature Groups**: Elo, EWMA, Rolling, Form, Venue (odds disabled by default)
- **Models**: Baseline Poisson, XGBoost, Bayesian

Configuration can be customized via code or YAML (to be implemented).

## Development Status

Currently implemented:
- ✅ Project structure and configuration
- ✅ API-Football client with DIRECT auth
- ✅ Data ingestion (fixtures, teams, leagues)
- ✅ Data canonicalization and deduplication
- ✅ UTC timestamp enforcement
- ✅ Training/tracking data split
- ✅ Basic CLI interface

### Implemented (Phase 2)
- ✅ Feature engineering (Elo, EWMA, Rolling statistics)
- ✅ Elo-style attack/defence ratings with leakage protection
- ✅ EWMA and rolling average features
- ✅ Feature dictionary for interpretability
- ✅ Model implementations (Poisson baseline, XGBoost)
- ✅ Goal distribution predictions (home/away)
- ✅ Over/Under probability calculations
- ✅ Evaluation metrics (MAE, RMSE, log-likelihood, calibration)
- ✅ Brier score for probabilistic predictions
- ✅ Diagnostic P&L backtest (not for trading)
- ✅ DuckDB query layer for analysis
- ✅ Per-run Markdown report generation
- ✅ Full CLI with train, features, report commands
- ✅ Synthetic data generation for testing

### Coming Soon
- Unit tests for core modules
- Integration tests for full pipeline
- Bayesian model implementation
- Advanced calibration plots
- Feature importance visualization
- Model persistence and loading
- YAML configuration support
- Time-series cross-validation
- Snapshot cutoff implementations (T-7d, T-3d, etc.)

## Example Output

After running `python -m src.cli train --run-id demo`, you'll get:

**Console Output:**
```
=== poisson_baseline Metrics ===
Home Goals:
  MAE: 1.145
  RMSE: 1.347
  Log-Likelihood: -65.64
  Accuracy (±1): 45.00%

Away Goals:
  MAE: 0.785
  RMSE: 0.907
  Log-Likelihood: -51.02
  Accuracy (±1): 62.50%

Total Goals:
  MAE: 1.284

Probabilistic:
  Brier Score: 0.7914
```

**Generated Files:**
- `artifacts/runs/demo/report.md` - Comprehensive Markdown report
- `artifacts/parquet/predictions/predictions_*.parquet` - Predictions with probabilities
- `artifacts/parquet/features/features_*.parquet` - Engineered features
- `artifacts/parquet/features/feature_dict_*.parquet` - Feature metadata

## Architecture Highlights

### Leakage Protection
- **Temporal Ordering**: All features respect match chronology
- **No Future Data**: Features computed only from past matches
- **Odds Safeguard**: Odds disabled by default, requires explicit flag
- **Validation**: Automated leakage detection at feature engineering stage

### Interpretability
- **Feature Dictionary**: Every feature documented with name, description, group
- **Feature Groups**: Supports ablation studies (enable/disable feature sets)
- **Model Comparison**: Side-by-side evaluation of multiple models
- **Transparent Metrics**: Clear reporting of all evaluation metrics

### Data Quality
- **Status Filtering**: Only completed matches (FT, AET, PEN) for training
- **Deduplication**: Fixture-level deduplication with most recent data
- **UTC Timestamps**: Enforced timezone consistency
- **Quality Checks**: Automated validation of data integrity

## License

MIT
