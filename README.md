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

### Check API Connection
```bash
goalie health-check
```

### Ingest Data
```bash
# Ingest fixtures and teams for all configured leagues
goalie ingest

# Ingest specific league
goalie ingest --league-id 39  # Premier League

# Ingest only fixtures
goalie ingest --no-teams

# Include leagues catalog
goalie ingest --leagues
```

### Canonicalize Data
```bash
# Transform raw data to canonical format
goalie canonicalize
```

### List Available Data
```bash
goalie list-data
```

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

Coming soon:
- Feature engineering (Elo, EWMA, Rolling)
- Model implementations (Baseline, XGBoost, Bayesian)
- Evaluation metrics and calibration
- DuckDB query layer
- Per-run reporting
- Full test coverage

## License

MIT
