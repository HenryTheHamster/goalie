"""Command-line interface for Goalie prediction framework."""

import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from src.config import Config, LeagueConfig
from src.data import DataCanonicalizer, TeamCanonicalizer
from src.eval import BacktestSimulator, MetricsCalculator
from src.features import FeatureEngineer
from src.ingest import APIFootballClient
from src.models import ModelTrainer
from src.reports import DuckDBQueryLayer, MarkdownReportGenerator

# Load environment variables
load_dotenv()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Goalie: AI/ML prediction framework for football (soccer) goals."""
    pass


@main.command()
@click.option("--api-key", envvar="API_FOOTBALL_KEY", required=True, help="API-Football API key")
def health_check(api_key: str):
    """Check API-Football connection and authentication."""
    click.echo("Checking API-Football health...")
    
    config = Config.default(api_key=api_key)
    client = APIFootballClient(config.api)
    
    if client.health_check():
        click.echo("✓ API-Football connection successful")
    else:
        click.echo("✗ API-Football connection failed", err=True)
        sys.exit(1)


@main.command()
@click.option("--api-key", envvar="API_FOOTBALL_KEY", required=True, help="API-Football API key")
@click.option("--league-id", type=int, help="Specific league ID to ingest (default: all configured)")
@click.option("--fixtures/--no-fixtures", default=True, help="Ingest fixtures")
@click.option("--teams/--no-teams", default=True, help="Ingest teams")
@click.option("--leagues/--no-leagues", default=False, help="Ingest leagues catalog")
def ingest(api_key: str, league_id: int, fixtures: bool, teams: bool, leagues: bool):
    """Ingest data from API-Football."""
    config = Config.default(api_key=api_key)
    client = APIFootballClient(config.api)
    
    # Filter leagues if specific league_id is provided
    leagues_to_ingest = config.leagues
    if league_id:
        leagues_to_ingest = [lc for lc in config.leagues if lc.league_id == league_id]
        if not leagues_to_ingest:
            click.echo(f"League ID {league_id} not found in configuration", err=True)
            sys.exit(1)
    
    # Ingest leagues catalog
    if leagues:
        click.echo("\n=== Ingesting Leagues ===")
        client.ingest_leagues(config.paths.raw_leagues)
    
    # Ingest data for each league
    for league_config in leagues_to_ingest:
        click.echo(f"\n=== Processing {league_config.name} (ID: {league_config.league_id}) ===")
        
        if teams:
            click.echo("\n--- Ingesting Teams ---")
            client.ingest_teams(league_config, config.paths.raw_teams)
        
        if fixtures:
            click.echo("\n--- Ingesting Fixtures ---")
            client.ingest_fixtures(league_config, config.paths.raw_fixtures)
    
    click.echo("\n✓ Ingestion complete")


@main.command()
@click.option("--api-key", envvar="API_FOOTBALL_KEY", help="API-Football API key (for config)")
def canonicalize(api_key: str):
    """Canonicalize raw data into training-ready format."""
    # Use default config to get paths
    if api_key:
        config = Config.default(api_key=api_key)
    else:
        # Mock API key for paths only
        config = Config.default(api_key="test-key-for-paths-only")
    
    # Canonicalize fixtures
    click.echo("\n=== Canonicalizing Fixtures ===")
    canonicalizer = DataCanonicalizer(config.paths)
    df_training, df_tracking = canonicalizer.canonicalize_fixtures()
    
    click.echo(f"\n✓ Training matches: {len(df_training)}")
    click.echo(f"✓ Tracking fixtures: {len(df_tracking)}")
    
    # Canonicalize teams
    click.echo("\n=== Canonicalizing Teams ===")
    team_canonicalizer = TeamCanonicalizer(config.paths)
    df_teams = team_canonicalizer.canonicalize_teams()
    
    click.echo(f"\n✓ Team records: {len(df_teams)}")
    click.echo("\n✓ Canonicalization complete")


@main.command()
@click.option("--api-key", envvar="API_FOOTBALL_KEY", help="API-Football API key (for config)")
def list_data(api_key: str):
    """List available data files."""
    if api_key:
        config = Config.default(api_key=api_key)
    else:
        config = Config.default(api_key="test-key-for-paths-only")
    
    click.echo("=== Available Data ===\n")
    
    # Raw data
    click.echo("Raw Fixtures:")
    raw_fixtures = list(config.paths.raw_fixtures.glob("*.parquet"))
    for f in raw_fixtures:
        click.echo(f"  - {f.name}")
    if not raw_fixtures:
        click.echo("  (none)")
    
    click.echo("\nRaw Teams:")
    raw_teams = list(config.paths.raw_teams.glob("*.parquet"))
    for f in raw_teams:
        click.echo(f"  - {f.name}")
    if not raw_teams:
        click.echo("  (none)")
    
    # Canonical data
    click.echo("\nCanonical Matches:")
    canonical_matches = list(config.paths.canonical_matches.glob("*.parquet"))
    for f in canonical_matches:
        click.echo(f"  - {f.name}")
    if not canonical_matches:
        click.echo("  (none)")
    
    click.echo("\nCanonical Teams:")
    canonical_teams = list(config.paths.canonical_teams.glob("*.parquet"))
    for f in canonical_teams:
        click.echo(f"  - {f.name}")
    if not canonical_teams:
        click.echo("  (none)")


@main.command()
@click.option("--api-key", envvar="API_FOOTBALL_KEY", help="API-Football API key (for config)")
@click.option("--cutoff", type=str, help="Snapshot cutoff (e.g., T-7d, T-3d, T-24h, T-6h, T-1h)")
def features(api_key: str, cutoff: str):
    """Engineer features from canonical matches."""
    if api_key:
        config = Config.default(api_key=api_key)
    else:
        config = Config.default(api_key="test-key-for-paths-only")
    
    # Load canonical matches
    click.echo("Loading canonical matches...")
    canonicalizer = DataCanonicalizer(config.paths)
    matches = canonicalizer.load_canonical_matches(latest=True)
    
    # Engineer features
    engineer = FeatureEngineer(config)
    features_df = engineer.engineer_features(matches, cutoff=None)
    
    # Save features
    features_with_context = engineer.save_features(features_df, matches, cutoff=None)
    
    click.echo(f"\n✓ Feature engineering complete")
    click.echo(f"  Matches: {len(features_df)}")
    click.echo(f"  Features: {len(features_df.columns) - 1}")


@main.command()
@click.option("--api-key", envvar="API_FOOTBALL_KEY", help="API-Football API key (for config)")
@click.option("--run-id", help="Run ID for tracking (auto-generated if not provided)")
def train(api_key: str, run_id: str):
    """Train prediction models on engineered features."""
    from datetime import datetime, timezone
    
    if api_key:
        config = Config.default(api_key=api_key)
    else:
        config = Config.default(api_key="test-key-for-paths-only")
    
    # Generate run ID if not provided
    if not run_id:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    click.echo(f"Run ID: {run_id}")
    
    # Load features
    click.echo("Loading features...")
    engineer = FeatureEngineer(config)
    features = engineer.load_latest_features()
    
    # Initialize and train models
    trainer = ModelTrainer(config)
    trainer.initialize_models()
    
    X_test, y_home_test, y_away_test = trainer.train_all(features)
    
    # Generate predictions on test set
    click.echo("\n=== Generating Test Predictions ===")
    predictions = trainer.predict_all(X_test)
    
    # Evaluate models
    click.echo("\n=== Evaluating Models ===")
    all_metrics = {}
    
    for model_name, preds in predictions.items():
        metrics = MetricsCalculator.evaluate_model(preds, y_home_test, y_away_test)
        all_metrics[model_name] = metrics
        MetricsCalculator.print_metrics(model_name, metrics)
        
        # Diagnostic P&L backtest
        pl_results = BacktestSimulator.simulate_simple_pl(preds, y_home_test, y_away_test)
        click.echo(f"\nDiagnostic P&L (Over/Under 2.5):")
        click.echo(f"  Total bets: {pl_results['total_bets']}")
        click.echo(f"  Correct: {pl_results['correct_bets']}")
        click.echo(f"  Accuracy: {pl_results['accuracy']:.2%}")
        click.echo(f"  Total P&L: ${pl_results['total_pl']:.2f}")
        click.echo(f"  ROI: {pl_results['roi']:.2%}")
    
    # Save predictions
    trainer.save_predictions(predictions, features.loc[X_test.index])
    
    # Generate report
    click.echo("\n=== Generating Report ===")
    report_gen = MarkdownReportGenerator(config)
    report_path = report_gen.save_report(run_id, all_metrics)
    
    click.echo(f"\n✓ Training and evaluation complete")
    click.echo(f"✓ Report: {report_path}")


@main.command()
@click.option("--api-key", envvar="API_FOOTBALL_KEY", help="API-Football API key (for config)")
def report(api_key: str):
    """Generate analysis report using DuckDB."""
    if api_key:
        config = Config.default(api_key=api_key)
    else:
        config = Config.default(api_key="test-key-for-paths-only")
    
    click.echo("=== Data Analysis Report ===\n")
    
    query_layer = DuckDBQueryLayer(config.paths)
    
    # Match summary
    click.echo("Match Summary:")
    try:
        match_summary = query_layer.get_match_summary()
        click.echo(match_summary.to_string(index=False))
    except Exception as e:
        click.echo(f"  (unavailable: {e})")
    
    click.echo("\n" + "="*50 + "\n")
    
    # Team statistics
    click.echo("Top 10 Teams by Points:")
    try:
        team_stats = query_layer.get_team_statistics()
        click.echo(team_stats.head(10).to_string(index=False))
    except Exception as e:
        click.echo(f"  (unavailable: {e})")
    
    click.echo("\n" + "="*50 + "\n")
    
    # Prediction performance
    click.echo("Prediction Performance:")
    try:
        pred_perf = query_layer.get_prediction_performance()
        click.echo(pred_perf.to_string(index=False))
    except Exception as e:
        click.echo(f"  (unavailable: {e})")
    
    click.echo("\n" + "="*50 + "\n")
    
    # Over/Under performance
    click.echo("Over/Under 2.5 Performance:")
    try:
        ou_perf = query_layer.get_over_under_performance(threshold=2.5)
        click.echo(ou_perf.to_string(index=False))
    except Exception as e:
        click.echo(f"  (unavailable: {e})")
    
    query_layer.close()
    click.echo("\n✓ Analysis complete")


if __name__ == "__main__":
    main()
