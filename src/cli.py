"""Command-line interface for Goalie prediction framework."""

import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from src.config import Config, LeagueConfig
from src.data import DataCanonicalizer, TeamCanonicalizer
from src.ingest import APIFootballClient

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


if __name__ == "__main__":
    main()
