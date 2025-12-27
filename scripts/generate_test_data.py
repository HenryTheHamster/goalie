"""Generate synthetic test data for demonstration."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path


def generate_synthetic_fixtures(
    n_teams: int = 20,
    n_matches: int = 200,
    league_id: int = 39,
    season: int = 2024,
) -> pd.DataFrame:
    """Generate synthetic fixture data for testing."""
    np.random.seed(42)
    
    fixtures = []
    fixture_id = 1000
    
    # Generate team IDs
    team_ids = list(range(100, 100 + n_teams))
    
    # Generate matches over a season
    start_date = datetime(season, 8, 1, tzinfo=timezone.utc)
    
    for i in range(n_matches):
        # Random team pairing
        home_team_id, away_team_id = np.random.choice(team_ids, size=2, replace=False)
        
        # Match date
        match_date = start_date + timedelta(days=i * 4)
        
        # Generate realistic goals (Poisson-ish distribution)
        home_goals = np.random.poisson(1.5)
        away_goals = np.random.poisson(1.2)
        
        fixtures.append({
            "fixture_id": fixture_id,
            "match_datetime_utc": match_date.isoformat(),
            "season": season,
            "round": f"Regular Season - {i // 10 + 1}",
            "league_id": league_id,
            "league_name": "Premier League",
            "league_country": "England",
            "home_team_id": home_team_id,
            "home_team_name": f"Team {home_team_id}",
            "away_team_id": away_team_id,
            "away_team_name": f"Team {away_team_id}",
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_goals_halftime": home_goals // 2 if home_goals > 0 else 0,
            "away_goals_halftime": away_goals // 2 if away_goals > 0 else 0,
            "status": "FT",
            "status_long": "Match Finished",
            "venue_id": home_team_id + 1000,
            "venue_name": f"Stadium {home_team_id}",
            "venue_city": f"City {home_team_id}",
            "canonicalized_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        
        fixture_id += 1
    
    return pd.DataFrame(fixtures)


def main():
    """Generate and save synthetic data."""
    print("Generating synthetic test data...")
    
    # Create artifacts directory
    artifacts_dir = Path("artifacts/parquet/canonical/matches")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate fixtures
    df = generate_synthetic_fixtures(n_teams=20, n_matches=200)
    
    # Save to Parquet
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = artifacts_dir / f"matches_training_{timestamp}.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"✓ Generated {len(df)} synthetic matches")
    print(f"✓ Saved to {output_path}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Average home goals: {df['home_goals'].mean():.2f}")
    print(f"  Average away goals: {df['away_goals'].mean():.2f}")
    print(f"  Total goals per match: {(df['home_goals'] + df['away_goals']).mean():.2f}")
    print(f"  Home wins: {(df['home_goals'] > df['away_goals']).sum()}")
    print(f"  Draws: {(df['home_goals'] == df['away_goals']).sum()}")
    print(f"  Away wins: {(df['home_goals'] < df['away_goals']).sum()}")


if __name__ == "__main__":
    main()
