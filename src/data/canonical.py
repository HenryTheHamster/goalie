"""Data canonicalization and quality checks."""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.config import MatchStatus, PathConfig


class DataCanonicalizer:
    """Canonicalize raw API-Football data."""
    
    def __init__(self, paths: PathConfig):
        """Initialize with path configuration."""
        self.paths = paths
    
    def _load_raw_fixtures(self) -> pd.DataFrame:
        """Load all raw fixture files."""
        raw_files = list(self.paths.raw_fixtures.glob("*.parquet"))
        if not raw_files:
            raise FileNotFoundError(f"No raw fixture files found in {self.paths.raw_fixtures}")
        
        dfs = [pd.read_parquet(f) for f in raw_files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df)} raw fixture records from {len(raw_files)} files")
        return df
    
    def _deduplicate_fixtures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate by fixture_id, keeping most recent ingestion."""
        # Sort by ingestion time (most recent last)
        df = df.sort_values("ingested_at_utc")
        
        # Keep last occurrence of each fixture_id
        df_deduped = df.drop_duplicates(subset=["fixture_id"], keep="last")
        
        n_removed = len(df) - len(df_deduped)
        if n_removed > 0:
            print(f"Removed {n_removed} duplicate fixtures (kept most recent)")
        
        return df_deduped
    
    def _validate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all timestamps are in UTC."""
        # Check if UTC timestamp exists
        if "fixture_timestamp_utc" not in df.columns or df["fixture_timestamp_utc"].isna().all():
            raise ValueError("fixture_timestamp_utc is missing or all null")
        
        # Parse and validate UTC timestamps
        df["match_datetime_utc"] = pd.to_datetime(df["fixture_timestamp_utc"], errors="coerce")
        
        invalid_timestamps = df["match_datetime_utc"].isna().sum()
        if invalid_timestamps > 0:
            print(f"Warning: {invalid_timestamps} fixtures have invalid timestamps")
        
        return df
    
    def _filter_training_matches(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training-valid and tracking-only matches."""
        # Training: only completed matches
        training_mask = df["fixture_status_short"].isin(MatchStatus.training_statuses())
        df_training = df[training_mask].copy()
        
        # Tracking: all other statuses (for fixture tracking)
        df_tracking = df[~training_mask].copy()
        
        print(f"Training matches: {len(df_training)} (status in {MatchStatus.training_statuses()})")
        print(f"Tracking only: {len(df_tracking)} (other statuses)")
        
        return df_training, df_tracking
    
    def _create_canonical_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform to canonical match schema."""
        canonical = pd.DataFrame({
            # Primary key
            "fixture_id": df["fixture_id"],
            
            # Match timing
            "match_datetime_utc": df["match_datetime_utc"],
            "season": df["league_season"],
            "round": df["league_round"],
            
            # League
            "league_id": df["league_id"],
            "league_name": df["league_name"],
            "league_country": df["league_country"],
            
            # Teams
            "home_team_id": df["teams_home_id"],
            "home_team_name": df["teams_home_name"],
            "away_team_id": df["teams_away_id"],
            "away_team_name": df["teams_away_name"],
            
            # Results
            "home_goals": df["goals_home"],
            "away_goals": df["goals_away"],
            "home_goals_halftime": df["score_halftime_home"],
            "away_goals_halftime": df["score_halftime_away"],
            
            # Status
            "status": df["fixture_status_short"],
            "status_long": df["fixture_status_long"],
            
            # Venue
            "venue_id": df["fixture_venue_id"],
            "venue_name": df["fixture_venue_name"],
            "venue_city": df["fixture_venue_city"],
            
            # Metadata
            "canonicalized_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        
        return canonical
    
    def _validate_data_quality(self, df: pd.DataFrame):
        """Run data quality checks."""
        issues = []
        
        # Check for missing fixture IDs
        if df["fixture_id"].isna().any():
            issues.append("Missing fixture_id values")
        
        # Check for missing team IDs
        if df["home_team_id"].isna().any() or df["away_team_id"].isna().any():
            issues.append("Missing team_id values")
        
        # Check for missing league IDs
        if df["league_id"].isna().any():
            issues.append("Missing league_id values")
        
        # Check for missing match times
        if df["match_datetime_utc"].isna().any():
            issues.append(f"{df['match_datetime_utc'].isna().sum()} missing match timestamps")
        
        # Check for completed matches with missing goals
        completed = df[df["status"].isin(MatchStatus.training_statuses())]
        if not completed.empty:
            missing_goals = completed["home_goals"].isna() | completed["away_goals"].isna()
            if missing_goals.any():
                issues.append(
                    f"{missing_goals.sum()} completed matches missing goal data"
                )
        
        if issues:
            print("Data quality issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Data quality: OK")
        
        return issues
    
    def canonicalize_fixtures(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Canonicalize raw fixtures into training and tracking datasets."""
        print("\n=== Canonicalizing Fixtures ===")
        
        # Load raw data
        df_raw = self._load_raw_fixtures()
        
        # Deduplicate by fixture_id
        df_deduped = self._deduplicate_fixtures(df_raw)
        
        # Validate timestamps
        df_validated = self._validate_timestamps(df_deduped)
        
        # Split into training and tracking
        df_training_raw, df_tracking_raw = self._filter_training_matches(df_validated)
        
        # Transform to canonical schema
        df_training = self._create_canonical_schema(df_training_raw)
        df_tracking = self._create_canonical_schema(df_tracking_raw)
        
        # Validate data quality
        print("\nTraining data quality:")
        self._validate_data_quality(df_training)
        
        # Save canonical data
        self._save_canonical_data(df_training, df_tracking)
        
        return df_training, df_tracking
    
    def _save_canonical_data(self, df_training: pd.DataFrame, df_tracking: pd.DataFrame):
        """Save canonical datasets to Parquet."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Save training matches
        training_path = self.paths.canonical_matches / f"matches_training_{timestamp}.parquet"
        df_training.to_parquet(training_path, index=False)
        print(f"\nSaved {len(df_training)} training matches to {training_path}")
        
        # Save tracking fixtures
        tracking_path = self.paths.canonical_fixtures / f"fixtures_tracking_{timestamp}.parquet"
        df_tracking.to_parquet(tracking_path, index=False)
        print(f"Saved {len(df_tracking)} tracking fixtures to {tracking_path}")
    
    def load_canonical_matches(self, latest: bool = True) -> pd.DataFrame:
        """Load canonical training matches."""
        match_files = sorted(self.paths.canonical_matches.glob("matches_training_*.parquet"))
        
        if not match_files:
            raise FileNotFoundError(f"No canonical match files found in {self.paths.canonical_matches}")
        
        if latest:
            # Load only the most recent file
            df = pd.read_parquet(match_files[-1])
            print(f"Loaded {len(df)} matches from {match_files[-1].name}")
        else:
            # Load all files
            dfs = [pd.read_parquet(f) for f in match_files]
            df = pd.concat(dfs, ignore_index=True)
            df = df.drop_duplicates(subset=["fixture_id"], keep="last")
            print(f"Loaded {len(df)} matches from {len(match_files)} files")
        
        # Ensure datetime is parsed
        df["match_datetime_utc"] = pd.to_datetime(df["match_datetime_utc"])
        
        return df


class TeamCanonicalizer:
    """Canonicalize team data."""
    
    def __init__(self, paths: PathConfig):
        """Initialize with path configuration."""
        self.paths = paths
    
    def canonicalize_teams(self) -> pd.DataFrame:
        """Canonicalize raw team data."""
        print("\n=== Canonicalizing Teams ===")
        
        # Load raw teams
        raw_files = list(self.paths.raw_teams.glob("*.parquet"))
        if not raw_files:
            raise FileNotFoundError(f"No raw team files found in {self.paths.raw_teams}")
        
        dfs = [pd.read_parquet(f) for f in raw_files]
        df_raw = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df_raw)} raw team records")
        
        # Deduplicate by team_id and season
        df_raw = df_raw.sort_values("ingested_at_utc")
        df_deduped = df_raw.drop_duplicates(subset=["team_id", "season"], keep="last")
        print(f"After deduplication: {len(df_deduped)} team-season records")
        
        # Create canonical schema
        canonical = pd.DataFrame({
            "team_id": df_deduped["team_id"],
            "team_name": df_deduped["team_name"],
            "team_code": df_deduped["team_code"],
            "team_country": df_deduped["team_country"],
            "league_id": df_deduped["league_id"],
            "season": df_deduped["season"],
            "venue_id": df_deduped["venue_id"],
            "venue_name": df_deduped["venue_name"],
            "venue_city": df_deduped["venue_city"],
            "canonicalized_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        
        # Save
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = self.paths.canonical_teams / f"teams_{timestamp}.parquet"
        canonical.to_parquet(output_path, index=False)
        print(f"Saved {len(canonical)} team records to {output_path}")
        
        return canonical
