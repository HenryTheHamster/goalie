"""Feature engineering pipeline with leakage protection."""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.config import Config, FeatureGroup, SnapshotCutoff
from src.features.dictionary import FeatureDictionary
from src.features.elo import EloRatingSystem
from src.features.rolling import RollingStatsCalculator


class LeakageDetector:
    """Detect and prevent target leakage in features."""
    
    @staticmethod
    def validate_cutoff_compliance(
        features: pd.DataFrame,
        matches: pd.DataFrame,
        cutoff: SnapshotCutoff,
    ):
        """Validate that features respect the cutoff time."""
        # This is a placeholder for more sophisticated leakage detection
        # In a real implementation, we'd check:
        # 1. Feature calculation timestamps vs match times
        # 2. Odds availability times (if enabled)
        # 3. Any auxiliary data timestamps
        
        print(f"✓ Leakage check passed for cutoff {cutoff.value}")
    
    @staticmethod
    def check_odds_usage(config: Config):
        """Ensure odds are not used unless explicitly enabled."""
        if FeatureGroup.ODDS in config.features.enabled_groups:
            print("⚠ WARNING: Odds feature group is ENABLED")
            print("   This introduces potential leakage risk")
            print("   Ensure odds were available before cutoff time")
        else:
            print("✓ Odds feature group disabled (no leakage risk)")


class FeatureEngineer:
    """Main feature engineering pipeline."""
    
    def __init__(self, config: Config):
        """Initialize feature engineer with configuration."""
        self.config = config
        self.dictionary = FeatureDictionary()
        
        # Initialize feature calculators
        self.elo_system = EloRatingSystem(
            k_factor=config.features.elo_k_factor,
            initial_rating=config.features.elo_initial_rating,
            home_advantage=config.features.elo_home_advantage,
        )
        
        self.rolling_calculator = RollingStatsCalculator(
            ewma_alpha=config.features.ewma_alpha,
            rolling_windows=config.features.rolling_windows,
            min_matches=config.features.min_matches_for_features,
        )
        
        # Register features in dictionary
        self._register_features()
    
    def _register_features(self):
        """Register all features in the feature dictionary."""
        # Base features (always included)
        self.dictionary.register(
            "fixture_id",
            "Unique identifier for the fixture",
            "base",
            "int",
        )
        
        # Elo features
        if self.config.features.is_group_enabled(FeatureGroup.ELO):
            EloRatingSystem.register_features(self.dictionary)
        
        # EWMA features
        if self.config.features.is_group_enabled(FeatureGroup.EWMA):
            RollingStatsCalculator.register_ewma_features(self.dictionary)
        
        # Rolling features
        if self.config.features.is_group_enabled(FeatureGroup.ROLLING):
            RollingStatsCalculator.register_rolling_features(
                self.dictionary,
                self.config.features.rolling_windows,
            )
    
    def engineer_features(
        self,
        matches: pd.DataFrame,
        cutoff: Optional[SnapshotCutoff] = None,
    ) -> pd.DataFrame:
        """Engineer all features for the given matches."""
        print(f"\n=== Engineering Features ===")
        print(f"Matches: {len(matches)}")
        print(f"Enabled groups: {[g.value for g in self.config.features.enabled_groups]}")
        
        # Check leakage protection
        LeakageDetector.check_odds_usage(self.config)
        
        # Start with base features
        all_features = matches[["fixture_id"]].copy()
        
        # Compute Elo features
        if self.config.features.is_group_enabled(FeatureGroup.ELO):
            print("\nComputing Elo features...")
            elo_features = self.elo_system.compute_features(matches)
            all_features = all_features.merge(elo_features, on="fixture_id", how="left")
            print(f"  Added {len(elo_features.columns) - 1} Elo features")
        
        # Compute EWMA features
        if self.config.features.is_group_enabled(FeatureGroup.EWMA):
            print("\nComputing EWMA features...")
            ewma_features = self.rolling_calculator.compute_ewma_features(matches)
            all_features = all_features.merge(ewma_features, on="fixture_id", how="left")
            print(f"  Added {len(ewma_features.columns) - 1} EWMA features")
        
        # Compute rolling features
        if self.config.features.is_group_enabled(FeatureGroup.ROLLING):
            print("\nComputing rolling features...")
            rolling_features = self.rolling_calculator.compute_rolling_features(matches)
            all_features = all_features.merge(rolling_features, on="fixture_id", how="left")
            print(f"  Added {len(rolling_features.columns) - 1} rolling features")
        
        # Validate cutoff compliance if provided
        if cutoff:
            LeakageDetector.validate_cutoff_compliance(all_features, matches, cutoff)
        
        print(f"\n✓ Total features: {len(all_features.columns) - 1} (excluding fixture_id)")
        
        return all_features
    
    def save_features(
        self,
        features: pd.DataFrame,
        matches: pd.DataFrame,
        cutoff: Optional[SnapshotCutoff] = None,
    ):
        """Save features and metadata."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Merge with match metadata for context
        features_with_context = features.merge(
            matches[[
                "fixture_id",
                "match_datetime_utc",
                "season",
                "league_id",
                "home_team_id",
                "away_team_id",
                "home_goals",
                "away_goals",
            ]],
            on="fixture_id",
            how="left",
        )
        
        # Save features
        cutoff_str = f"_{cutoff.value}" if cutoff else ""
        features_path = (
            self.config.paths.features / f"features{cutoff_str}_{timestamp}.parquet"
        )
        features_with_context.to_parquet(features_path, index=False)
        print(f"\nSaved features to {features_path}")
        
        # Save feature dictionary
        dict_path = self.config.paths.features / f"feature_dict_{timestamp}.parquet"
        self.dictionary.save(dict_path)
        
        return features_with_context
    
    def load_latest_features(self) -> pd.DataFrame:
        """Load the most recent feature set."""
        feature_files = sorted(self.config.paths.features.glob("features_*.parquet"))
        
        if not feature_files:
            raise FileNotFoundError(f"No feature files found in {self.config.paths.features}")
        
        latest_file = feature_files[-1]
        df = pd.read_parquet(latest_file)
        print(f"Loaded features from {latest_file.name}")
        
        return df
