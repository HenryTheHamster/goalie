"""Configuration management for Goalie prediction framework."""

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class MatchStatus(str, Enum):
    """Match status values from API-Football."""
    
    # Valid for training (completed matches)
    FT = "FT"  # Full Time
    AET = "AET"  # After Extra Time
    PEN = "PEN"  # Penalty Shootout
    
    # Invalid for training (not completed)
    TBD = "TBD"  # To Be Defined
    NS = "NS"  # Not Started
    LIVE = "LIVE"  # In Progress
    PST = "PST"  # Postponed
    CANC = "CANC"  # Cancelled
    ABD = "ABD"  # Abandoned
    SUSP = "SUSP"  # Suspended
    INT = "INT"  # Interrupted
    AWD = "AWD"  # Technical Loss
    WO = "WO"  # WalkOver
    
    @classmethod
    def training_statuses(cls) -> List[str]:
        """Return statuses valid for training."""
        return [cls.FT.value, cls.AET.value, cls.PEN.value]
    
    @classmethod
    def is_training_status(cls, status: str) -> bool:
        """Check if status is valid for training."""
        return status in cls.training_statuses()


class FeatureGroup(str, Enum):
    """Feature groups for ablation studies."""
    
    ELO = "elo"  # Elo-style attack/defence ratings
    EWMA = "ewma"  # Exponentially weighted moving averages
    ROLLING = "rolling"  # Rolling average statistics
    ODDS = "odds"  # Betting odds (LEAKAGE RISK - must be explicit)
    HEAD_TO_HEAD = "h2h"  # Head-to-head statistics
    FORM = "form"  # Recent form indicators
    VENUE = "venue"  # Home/away venue statistics


class SnapshotCutoff(str, Enum):
    """Pre-match snapshot timing cutoffs."""
    
    T_7D = "T-7d"  # 7 days before match
    T_3D = "T-3d"  # 3 days before match
    T_24H = "T-24h"  # 24 hours before match
    T_6H = "T-6h"  # 6 hours before match
    T_1H = "T-1h"  # 1 hour before match
    
    def to_timedelta(self) -> timedelta:
        """Convert cutoff to timedelta."""
        mapping = {
            self.T_7D: timedelta(days=7),
            self.T_3D: timedelta(days=3),
            self.T_24H: timedelta(hours=24),
            self.T_6H: timedelta(hours=6),
            self.T_1H: timedelta(hours=1),
        }
        return mapping[self]


@dataclass
class LeagueConfig:
    """Configuration for a specific league."""
    
    league_id: int
    name: str
    country: str
    seasons: List[int]  # List of years, e.g., [2022, 2023, 2024]
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.league_id:
            raise ValueError("league_id is required")
        if not self.seasons:
            raise ValueError("At least one season is required")


@dataclass
class APIFootballConfig:
    """Configuration for API-Football integration."""
    
    api_key: str
    base_url: str = "https://v3.football.api-sports.io"
    timeout: int = 30
    rate_limit_delay: float = 1.0  # seconds between requests
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.api_key:
            raise ValueError("API key is required")
        if not self.api_key.startswith("test-") and len(self.api_key) < 20:
            raise ValueError("Invalid API key format")


@dataclass
class PathConfig:
    """Configuration for data paths."""
    
    root: Path = Path("artifacts")
    
    @property
    def raw_fixtures(self) -> Path:
        return self.root / "parquet" / "raw" / "api_football" / "fixtures"
    
    @property
    def raw_teams(self) -> Path:
        return self.root / "parquet" / "raw" / "api_football" / "teams"
    
    @property
    def raw_leagues(self) -> Path:
        return self.root / "parquet" / "raw" / "api_football" / "leagues"
    
    @property
    def canonical_matches(self) -> Path:
        return self.root / "parquet" / "canonical" / "matches"
    
    @property
    def canonical_fixtures(self) -> Path:
        return self.root / "parquet" / "canonical" / "fixtures"
    
    @property
    def canonical_teams(self) -> Path:
        return self.root / "parquet" / "canonical" / "teams"
    
    @property
    def canonical_leagues(self) -> Path:
        return self.root / "parquet" / "canonical" / "leagues"
    
    @property
    def features(self) -> Path:
        return self.root / "parquet" / "features"
    
    @property
    def predictions(self) -> Path:
        return self.root / "parquet" / "predictions"
    
    @property
    def metrics(self) -> Path:
        return self.root / "parquet" / "metrics"
    
    @property
    def runs(self) -> Path:
        return self.root / "runs"
    
    def ensure_directories(self):
        """Create all necessary directories."""
        for path_attr in dir(self):
            if not path_attr.startswith("_") and path_attr != "root":
                attr = getattr(self, path_attr)
                if isinstance(attr, Path):
                    attr.mkdir(parents=True, exist_ok=True)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    enabled_groups: List[FeatureGroup] = field(
        default_factory=lambda: [
            FeatureGroup.ELO,
            FeatureGroup.EWMA,
            FeatureGroup.ROLLING,
            FeatureGroup.FORM,
            FeatureGroup.VENUE,
        ]
    )
    
    # Elo configuration
    elo_k_factor: float = 20.0
    elo_initial_rating: float = 1500.0
    elo_home_advantage: float = 100.0
    
    # EWMA configuration
    ewma_alpha: float = 0.3  # Weight for recent observations
    
    # Rolling window configuration
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Minimum matches for feature calculation
    min_matches_for_features: int = 5
    
    def is_group_enabled(self, group: FeatureGroup) -> bool:
        """Check if a feature group is enabled."""
        return group in self.enabled_groups
    
    def validate_no_odds_leakage(self):
        """Ensure odds are not used unless explicitly enabled."""
        if FeatureGroup.ODDS in self.enabled_groups:
            raise ValueError(
                "LEAKAGE PROTECTION: odds feature group is enabled. "
                "This must be explicitly allowed with --feature-group odds flag."
            )


@dataclass
class ModelConfig:
    """Configuration for modeling."""
    
    models: List[str] = field(default_factory=lambda: ["baseline", "xgboost", "bayesian"])
    
    # Model-specific parameters
    xgboost_params: Dict = field(
        default_factory=lambda: {
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "objective": "count:poisson",
            "random_state": 42,
        }
    )
    
    # Validation split
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42


@dataclass
class Config:
    """Main configuration for Goalie framework."""
    
    # Sub-configurations
    api: APIFootballConfig
    paths: PathConfig
    features: FeatureConfig
    models: ModelConfig
    
    # League configurations
    leagues: List[LeagueConfig] = field(
        default_factory=lambda: [
            LeagueConfig(
                league_id=39,
                name="Premier League",
                country="England",
                seasons=[2022, 2023, 2024],
            ),
            LeagueConfig(
                league_id=40,
                name="Championship",
                country="England",
                seasons=[2022, 2023, 2024],
            ),
        ]
    )
    
    # Snapshot cutoffs
    snapshot_cutoffs: List[SnapshotCutoff] = field(
        default_factory=lambda: [
            SnapshotCutoff.T_7D,
            SnapshotCutoff.T_3D,
            SnapshotCutoff.T_24H,
            SnapshotCutoff.T_6H,
            SnapshotCutoff.T_1H,
        ]
    )
    
    # Leakage protection
    enforce_leakage_protection: bool = True
    
    # Run metadata
    run_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        self.paths.ensure_directories()
        
        if self.enforce_leakage_protection:
            self.features.validate_no_odds_leakage()
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Note: Simplified - full implementation would parse nested configs
        raise NotImplementedError("YAML loading to be implemented")
    
    @classmethod
    def default(cls, api_key: str) -> "Config":
        """Create default configuration."""
        return cls(
            api=APIFootballConfig(api_key=api_key),
            paths=PathConfig(),
            features=FeatureConfig(),
            models=ModelConfig(),
        )
