"""Rolling statistics features (EWMA and simple rolling averages)."""

import numpy as np
import pandas as pd

from src.config import FeatureGroup
from src.features.dictionary import FeatureDictionary


class RollingStatsCalculator:
    """Calculate rolling statistics respecting temporal order and preventing leakage."""
    
    def __init__(
        self,
        ewma_alpha: float = 0.3,
        rolling_windows: list = None,
        min_matches: int = 5,
    ):
        """Initialize rolling stats calculator."""
        self.ewma_alpha = ewma_alpha
        self.rolling_windows = rolling_windows or [5, 10, 20]
        self.min_matches = min_matches
    
    def compute_ewma_features(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Compute EWMA (Exponentially Weighted Moving Average) features."""
        # Ensure matches are sorted by time
        matches = matches.sort_values("match_datetime_utc").reset_index(drop=True)
        
        # Group by team and compute EWMA for each team separately
        features = []
        
        # Track stats for each team: {team_id: {metric: values}}
        team_stats: dict = {}
        
        for idx, match in matches.iterrows():
            fixture_id = match["fixture_id"]
            home_team_id = match["home_team_id"]
            away_team_id = match["away_team_id"]
            
            # Initialize team stats if needed
            if home_team_id not in team_stats:
                team_stats[home_team_id] = {"goals_scored": [], "goals_conceded": []}
            if away_team_id not in team_stats:
                team_stats[away_team_id] = {"goals_scored": [], "goals_conceded": []}
            
            # Compute EWMA for home team (before this match - no leakage)
            home_goals_scored_ewma = self._compute_ewma(team_stats[home_team_id]["goals_scored"])
            home_goals_conceded_ewma = self._compute_ewma(team_stats[home_team_id]["goals_conceded"])
            
            # Compute EWMA for away team (before this match - no leakage)
            away_goals_scored_ewma = self._compute_ewma(team_stats[away_team_id]["goals_scored"])
            away_goals_conceded_ewma = self._compute_ewma(team_stats[away_team_id]["goals_conceded"])
            
            features.append({
                "fixture_id": fixture_id,
                "ewma_home_goals_scored": home_goals_scored_ewma,
                "ewma_home_goals_conceded": home_goals_conceded_ewma,
                "ewma_away_goals_scored": away_goals_scored_ewma,
                "ewma_away_goals_conceded": away_goals_conceded_ewma,
                "ewma_home_goal_diff": (
                    home_goals_scored_ewma - home_goals_conceded_ewma
                    if home_goals_scored_ewma is not None
                    else None
                ),
                "ewma_away_goal_diff": (
                    away_goals_scored_ewma - away_goals_conceded_ewma
                    if away_goals_scored_ewma is not None
                    else None
                ),
            })
            
            # Update team stats AFTER recording features (no leakage)
            if pd.notna(match["home_goals"]) and pd.notna(match["away_goals"]):
                home_goals = int(match["home_goals"])
                away_goals = int(match["away_goals"])
                
                team_stats[home_team_id]["goals_scored"].append(home_goals)
                team_stats[home_team_id]["goals_conceded"].append(away_goals)
                
                team_stats[away_team_id]["goals_scored"].append(away_goals)
                team_stats[away_team_id]["goals_conceded"].append(home_goals)
        
        return pd.DataFrame(features)
    
    def _compute_ewma(self, values: list) -> float:
        """Compute exponentially weighted moving average."""
        if len(values) < self.min_matches:
            return None
        
        if not values:
            return None
        
        # Compute EWMA manually
        ewma = values[0]
        for value in values[1:]:
            ewma = self.ewma_alpha * value + (1 - self.ewma_alpha) * ewma
        
        return float(ewma)
    
    def compute_rolling_features(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling average features for multiple windows."""
        # Ensure matches are sorted by time
        matches = matches.sort_values("match_datetime_utc").reset_index(drop=True)
        
        features = []
        
        # Track stats for each team
        team_stats: dict = {}
        
        for idx, match in matches.iterrows():
            fixture_id = match["fixture_id"]
            home_team_id = match["home_team_id"]
            away_team_id = match["away_team_id"]
            
            # Initialize team stats if needed
            if home_team_id not in team_stats:
                team_stats[home_team_id] = {"goals_scored": [], "goals_conceded": [], "points": []}
            if away_team_id not in team_stats:
                team_stats[away_team_id] = {"goals_scored": [], "goals_conceded": [], "points": []}
            
            # Compute rolling averages for each window
            feature_row = {"fixture_id": fixture_id}
            
            for window in self.rolling_windows:
                # Home team
                home_goals_scored = self._compute_rolling_avg(
                    team_stats[home_team_id]["goals_scored"], window
                )
                home_goals_conceded = self._compute_rolling_avg(
                    team_stats[home_team_id]["goals_conceded"], window
                )
                home_points = self._compute_rolling_avg(
                    team_stats[home_team_id]["points"], window
                )
                
                # Away team
                away_goals_scored = self._compute_rolling_avg(
                    team_stats[away_team_id]["goals_scored"], window
                )
                away_goals_conceded = self._compute_rolling_avg(
                    team_stats[away_team_id]["goals_conceded"], window
                )
                away_points = self._compute_rolling_avg(
                    team_stats[away_team_id]["points"], window
                )
                
                feature_row.update({
                    f"rolling_{window}_home_goals_scored": home_goals_scored,
                    f"rolling_{window}_home_goals_conceded": home_goals_conceded,
                    f"rolling_{window}_home_points": home_points,
                    f"rolling_{window}_away_goals_scored": away_goals_scored,
                    f"rolling_{window}_away_goals_conceded": away_goals_conceded,
                    f"rolling_{window}_away_points": away_points,
                })
            
            features.append(feature_row)
            
            # Update team stats AFTER recording features (no leakage)
            if pd.notna(match["home_goals"]) and pd.notna(match["away_goals"]):
                home_goals = int(match["home_goals"])
                away_goals = int(match["away_goals"])
                
                # Determine points (3 for win, 1 for draw, 0 for loss)
                if home_goals > away_goals:
                    home_points, away_points = 3, 0
                elif home_goals < away_goals:
                    home_points, away_points = 0, 3
                else:
                    home_points, away_points = 1, 1
                
                team_stats[home_team_id]["goals_scored"].append(home_goals)
                team_stats[home_team_id]["goals_conceded"].append(away_goals)
                team_stats[home_team_id]["points"].append(home_points)
                
                team_stats[away_team_id]["goals_scored"].append(away_goals)
                team_stats[away_team_id]["goals_conceded"].append(home_goals)
                team_stats[away_team_id]["points"].append(away_points)
        
        return pd.DataFrame(features)
    
    def _compute_rolling_avg(self, values: list, window: int) -> float:
        """Compute rolling average for a specific window."""
        if len(values) < self.min_matches:
            return None
        
        if not values:
            return None
        
        # Take last 'window' values
        recent_values = values[-window:]
        return float(np.mean(recent_values))
    
    @staticmethod
    def register_ewma_features(dictionary: FeatureDictionary):
        """Register EWMA features in the feature dictionary."""
        group = FeatureGroup.EWMA.value
        
        dictionary.register(
            "ewma_home_goals_scored",
            "EWMA of goals scored by home team (before match)",
            group,
            "float",
            nullable=True,
        )
        dictionary.register(
            "ewma_home_goals_conceded",
            "EWMA of goals conceded by home team (before match)",
            group,
            "float",
            nullable=True,
        )
        dictionary.register(
            "ewma_away_goals_scored",
            "EWMA of goals scored by away team (before match)",
            group,
            "float",
            nullable=True,
        )
        dictionary.register(
            "ewma_away_goals_conceded",
            "EWMA of goals conceded by away team (before match)",
            group,
            "float",
            nullable=True,
        )
        dictionary.register(
            "ewma_home_goal_diff",
            "EWMA of goal difference for home team",
            group,
            "float",
            nullable=True,
        )
        dictionary.register(
            "ewma_away_goal_diff",
            "EWMA of goal difference for away team",
            group,
            "float",
            nullable=True,
        )
    
    @staticmethod
    def register_rolling_features(dictionary: FeatureDictionary, windows: list):
        """Register rolling average features in the feature dictionary."""
        group = FeatureGroup.ROLLING.value
        
        for window in windows:
            dictionary.register(
                f"rolling_{window}_home_goals_scored",
                f"Rolling {window}-match average of goals scored by home team (before match)",
                group,
                "float",
                nullable=True,
            )
            dictionary.register(
                f"rolling_{window}_home_goals_conceded",
                f"Rolling {window}-match average of goals conceded by home team (before match)",
                group,
                "float",
                nullable=True,
            )
            dictionary.register(
                f"rolling_{window}_home_points",
                f"Rolling {window}-match average of points earned by home team (before match)",
                group,
                "float",
                nullable=True,
            )
            dictionary.register(
                f"rolling_{window}_away_goals_scored",
                f"Rolling {window}-match average of goals scored by away team (before match)",
                group,
                "float",
                nullable=True,
            )
            dictionary.register(
                f"rolling_{window}_away_goals_conceded",
                f"Rolling {window}-match average of goals conceded by away team (before match)",
                group,
                "float",
                nullable=True,
            )
            dictionary.register(
                f"rolling_{window}_away_points",
                f"Rolling {window}-match average of points earned by away team (before match)",
                group,
                "float",
                nullable=True,
            )
