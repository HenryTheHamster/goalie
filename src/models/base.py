"""Base model interface and utilities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    """Structured prediction result."""
    
    fixture_id: int
    home_goals_pred: float
    away_goals_pred: float
    home_goals_dist: Optional[np.ndarray] = None  # Probability distribution over goals [0, 1, 2, ...]
    away_goals_dist: Optional[np.ndarray] = None
    
    def get_over_under_prob(self, threshold: float = 2.5) -> Tuple[float, float]:
        """Calculate Over/Under probabilities for total goals.
        
        For threshold 2.5:
        - Under 2.5: includes 0, 1, 2 goals (indices 0, 1, 2)
        - Over 2.5: includes 3+ goals (indices 3, 4, ...)
        
        The calculation int(threshold) + 1 = 3, so [:3] gives us indices 0, 1, 2 which is correct.
        """
        total_goals_mean = self.home_goals_pred + self.away_goals_pred
        
        # If we have distributions, use them
        if self.home_goals_dist is not None and self.away_goals_dist is not None:
            # Convolve distributions to get total goals distribution
            total_dist = np.convolve(self.home_goals_dist, self.away_goals_dist)
            
            # Calculate Under probability (up to and including floor(threshold))
            # For 2.5, this is goals 0, 1, 2 (indices [:3])
            under_prob = total_dist[:int(threshold) + 1].sum()
            over_prob = 1.0 - under_prob
            
            return over_prob, under_prob
        
        # Fallback: simple threshold comparison
        # This is approximate and should be improved
        over_prob = 1.0 if total_goals_mean > threshold else 0.0
        under_prob = 1.0 - over_prob
        
        return over_prob, under_prob


class BaseModel(ABC):
    """Abstract base class for goal prediction models."""
    
    def __init__(self, name: str):
        """Initialize model with a name."""
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series):
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> List[PredictionResult]:
        """Generate predictions for given features."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {"name": self.name}


def prepare_features_for_modeling(
    features: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare features for modeling by handling missing values and selecting columns."""
    if feature_columns is None:
        # Auto-select numeric columns, excluding IDs and targets
        exclude_cols = [
            "fixture_id",
            "match_datetime_utc",
            "season",
            "league_id",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
            "home_goals_halftime",
            "away_goals_halftime",
            "canonicalized_at_utc",
        ]
        
        feature_columns = [
            col for col in features.columns
            if col not in exclude_cols and features[col].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
    
    # Select feature columns
    X = features[feature_columns].copy()
    
    # Handle missing values (fill with 0 for now - could be improved)
    X = X.fillna(0)
    
    return X, feature_columns


def prepare_targets(features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Prepare target variables (home and away goals)."""
    y_home = features["home_goals"].copy()
    y_away = features["away_goals"].copy()
    
    # Remove rows with missing targets
    valid_mask = y_home.notna() & y_away.notna()
    
    return y_home[valid_mask], y_away[valid_mask]
