"""Poisson baseline model for goal prediction."""

from typing import List

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.models.base import BaseModel, PredictionResult


class PoissonBaselineModel(BaseModel):
    """Statistical baseline using Poisson distribution for goals."""
    
    def __init__(self, max_goals: int = 10):
        """Initialize Poisson baseline model."""
        super().__init__(name="poisson_baseline")
        self.max_goals = max_goals
        
        # Historical averages (will be computed during fit)
        self.home_avg = None
        self.away_avg = None
        self.global_home_avg = None
        self.global_away_avg = None
    
    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series):
        """Fit the model by computing historical averages."""
        print(f"\n=== Fitting {self.name} ===")
        
        # Compute global averages
        self.global_home_avg = y_home.mean()
        self.global_away_avg = y_away.mean()
        
        print(f"Global home average: {self.global_home_avg:.3f}")
        print(f"Global away average: {self.global_away_avg:.3f}")
        
        # Store historical data for team-specific adjustments (if team IDs available)
        # For now, using global averages
        self.is_fitted = True
        
        print(f"✓ {self.name} fitted")
    
    def predict(self, X: pd.DataFrame) -> List[PredictionResult]:
        """Generate Poisson predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        for idx, row in X.iterrows():
            # Use global averages (can be enhanced with team-specific adjustments)
            home_lambda = self.global_home_avg
            away_lambda = self.global_away_avg
            
            # Adjust based on features if available
            if "elo_attack_diff" in X.columns and pd.notna(row.get("elo_attack_diff")):
                # Adjust lambda based on Elo rating difference
                elo_diff = row["elo_attack_diff"]
                home_lambda *= (1 + elo_diff / 1000)  # Adjust by rating difference
                away_lambda *= (1 - elo_diff / 1000)
            
            # Ensure positive lambdas
            home_lambda = max(0.1, home_lambda)
            away_lambda = max(0.1, away_lambda)
            
            # Generate goal distributions
            home_dist = np.array([
                poisson.pmf(k, home_lambda) for k in range(self.max_goals + 1)
            ])
            away_dist = np.array([
                poisson.pmf(k, away_lambda) for k in range(self.max_goals + 1)
            ])
            
            # Normalize distributions
            home_dist = home_dist / home_dist.sum()
            away_dist = away_dist / away_dist.sum()
            
            predictions.append(PredictionResult(
                fixture_id=idx if "fixture_id" not in X.columns else row.get("fixture_id", idx),
                home_goals_pred=home_lambda,
                away_goals_pred=away_lambda,
                home_goals_dist=home_dist,
                away_goals_dist=away_dist,
            ))
        
        return predictions
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            **super().get_params(),
            "max_goals": self.max_goals,
            "global_home_avg": self.global_home_avg,
            "global_away_avg": self.global_away_avg,
        }
