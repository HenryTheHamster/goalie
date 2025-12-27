"""XGBoost model for goal prediction."""

from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson

from src.models.base import BaseModel, PredictionResult


class XGBoostModel(BaseModel):
    """XGBoost model for goal prediction with Poisson objective."""
    
    def __init__(self, params: dict = None):
        """Initialize XGBoost model."""
        super().__init__(name="xgboost")
        
        # Default parameters
        self.params = params or {
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "objective": "count:poisson",
            "random_state": 42,
        }
        
        # Separate models for home and away goals
        self.model_home = None
        self.model_away = None
        self.feature_columns = None
    
    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series):
        """Fit XGBoost models for home and away goals."""
        print(f"\n=== Fitting {self.name} ===")
        print(f"Training samples: {len(X)}")
        print(f"Features: {len(X.columns)}")
        
        # Store feature columns
        self.feature_columns = list(X.columns)
        
        # Fit home goals model
        print("\nFitting home goals model...")
        self.model_home = xgb.XGBRegressor(**self.params)
        self.model_home.fit(X, y_home, verbose=False)
        
        # Fit away goals model
        print("Fitting away goals model...")
        self.model_away = xgb.XGBRegressor(**self.params)
        self.model_away.fit(X, y_away, verbose=False)
        
        self.is_fitted = True
        print(f"\n✓ {self.name} fitted")
    
    def predict(self, X: pd.DataFrame) -> List[PredictionResult]:
        """Generate predictions using XGBoost models."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Ensure features match training
        X_pred = X[self.feature_columns].copy()
        
        # Predict expected goals
        home_preds = self.model_home.predict(X_pred)
        away_preds = self.model_away.predict(X_pred)
        
        # Convert to predictions with distributions
        predictions = []
        max_goals = 10
        
        for idx, (home_lambda, away_lambda) in enumerate(zip(home_preds, away_preds)):
            # Ensure positive lambdas
            home_lambda = max(0.1, home_lambda)
            away_lambda = max(0.1, away_lambda)
            
            # Generate Poisson distributions
            home_dist = np.array([
                poisson.pmf(k, home_lambda) for k in range(max_goals + 1)
            ])
            away_dist = np.array([
                poisson.pmf(k, away_lambda) for k in range(max_goals + 1)
            ])
            
            # Normalize
            home_dist = home_dist / home_dist.sum()
            away_dist = away_dist / away_dist.sum()
            
            predictions.append(PredictionResult(
                fixture_id=idx if "fixture_id" not in X.columns else X.iloc[idx].get("fixture_id", idx),
                home_goals_pred=float(home_lambda),
                away_goals_pred=float(away_lambda),
                home_goals_dist=home_dist,
                away_goals_dist=away_dist,
            ))
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from both models."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        home_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance_home": self.model_home.feature_importances_,
        })
        
        away_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance_away": self.model_away.feature_importances_,
        })
        
        # Merge and compute average importance
        importance = home_importance.merge(away_importance, on="feature")
        importance["importance_avg"] = (
            importance["importance_home"] + importance["importance_away"]
        ) / 2
        
        return importance.sort_values("importance_avg", ascending=False)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            **super().get_params(),
            **self.params,
        }
