"""Model training and prediction pipeline."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Config
from src.models.base import BaseModel, PredictionResult, prepare_features_for_modeling, prepare_targets
from src.models.poisson import PoissonBaselineModel
from src.models.xgboost_model import XGBoostModel


class ModelTrainer:
    """Train and manage multiple models."""
    
    def __init__(self, config: Config):
        """Initialize model trainer with configuration."""
        self.config = config
        self.models: Dict[str, BaseModel] = {}
        self.feature_columns: Optional[List[str]] = None
    
    def initialize_models(self):
        """Initialize all configured models."""
        if "baseline" in self.config.models.models:
            self.models["poisson_baseline"] = PoissonBaselineModel(max_goals=10)
        
        if "xgboost" in self.config.models.models:
            self.models["xgboost"] = XGBoostModel(params=self.config.models.xgboost_params)
        
        print(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def prepare_data(
        self,
        features: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare train/validation/test splits."""
        print("\n=== Preparing Data ===")
        
        # Prepare features
        X, self.feature_columns = prepare_features_for_modeling(features)
        y_home, y_away = prepare_targets(features)
        
        # Ensure alignment
        X = X.loc[y_home.index]
        
        print(f"Total samples: {len(X)}")
        print(f"Features: {len(self.feature_columns)}")
        
        # First split: train+val vs test
        X_trainval, X_test, y_home_trainval, y_home_test, y_away_trainval, y_away_test = train_test_split(
            X, y_home, y_away,
            test_size=test_size,
            random_state=self.config.models.random_state,
        )
        
        # Second split: train vs validation
        val_ratio = validation_size / (1 - test_size)
        X_train, X_val, y_home_train, y_home_val, y_away_train, y_away_val = train_test_split(
            X_trainval, y_home_trainval, y_away_trainval,
            test_size=val_ratio,
            random_state=self.config.models.random_state,
        )
        
        print(f"Train: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_home_train, y_home_val, y_home_test, y_away_train, y_away_val, y_away_test
    
    def train_all(self, features: pd.DataFrame):
        """Train all models."""
        print("\n=== Training Models ===")
        
        # Prepare data
        splits = self.prepare_data(
            features,
            test_size=self.config.models.test_size,
            validation_size=self.config.models.validation_size,
        )
        X_train, X_val, X_test, y_home_train, y_home_val, y_home_test, y_away_train, y_away_val, y_away_test = splits
        
        # Train each model
        for model_name, model in self.models.items():
            print(f"\n--- Training {model_name} ---")
            model.fit(X_train, y_home_train, y_away_train)
        
        print("\n✓ All models trained")
        
        # Return test set for evaluation
        return X_test, y_home_test, y_away_test
    
    def predict_all(self, X: pd.DataFrame) -> Dict[str, List[PredictionResult]]:
        """Generate predictions from all models."""
        predictions = {}
        
        for model_name, model in self.models.items():
            print(f"Predicting with {model_name}...")
            predictions[model_name] = model.predict(X)
        
        return predictions
    
    def save_predictions(
        self,
        predictions: Dict[str, List[PredictionResult]],
        features: pd.DataFrame,
    ):
        """Save predictions to Parquet."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        for model_name, preds in predictions.items():
            # Convert predictions to DataFrame
            records = []
            for pred in preds:
                # Calculate Over/Under for common thresholds
                over_2_5, under_2_5 = pred.get_over_under_prob(2.5)
                over_1_5, under_1_5 = pred.get_over_under_prob(1.5)
                over_3_5, under_3_5 = pred.get_over_under_prob(3.5)
                
                records.append({
                    "fixture_id": pred.fixture_id,
                    "model": model_name,
                    "home_goals_pred": pred.home_goals_pred,
                    "away_goals_pred": pred.away_goals_pred,
                    "total_goals_pred": pred.home_goals_pred + pred.away_goals_pred,
                    "over_1_5_prob": over_1_5,
                    "under_1_5_prob": under_1_5,
                    "over_2_5_prob": over_2_5,
                    "under_2_5_prob": under_2_5,
                    "over_3_5_prob": over_3_5,
                    "under_3_5_prob": under_3_5,
                })
            
            df_preds = pd.DataFrame(records)
            
            # Merge with match context
            df_preds = df_preds.merge(
                features[[
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
            
            # Save
            pred_path = self.config.paths.predictions / f"predictions_{model_name}_{timestamp}.parquet"
            df_preds.to_parquet(pred_path, index=False)
            print(f"Saved {len(df_preds)} predictions for {model_name} to {pred_path}")


class ModelPredictor:
    """Load trained models and generate predictions."""
    
    def __init__(self, config: Config):
        """Initialize predictor with configuration."""
        self.config = config
        self.models: Dict[str, BaseModel] = {}
    
    def load_models(self):
        """Load pre-trained models."""
        # Placeholder for model persistence
        # In a full implementation, models would be saved and loaded
        raise NotImplementedError("Model persistence not yet implemented")
    
    def predict(self, features: pd.DataFrame) -> Dict[str, List[PredictionResult]]:
        """Generate predictions for new data."""
        predictions = {}
        
        for model_name, model in self.models.items():
            X, _ = prepare_features_for_modeling(features)
            predictions[model_name] = model.predict(X)
        
        return predictions
