"""Evaluation metrics for goal prediction models."""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.base import PredictionResult


class MetricsCalculator:
    """Calculate evaluation metrics for predictions."""
    
    @staticmethod
    def calculate_mae(y_true: pd.Series, y_pred: List[float]) -> float:
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def calculate_rmse(y_true: pd.Series, y_pred: List[float]) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def calculate_poisson_log_likelihood(y_true: pd.Series, lambdas: List[float]) -> float:
        """Calculate Poisson log-likelihood (higher is better)."""
        # Pre-process lambdas to ensure they're positive (more efficient)
        lambdas_safe = np.maximum(0.001, np.array(lambdas))
        y_true_array = np.array(y_true, dtype=int)
        
        # Vectorized calculation
        ll = poisson.logpmf(y_true_array, lambdas_safe).sum()
        return float(ll)
    
    @staticmethod
    def calculate_accuracy_within_n(y_true: pd.Series, y_pred: List[float], n: int = 1) -> float:
        """Calculate accuracy of predictions within n goals."""
        errors = np.abs(np.array(y_pred) - np.array(y_true))
        return (errors <= n).mean()
    
    @staticmethod
    def calculate_calibration_bins(
        y_true: pd.Series,
        predictions: List[PredictionResult],
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """Calculate calibration curve for goal predictions."""
        # Extract predicted probabilities for actual outcomes
        calibration_data = []
        
        for true_val, pred in zip(y_true, predictions):
            true_val = int(true_val)
            
            # Home goals calibration
            if pred.home_goals_dist is not None and true_val < len(pred.home_goals_dist):
                calibration_data.append({
                    "predicted_prob": pred.home_goals_dist[true_val],
                    "occurred": 1.0,
                    "type": "home",
                })
            
            # Away goals calibration (if applicable)
            if pred.away_goals_dist is not None and true_val < len(pred.away_goals_dist):
                calibration_data.append({
                    "predicted_prob": pred.away_goals_dist[true_val],
                    "occurred": 1.0,
                    "type": "away",
                })
        
        if not calibration_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(calibration_data)
        
        # Bin predictions
        df["bin"] = pd.cut(df["predicted_prob"], bins=n_bins, labels=False)
        
        # Calculate calibration per bin
        calibration = df.groupby("bin").agg({
            "predicted_prob": "mean",
            "occurred": "mean",
        }).reset_index()
        
        return calibration
    
    @staticmethod
    def calculate_brier_score(
        y_true: pd.Series,
        predictions: List[PredictionResult],
    ) -> float:
        """Calculate Brier score for probabilistic predictions."""
        scores = []
        
        for true_val, pred in zip(y_true, predictions):
            true_val = int(true_val)
            
            # For home goals
            if pred.home_goals_dist is not None:
                # Create one-hot encoded true distribution
                true_dist = np.zeros_like(pred.home_goals_dist)
                if true_val < len(true_dist):
                    true_dist[true_val] = 1.0
                
                # Calculate squared difference
                brier = ((pred.home_goals_dist - true_dist) ** 2).sum()
                scores.append(brier)
        
        return np.mean(scores) if scores else np.nan
    
    @staticmethod
    def evaluate_model(
        predictions: List[PredictionResult],
        y_home_true: pd.Series,
        y_away_true: pd.Series,
    ) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        # Extract predicted values
        home_preds = [p.home_goals_pred for p in predictions]
        away_preds = [p.away_goals_pred for p in predictions]
        
        metrics = {
            # Home goals metrics
            "mae_home": MetricsCalculator.calculate_mae(y_home_true, home_preds),
            "rmse_home": MetricsCalculator.calculate_rmse(y_home_true, home_preds),
            "ll_home": MetricsCalculator.calculate_poisson_log_likelihood(y_home_true, home_preds),
            "acc_1_home": MetricsCalculator.calculate_accuracy_within_n(y_home_true, home_preds, n=1),
            
            # Away goals metrics
            "mae_away": MetricsCalculator.calculate_mae(y_away_true, away_preds),
            "rmse_away": MetricsCalculator.calculate_rmse(y_away_true, away_preds),
            "ll_away": MetricsCalculator.calculate_poisson_log_likelihood(y_away_true, away_preds),
            "acc_1_away": MetricsCalculator.calculate_accuracy_within_n(y_away_true, away_preds, n=1),
            
            # Total goals metrics
            "mae_total": MetricsCalculator.calculate_mae(
                y_home_true + y_away_true,
                [h + a for h, a in zip(home_preds, away_preds)],
            ),
            
            # Probabilistic metrics
            "brier_score": MetricsCalculator.calculate_brier_score(y_home_true, predictions),
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(model_name: str, metrics: Dict[str, float]):
        """Pretty print metrics."""
        print(f"\n=== {model_name} Metrics ===")
        print(f"Home Goals:")
        print(f"  MAE: {metrics['mae_home']:.3f}")
        print(f"  RMSE: {metrics['rmse_home']:.3f}")
        print(f"  Log-Likelihood: {metrics['ll_home']:.2f}")
        print(f"  Accuracy (±1): {metrics['acc_1_home']:.2%}")
        
        print(f"\nAway Goals:")
        print(f"  MAE: {metrics['mae_away']:.3f}")
        print(f"  RMSE: {metrics['rmse_away']:.3f}")
        print(f"  Log-Likelihood: {metrics['ll_away']:.2f}")
        print(f"  Accuracy (±1): {metrics['acc_1_away']:.2%}")
        
        print(f"\nTotal Goals:")
        print(f"  MAE: {metrics['mae_total']:.3f}")
        
        print(f"\nProbabilistic:")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")


class BacktestSimulator:
    """Simulate P&L backtest (diagnostic only, not for actual trading)."""
    
    @staticmethod
    def simulate_simple_pl(
        predictions: List[PredictionResult],
        y_home_true: pd.Series,
        y_away_true: pd.Series,
        stake: float = 10.0,
    ) -> Dict[str, float]:
        """Simulate simple P&L for Over/Under 2.5 betting."""
        print("\n⚠ WARNING: P&L backtest is DIAGNOSTIC ONLY")
        print("   Do NOT use for actual trading decisions")
        
        total_bets = 0
        correct_bets = 0
        total_pl = 0.0
        
        for pred, h_true, a_true in zip(predictions, y_home_true, y_away_true):
            total_goals = h_true + a_true
            
            # Get Over/Under 2.5 probabilities
            over_prob, under_prob = pred.get_over_under_prob(2.5)
            
            # Simple strategy: bet on side with >60% confidence
            if over_prob > 0.6:
                total_bets += 1
                if total_goals > 2.5:
                    correct_bets += 1
                    # Simplified: assume 1.9 odds
                    total_pl += stake * 0.9
                else:
                    total_pl -= stake
            
            elif under_prob > 0.6:
                total_bets += 1
                if total_goals < 2.5:
                    correct_bets += 1
                    total_pl += stake * 0.9
                else:
                    total_pl -= stake
        
        if total_bets == 0:
            return {
                "total_bets": 0,
                "correct_bets": 0,
                "accuracy": 0.0,
                "total_pl": 0.0,
                "roi": 0.0,
            }
        
        return {
            "total_bets": total_bets,
            "correct_bets": correct_bets,
            "accuracy": correct_bets / total_bets,
            "total_pl": total_pl,
            "roi": total_pl / (total_bets * stake),
        }
