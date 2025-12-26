"""Elo rating system for team strength estimation."""

import numpy as np
import pandas as pd

from src.config import FeatureGroup
from src.features.dictionary import FeatureDictionary


class EloRatingSystem:
    """Elo-style rating system with separate attack and defence ratings."""
    
    def __init__(
        self,
        k_factor: float = 20.0,
        initial_rating: float = 1500.0,
        home_advantage: float = 100.0,
    ):
        """Initialize Elo rating system."""
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.home_advantage = home_advantage
        
        # Track ratings over time: {team_id: {"attack": rating, "defence": rating}}
        self.ratings: dict = {}
    
    def get_team_ratings(self, team_id: int) -> dict:
        """Get current attack and defence ratings for a team."""
        if team_id not in self.ratings:
            self.ratings[team_id] = {
                "attack": self.initial_rating,
                "defence": self.initial_rating,
            }
        return self.ratings[team_id]
    
    def expected_goals(self, attack_rating: float, defence_rating: float) -> float:
        """Calculate expected goals based on attack vs defence rating difference."""
        rating_diff = attack_rating - defence_rating
        # Sigmoid-like function mapped to reasonable goal range
        expected = 1.5 + (rating_diff / 400.0)
        return max(0.1, expected)  # Minimum expected goals
    
    def update_ratings(
        self,
        home_team_id: int,
        away_team_id: int,
        home_goals: int,
        away_goals: int,
        is_home_game: bool = True,
    ):
        """Update Elo ratings based on match result."""
        # Get current ratings
        home_ratings = self.get_team_ratings(home_team_id)
        away_ratings = self.get_team_ratings(away_team_id)
        
        # Apply home advantage
        home_attack_adj = home_ratings["attack"] + (self.home_advantage if is_home_game else 0)
        
        # Calculate expected goals
        home_expected = self.expected_goals(home_attack_adj, away_ratings["defence"])
        away_expected = self.expected_goals(away_ratings["attack"], home_ratings["defence"])
        
        # Goal difference as performance indicator
        home_goal_diff = home_goals - home_expected
        away_goal_diff = away_goals - away_expected
        
        # Update attack ratings
        self.ratings[home_team_id]["attack"] += self.k_factor * home_goal_diff
        self.ratings[away_team_id]["attack"] += self.k_factor * away_goal_diff
        
        # Update defence ratings (inverse - conceding goals reduces rating)
        self.ratings[home_team_id]["defence"] -= self.k_factor * away_goal_diff
        self.ratings[away_team_id]["defence"] -= self.k_factor * home_goal_diff
    
    def compute_features(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Compute Elo features for all matches (respecting temporal order)."""
        # Ensure matches are sorted by time
        matches = matches.sort_values("match_datetime_utc").reset_index(drop=True)
        
        # Initialize feature columns
        features = []
        
        for idx, match in matches.iterrows():
            fixture_id = match["fixture_id"]
            home_team_id = match["home_team_id"]
            away_team_id = match["away_team_id"]
            
            # Get ratings BEFORE this match (no leakage)
            home_ratings = self.get_team_ratings(home_team_id)
            away_ratings = self.get_team_ratings(away_team_id)
            
            # Store features
            features.append({
                "fixture_id": fixture_id,
                "elo_home_attack": home_ratings["attack"],
                "elo_home_defence": home_ratings["defence"],
                "elo_away_attack": away_ratings["attack"],
                "elo_away_defence": away_ratings["defence"],
                "elo_home_attack_adj": home_ratings["attack"] + self.home_advantage,
                "elo_attack_diff": home_ratings["attack"] - away_ratings["attack"],
                "elo_defence_diff": home_ratings["defence"] - away_ratings["defence"],
            })
            
            # Update ratings AFTER recording features (no leakage)
            if pd.notna(match["home_goals"]) and pd.notna(match["away_goals"]):
                self.update_ratings(
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    home_goals=int(match["home_goals"]),
                    away_goals=int(match["away_goals"]),
                    is_home_game=True,
                )
        
        return pd.DataFrame(features)
    
    @staticmethod
    def register_features(dictionary: FeatureDictionary):
        """Register Elo features in the feature dictionary."""
        group = FeatureGroup.ELO.value
        
        dictionary.register(
            "elo_home_attack",
            "Elo attack rating for home team (before match)",
            group,
            "float",
        )
        dictionary.register(
            "elo_home_defence",
            "Elo defence rating for home team (before match)",
            group,
            "float",
        )
        dictionary.register(
            "elo_away_attack",
            "Elo attack rating for away team (before match)",
            group,
            "float",
        )
        dictionary.register(
            "elo_away_defence",
            "Elo defence rating for away team (before match)",
            group,
            "float",
        )
        dictionary.register(
            "elo_home_attack_adj",
            "Elo attack rating for home team with home advantage adjustment",
            group,
            "float",
        )
        dictionary.register(
            "elo_attack_diff",
            "Difference between home and away attack ratings",
            group,
            "float",
        )
        dictionary.register(
            "elo_defence_diff",
            "Difference between home and away defence ratings",
            group,
            "float",
        )
