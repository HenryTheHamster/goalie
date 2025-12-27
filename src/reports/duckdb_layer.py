"""DuckDB query layer for data analysis."""

from pathlib import Path
from typing import List, Optional

import duckdb
import pandas as pd

from src.config import PathConfig


class DuckDBQueryLayer:
    """Query layer for analyzing predictions and features using DuckDB."""
    
    def __init__(self, paths: PathConfig, db_path: Optional[Path] = None):
        """Initialize DuckDB connection."""
        self.paths = paths
        self.db_path = db_path or Path("artifacts/goalie.duckdb")
        self.conn = duckdb.connect(str(self.db_path))
        
        # Register Parquet directories as views
        self._register_views()
    
    def _register_views(self):
        """Register Parquet directories as DuckDB views."""
        # Matches
        matches_pattern = str(self.paths.canonical_matches / "*.parquet")
        self.conn.execute(f"""
            CREATE OR REPLACE VIEW matches AS
            SELECT * FROM read_parquet('{matches_pattern}')
        """)
        
        # Features
        features_pattern = str(self.paths.features / "features_*.parquet")
        try:
            self.conn.execute(f"""
                CREATE OR REPLACE VIEW features AS
                SELECT * FROM read_parquet('{features_pattern}')
            """)
        except:
            # Features might not exist yet
            pass
        
        # Predictions
        predictions_pattern = str(self.paths.predictions / "predictions_*.parquet")
        try:
            self.conn.execute(f"""
                CREATE OR REPLACE VIEW predictions AS
                SELECT * FROM read_parquet('{predictions_pattern}')
            """)
        except:
            # Predictions might not exist yet
            pass
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        return self.conn.execute(sql).fetchdf()
    
    def get_match_summary(self) -> pd.DataFrame:
        """Get summary statistics for matches."""
        return self.query("""
            SELECT
                league_id,
                league_name,
                season,
                COUNT(*) as match_count,
                AVG(home_goals) as avg_home_goals,
                AVG(away_goals) as avg_away_goals,
                SUM(CASE WHEN home_goals > away_goals THEN 1 ELSE 0 END) as home_wins,
                SUM(CASE WHEN home_goals = away_goals THEN 1 ELSE 0 END) as draws,
                SUM(CASE WHEN home_goals < away_goals THEN 1 ELSE 0 END) as away_wins
            FROM matches
            GROUP BY league_id, league_name, season
            ORDER BY season DESC, league_id
        """)
    
    def get_team_statistics(self, team_id: Optional[int] = None) -> pd.DataFrame:
        """Get statistics for teams."""
        where_clause = f"WHERE home_team_id = {team_id} OR away_team_id = {team_id}" if team_id else ""
        
        return self.query(f"""
            WITH team_matches AS (
                SELECT
                    home_team_id as team_id,
                    home_team_name as team_name,
                    home_goals as goals_scored,
                    away_goals as goals_conceded,
                    CASE
                        WHEN home_goals > away_goals THEN 3
                        WHEN home_goals = away_goals THEN 1
                        ELSE 0
                    END as points
                FROM matches
                {where_clause}
                
                UNION ALL
                
                SELECT
                    away_team_id as team_id,
                    away_team_name as team_name,
                    away_goals as goals_scored,
                    home_goals as goals_conceded,
                    CASE
                        WHEN away_goals > home_goals THEN 3
                        WHEN away_goals = home_goals THEN 1
                        ELSE 0
                    END as points
                FROM matches
                {where_clause}
            )
            SELECT
                team_id,
                team_name,
                COUNT(*) as matches_played,
                SUM(points) as total_points,
                AVG(goals_scored) as avg_goals_scored,
                AVG(goals_conceded) as avg_goals_conceded,
                AVG(goals_scored - goals_conceded) as avg_goal_diff
            FROM team_matches
            GROUP BY team_id, team_name
            ORDER BY total_points DESC, avg_goal_diff DESC
        """)
    
    def get_prediction_performance(self, model: Optional[str] = None) -> pd.DataFrame:
        """Get prediction performance by model."""
        where_clause = f"WHERE model = '{model}'" if model else ""
        
        return self.query(f"""
            SELECT
                model,
                COUNT(*) as predictions_count,
                AVG(ABS(home_goals_pred - home_goals)) as mae_home,
                AVG(ABS(away_goals_pred - away_goals)) as mae_away,
                AVG(ABS(total_goals_pred - (home_goals + away_goals))) as mae_total,
                AVG(CASE
                    WHEN ABS(home_goals_pred - home_goals) <= 1 THEN 1.0
                    ELSE 0.0
                END) as accuracy_home_1,
                AVG(CASE
                    WHEN ABS(away_goals_pred - away_goals) <= 1 THEN 1.0
                    ELSE 0.0
                END) as accuracy_away_1
            FROM predictions
            {where_clause}
            GROUP BY model
            ORDER BY mae_total
        """)
    
    def get_over_under_performance(
        self,
        threshold: float = 2.5,
        model: Optional[str] = None
    ) -> pd.DataFrame:
        """Analyze Over/Under prediction performance."""
        where_clause = f"WHERE model = '{model}'" if model else ""
        
        return self.query(f"""
            WITH ou_analysis AS (
                SELECT
                    model,
                    fixture_id,
                    home_goals + away_goals as actual_total,
                    total_goals_pred as predicted_total,
                    over_{str(threshold).replace('.', '_')}_prob as over_prob,
                    under_{str(threshold).replace('.', '_')}_prob as under_prob,
                    CASE WHEN home_goals + away_goals > {threshold} THEN 1 ELSE 0 END as actual_over,
                    CASE WHEN over_{str(threshold).replace('.', '_')}_prob > under_{str(threshold).replace('.', '_')}_prob THEN 1 ELSE 0 END as predicted_over
                FROM predictions
                {where_clause}
            )
            SELECT
                model,
                COUNT(*) as total_predictions,
                AVG(CASE WHEN actual_over = predicted_over THEN 1.0 ELSE 0.0 END) as accuracy,
                AVG(over_prob) as avg_over_confidence,
                SUM(actual_over) as actual_overs,
                SUM(predicted_over) as predicted_overs
            FROM ou_analysis
            GROUP BY model
        """)
    
    def close(self):
        """Close DuckDB connection."""
        self.conn.close()
