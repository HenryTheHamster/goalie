"""Markdown report generation."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.config import Config
from src.reports.duckdb_layer import DuckDBQueryLayer


class MarkdownReportGenerator:
    """Generate comprehensive Markdown reports for model runs."""
    
    def __init__(self, config: Config):
        """Initialize report generator."""
        self.config = config
        self.query_layer = DuckDBQueryLayer(config.paths)
    
    def generate_report(
        self,
        run_id: str,
        metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> str:
        """Generate comprehensive Markdown report."""
        report_lines = [
            f"# Goalie Prediction Framework - Run Report",
            f"",
            f"**Run ID:** `{run_id}`",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"",
            f"---",
            f"",
        ]
        
        # Configuration summary
        report_lines.extend(self._configuration_section())
        
        # Data summary
        report_lines.extend(self._data_summary_section())
        
        # Feature summary
        report_lines.extend(self._feature_summary_section())
        
        # Model performance
        if metrics:
            report_lines.extend(self._model_performance_section(metrics))
        
        # Prediction analysis
        report_lines.extend(self._prediction_analysis_section())
        
        return "\n".join(report_lines)
    
    def _configuration_section(self) -> list:
        """Generate configuration section."""
        lines = [
            "## Configuration",
            "",
            "### Leagues",
            "",
        ]
        
        for league in self.config.leagues:
            lines.append(f"- **{league.name}** (ID: {league.league_id})")
            lines.append(f"  - Seasons: {', '.join(map(str, league.seasons))}")
        
        lines.extend([
            "",
            "### Feature Groups",
            "",
        ])
        
        for group in self.config.features.enabled_groups:
            lines.append(f"- {group.value}")
        
        lines.extend([
            "",
            "### Models",
            "",
        ])
        
        for model in self.config.models.models:
            lines.append(f"- {model}")
        
        lines.extend(["", "---", ""])
        
        return lines
    
    def _data_summary_section(self) -> list:
        """Generate data summary section."""
        lines = [
            "## Data Summary",
            "",
        ]
        
        try:
            match_summary = self.query_layer.get_match_summary()
            
            lines.append("### Matches by League and Season")
            lines.append("")
            lines.append(match_summary.to_markdown(index=False))
            lines.append("")
            
            # Overall statistics
            total_matches = match_summary["match_count"].sum()
            avg_home = match_summary["avg_home_goals"].mean()
            avg_away = match_summary["avg_away_goals"].mean()
            
            lines.extend([
                "### Overall Statistics",
                "",
                f"- **Total Matches:** {total_matches}",
                f"- **Average Home Goals:** {avg_home:.2f}",
                f"- **Average Away Goals:** {avg_away:.2f}",
                f"- **Average Total Goals:** {avg_home + avg_away:.2f}",
                "",
            ])
        except Exception as e:
            lines.append(f"_Data summary unavailable: {e}_")
            lines.append("")
        
        lines.extend(["---", ""])
        
        return lines
    
    def _feature_summary_section(self) -> list:
        """Generate feature summary section."""
        lines = [
            "## Features",
            "",
        ]
        
        # Try to get feature count
        try:
            features_df = self.query_layer.query("SELECT * FROM features LIMIT 1")
            feature_count = len([c for c in features_df.columns if c not in ["fixture_id", "match_datetime_utc", "season", "league_id", "home_team_id", "away_team_id", "home_goals", "away_goals"]])
            
            lines.append(f"**Total Features Engineered:** {feature_count}")
            lines.append("")
            
            # List feature groups
            lines.append("**Feature Groups:**")
            for group in self.config.features.enabled_groups:
                lines.append(f"- {group.value}")
            lines.append("")
            
        except Exception as e:
            lines.append(f"_Feature summary unavailable: {e}_")
            lines.append("")
        
        lines.extend(["---", ""])
        
        return lines
    
    def _model_performance_section(self, metrics: Dict[str, Dict[str, float]]) -> list:
        """Generate model performance section."""
        lines = [
            "## Model Performance",
            "",
        ]
        
        for model_name, model_metrics in metrics.items():
            lines.extend([
                f"### {model_name}",
                "",
                "#### Home Goals",
                f"- MAE: {model_metrics.get('mae_home', 0):.3f}",
                f"- RMSE: {model_metrics.get('rmse_home', 0):.3f}",
                f"- Log-Likelihood: {model_metrics.get('ll_home', 0):.2f}",
                f"- Accuracy (±1): {model_metrics.get('acc_1_home', 0):.2%}",
                "",
                "#### Away Goals",
                f"- MAE: {model_metrics.get('mae_away', 0):.3f}",
                f"- RMSE: {model_metrics.get('rmse_away', 0):.3f}",
                f"- Log-Likelihood: {model_metrics.get('ll_away', 0):.2f}",
                f"- Accuracy (±1): {model_metrics.get('acc_1_away', 0):.2%}",
                "",
                "#### Total Goals",
                f"- MAE: {model_metrics.get('mae_total', 0):.3f}",
                "",
                "#### Probabilistic Metrics",
                f"- Brier Score: {model_metrics.get('brier_score', 0):.4f}",
                "",
            ])
        
        lines.extend(["---", ""])
        
        return lines
    
    def _prediction_analysis_section(self) -> list:
        """Generate prediction analysis section."""
        lines = [
            "## Prediction Analysis",
            "",
        ]
        
        try:
            perf = self.query_layer.get_prediction_performance()
            
            lines.append("### Model Comparison")
            lines.append("")
            lines.append(perf.to_markdown(index=False))
            lines.append("")
            
            # Over/Under analysis
            ou_perf = self.query_layer.get_over_under_performance(threshold=2.5)
            
            lines.append("### Over/Under 2.5 Performance")
            lines.append("")
            lines.append(ou_perf.to_markdown(index=False))
            lines.append("")
            
        except Exception as e:
            lines.append(f"_Prediction analysis unavailable: {e}_")
            lines.append("")
        
        lines.extend(["---", ""])
        
        return lines
    
    def save_report(self, run_id: str, metrics: Optional[Dict[str, Dict[str, float]]] = None):
        """Generate and save report to file."""
        report = self.generate_report(run_id, metrics)
        
        # Create run directory
        run_dir = self.config.paths.runs / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save report
        report_path = run_dir / "report.md"
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"✓ Report saved to {report_path}")
        
        return report_path
