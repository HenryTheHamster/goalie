"""Feature dictionary for interpretability and tracking."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class FeatureMetadata:
    """Metadata for a single feature."""
    
    name: str
    description: str
    feature_group: str
    data_type: str  # "float", "int", "bool", etc.
    nullable: bool = False
    created_at: Optional[str] = None
    
    def __post_init__(self):
        """Set creation timestamp."""
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class FeatureDictionary:
    """Registry of all features with metadata for interpretability."""
    
    def __init__(self):
        """Initialize empty feature dictionary."""
        self.features: Dict[str, FeatureMetadata] = {}
    
    def register(
        self,
        name: str,
        description: str,
        feature_group: str,
        data_type: str = "float",
        nullable: bool = False,
    ):
        """Register a new feature."""
        if name in self.features:
            raise ValueError(f"Feature '{name}' already registered")
        
        self.features[name] = FeatureMetadata(
            name=name,
            description=description,
            feature_group=feature_group,
            data_type=data_type,
            nullable=nullable,
        )
    
    def get(self, name: str) -> Optional[FeatureMetadata]:
        """Get metadata for a feature."""
        return self.features.get(name)
    
    def list_by_group(self, feature_group: str) -> List[FeatureMetadata]:
        """List all features in a specific group."""
        return [f for f in self.features.values() if f.feature_group == feature_group]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert dictionary to DataFrame for export."""
        records = []
        for feature in self.features.values():
            records.append({
                "feature_name": feature.name,
                "description": feature.description,
                "feature_group": feature.feature_group,
                "data_type": feature.data_type,
                "nullable": feature.nullable,
                "created_at": feature.created_at,
            })
        return pd.DataFrame(records)
    
    def save(self, path: Path):
        """Save feature dictionary to Parquet."""
        df = self.to_dataframe()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"Saved feature dictionary ({len(df)} features) to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "FeatureDictionary":
        """Load feature dictionary from Parquet."""
        df = pd.read_parquet(path)
        dictionary = cls()
        
        for _, row in df.iterrows():
            dictionary.register(
                name=row["feature_name"],
                description=row["description"],
                feature_group=row["feature_group"],
                data_type=row["data_type"],
                nullable=row["nullable"],
            )
        
        print(f"Loaded feature dictionary ({len(dictionary.features)} features) from {path}")
        return dictionary
