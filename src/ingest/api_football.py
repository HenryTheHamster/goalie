"""API-Football client for data ingestion."""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

from src.config import APIFootballConfig, LeagueConfig, MatchStatus


class APIFootballClient:
    """Client for API-Football v3 (DIRECT auth, not RapidAPI)."""
    
    def __init__(self, config: APIFootballConfig):
        """Initialize client with configuration."""
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "x-apisports-key": config.api_key,
            "Accept": "application/json",
        })
        self._last_request_time = 0.0
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make API request with rate limiting and error handling."""
        self._rate_limit()
        
        url = f"{self.config.base_url}/{endpoint}"
        
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()
            
            # Check API response structure
            if "errors" in data and data["errors"]:
                raise ValueError(f"API errors: {data['errors']}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
    
    def get_leagues(self) -> List[Dict[str, Any]]:
        """Fetch all available leagues."""
        data = self._request("leagues")
        return data.get("response", [])
    
    def get_teams(self, league_id: int, season: int) -> List[Dict[str, Any]]:
        """Fetch teams for a specific league and season."""
        data = self._request("teams", params={"league": league_id, "season": season})
        return data.get("response", [])
    
    def get_fixtures(
        self,
        league_id: int,
        season: int,
        status: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch fixtures for a specific league and season."""
        params: Dict[str, Any] = {
            "league": league_id,
            "season": season,
        }
        
        if status:
            params["status"] = status
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        data = self._request("fixtures", params=params)
        return data.get("response", [])
    
    def get_fixture_statistics(self, fixture_id: int) -> Dict[str, Any]:
        """Fetch statistics for a specific fixture."""
        data = self._request("fixtures/statistics", params={"fixture": fixture_id})
        return data.get("response", [])
    
    def ingest_leagues(self, output_path: Path) -> pd.DataFrame:
        """Ingest all leagues and save to Parquet."""
        print("Fetching leagues from API-Football...")
        leagues_data = self.get_leagues()
        
        # Flatten nested structure
        records = []
        for item in leagues_data:
            league = item.get("league", {})
            country = item.get("country", {})
            seasons = item.get("seasons", [])
            
            for season in seasons:
                records.append({
                    "league_id": league.get("id"),
                    "league_name": league.get("name"),
                    "league_type": league.get("type"),
                    "league_logo": league.get("logo"),
                    "country_name": country.get("name"),
                    "country_code": country.get("code"),
                    "country_flag": country.get("flag"),
                    "season": season.get("year"),
                    "season_start": season.get("start"),
                    "season_end": season.get("end"),
                    "season_current": season.get("current", False),
                    "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
                })
        
        df = pd.DataFrame(records)
        
        # Save to Parquet (append-only)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"leagues_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.parquet"
        filepath = output_path / filename
        df.to_parquet(filepath, index=False)
        
        print(f"Ingested {len(df)} league-season records to {filepath}")
        return df
    
    def ingest_teams(self, league_config: LeagueConfig, output_path: Path) -> pd.DataFrame:
        """Ingest teams for a specific league across seasons."""
        print(f"Fetching teams for {league_config.name}...")
        
        all_records = []
        for season in tqdm(league_config.seasons, desc="Seasons"):
            teams_data = self.get_teams(league_config.league_id, season)
            
            for item in teams_data:
                team = item.get("team", {})
                venue = item.get("venue", {})
                
                all_records.append({
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "team_code": team.get("code"),
                    "team_country": team.get("country"),
                    "team_founded": team.get("founded"),
                    "team_national": team.get("national", False),
                    "team_logo": team.get("logo"),
                    "venue_id": venue.get("id"),
                    "venue_name": venue.get("name"),
                    "venue_address": venue.get("address"),
                    "venue_city": venue.get("city"),
                    "venue_capacity": venue.get("capacity"),
                    "venue_surface": venue.get("surface"),
                    "venue_image": venue.get("image"),
                    "league_id": league_config.league_id,
                    "season": season,
                    "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
                })
        
        df = pd.DataFrame(all_records)
        
        # Save to Parquet (append-only)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"teams_league{league_config.league_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.parquet"
        filepath = output_path / filename
        df.to_parquet(filepath, index=False)
        
        print(f"Ingested {len(df)} team records to {filepath}")
        return df
    
    def ingest_fixtures(self, league_config: LeagueConfig, output_path: Path) -> pd.DataFrame:
        """Ingest fixtures for a specific league across seasons."""
        print(f"Fetching fixtures for {league_config.name}...")
        
        all_records = []
        for season in tqdm(league_config.seasons, desc="Seasons"):
            fixtures_data = self.get_fixtures(league_config.league_id, season)
            
            for item in fixtures_data:
                fixture = item.get("fixture", {})
                league = item.get("league", {})
                teams = item.get("teams", {})
                goals = item.get("goals", {})
                score = item.get("score", {})
                
                # Extract timestamp and ensure UTC
                timestamp_str = fixture.get("date")
                timestamp_utc = None
                if timestamp_str:
                    try:
                        # Parse ISO format and convert to UTC
                        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        timestamp_utc = dt.astimezone(timezone.utc).isoformat()
                    except Exception as e:
                        print(f"Warning: Failed to parse timestamp {timestamp_str}: {e}")
                
                all_records.append({
                    "fixture_id": fixture.get("id"),
                    "fixture_referee": fixture.get("referee"),
                    "fixture_timezone": fixture.get("timezone"),
                    "fixture_date": timestamp_str,
                    "fixture_timestamp": fixture.get("timestamp"),
                    "fixture_timestamp_utc": timestamp_utc,
                    "fixture_venue_id": fixture.get("venue", {}).get("id"),
                    "fixture_venue_name": fixture.get("venue", {}).get("name"),
                    "fixture_venue_city": fixture.get("venue", {}).get("city"),
                    "fixture_status_long": fixture.get("status", {}).get("long"),
                    "fixture_status_short": fixture.get("status", {}).get("short"),
                    "fixture_status_elapsed": fixture.get("status", {}).get("elapsed"),
                    "league_id": league.get("id"),
                    "league_name": league.get("name"),
                    "league_country": league.get("country"),
                    "league_logo": league.get("logo"),
                    "league_flag": league.get("flag"),
                    "league_season": league.get("season"),
                    "league_round": league.get("round"),
                    "teams_home_id": teams.get("home", {}).get("id"),
                    "teams_home_name": teams.get("home", {}).get("name"),
                    "teams_home_logo": teams.get("home", {}).get("logo"),
                    "teams_home_winner": teams.get("home", {}).get("winner"),
                    "teams_away_id": teams.get("away", {}).get("id"),
                    "teams_away_name": teams.get("away", {}).get("name"),
                    "teams_away_logo": teams.get("away", {}).get("logo"),
                    "teams_away_winner": teams.get("away", {}).get("winner"),
                    "goals_home": goals.get("home"),
                    "goals_away": goals.get("away"),
                    "score_halftime_home": score.get("halftime", {}).get("home"),
                    "score_halftime_away": score.get("halftime", {}).get("away"),
                    "score_fulltime_home": score.get("fulltime", {}).get("home"),
                    "score_fulltime_away": score.get("fulltime", {}).get("away"),
                    "score_extratime_home": score.get("extratime", {}).get("home"),
                    "score_extratime_away": score.get("extratime", {}).get("away"),
                    "score_penalty_home": score.get("penalty", {}).get("home"),
                    "score_penalty_away": score.get("penalty", {}).get("away"),
                    "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
                })
        
        df = pd.DataFrame(all_records)
        
        # Save to Parquet (append-only)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"fixtures_league{league_config.league_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.parquet"
        filepath = output_path / filename
        df.to_parquet(filepath, index=False)
        
        print(f"Ingested {len(df)} fixture records to {filepath}")
        return df
    
    def health_check(self) -> bool:
        """Check API health and authentication."""
        try:
            data = self._request("status")
            account = data.get("response", {})
            print(f"API Status: {account}")
            return True
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
