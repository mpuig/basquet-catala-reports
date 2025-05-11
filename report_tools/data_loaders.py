import json
from pathlib import Path
from typing import List

import pandas as pd

from report_tools.logger import logger


def load_schedule(path: Path) -> pd.DataFrame:
    """Read schedule CSV → DataFrame ensuring 'Match ID' exists."""
    if not path.exists():
        raise FileNotFoundError(f"Schedule file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    if "match_id" not in df.columns:
        logger.error(
            "Could not find 'match_id' column. Columns found: %s", df.columns.tolist()
        )
        raise ValueError("CSV missing required column 'match_id' or equivalent")
    return df


def load_match_moves(match_id: str, moves_dir: Path) -> List[dict]:
    """Return list of event dicts for a match or empty list if not found/invalid."""
    file_path = moves_dir / f"{match_id}.json"
    if not file_path.exists():
        logger.warning("Match moves JSON not found for %s", match_id)
        return []
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logger.warning("Could not decode JSON for %s – %s", match_id, exc)
        return []


def load_team_stats(team_id: str, season: str, team_stats_dir: Path) -> dict | None:
    """Loads team statistics JSON for a given team and season."""
    file_path = team_stats_dir / f"team_{team_id}_season_{season}.json"
    if not file_path.exists():
        logger.warning("Team stats file not found: %s", file_path)
        return None
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Could not decode team stats JSON for %s/%s: %s", team_id, season, exc
        )
        return None
    except Exception as e:
        logger.error("Error reading team stats file %s: %s", file_path, e)
        return None


def load_match_stats(match_id: str, stats_dir: Path) -> dict | None:
    """Loads and parses the aggregated match stats JSON data."""
    if not match_id or not isinstance(match_id, str) or len(match_id) < 5:
        logger.debug("Skipping load_match_stats for invalid match_id: %s", match_id)
        return None

    json_filepath = stats_dir / f"{match_id}.json"
    if not json_filepath.exists():
        logger.warning("Aggregated Match Stats JSON file not found: %s", json_filepath)
        return None

    try:
        with json_filepath.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning("Could not decode Match Stats JSON: %s", json_filepath)
        return None
    except Exception as e:
        logger.error("Error reading Match Stats JSON file %s: %s", json_filepath, e)
        return None
