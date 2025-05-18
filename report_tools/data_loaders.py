from datetime import datetime
import json
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from report_tools.logger import logger
from report_tools.models.matches import MatchStats, MatchMove

from report_tools.models.players import PlayerStats
from report_tools.models.teams import TeamStats


def load_schedule(group_id, data_dir: Path) -> pd.DataFrame:
    """Read schedule CSV → DataFrame ensuring 'Match ID' exists."""
    csv_path = data_dir / f"results_{group_id}.csv"

    if not csv_path.exists():
        logger.warning(f"Schedule file {csv_path} not found, skipping group.")
        return pd.DataFrame()

    if not data_dir.exists():
        raise FileNotFoundError(f"Schedule file not found: {data_dir}")
    df = pd.read_csv(csv_path, encoding="utf-8")
    if "id" not in df.columns:
        logger.error(
            "Could not find 'id' column. Columns found: %s", df.columns.tolist()
        )
        raise ValueError("CSV missing required column 'id' or equivalent")
    return df


def camel_to_snake(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def parse_timestamp(ts_str):
    try:
        return datetime.strptime(ts_str, "%Y%m%d%H%M%S")
    except Exception:
        return None


def load_match_moves(match_id: str, data_dir: Path) -> List[MatchMove]:
    """Loads and parses the match moves JSON data.

    Args:
        match_id: The match identifier
        data_dir: Directory containing the match moves JSON files

    Returns:
        List of MatchMove instances if successful, empty list otherwise
    """
    file_path = data_dir / "match_moves" / f"{match_id}.json"
    print(f"Looking for match moves file at: {file_path}")
    if not file_path.exists():
        logger.warning("Match moves JSON not found for %s", match_id)
        return []

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Loaded JSON data with {len(data)} moves")

        moves = []
        for i, move_dict in enumerate(data):
            print(f"Processing move {i+1}/{len(data)}")
            # Parse timestamp if present
            if "timestamp" in move_dict and isinstance(move_dict["timestamp"], str):
                move_dict["timestamp"] = parse_timestamp(move_dict["timestamp"])
            try:
                move = MatchMove.model_validate(move_dict)
                moves.append(move)
            except Exception as e:
                print(f"Error validating move {i+1}: {move_dict}")
                print(f"Validation error: {e}")
                raise
        print(f"Successfully processed {len(moves)} moves")
        return moves

    except json.JSONDecodeError as exc:
        logger.warning("Could not decode JSON for %s – %s", match_id, exc)
        return []
    except Exception as exc:
        logger.error("Error loading match moves for %s: %s", match_id, exc)
        return []


def load_team_stats(team_id: str, season: str, data_dir: Path) -> Optional[TeamStats]:
    """Loads team statistics JSON for a given team and season and returns a TeamStats instance.

    Args:
        team_id: The team identifier
        season: The season identifier
        data_dir: Directory containing the team stats JSON files

    Returns:
        TeamStats instance if successful, None otherwise
    """
    file_path = data_dir / f"team_{team_id}_season_{season}.json"
    if not file_path.exists():
        logger.warning("Team stats file not found: %s", file_path)
        return None

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # The team stats are nested under the "team" key
        if isinstance(data, dict) and "team" in data:
            data = data["team"]

        return TeamStats.model_validate(data)

    except json.JSONDecodeError as exc:
        logger.warning(
            "Could not decode team stats JSON for %s/%s: %s", team_id, season, exc
        )
        return None
    except Exception as e:
        logger.error("Error reading team stats file %s: %s", file_path, e)
        return None


def load_match_stats(match_id: str, data_dir: Path) -> Optional[MatchStats]:
    """Loads and parses the aggregated match stats JSON data.

    Args:
        match_id: The match identifier
        data_dir: Directory containing the match stats JSON files

    Returns:
        MatchStats instance if successful, None otherwise
    """
    file_path = data_dir / "match_stats" / f"{match_id}.json"
    if not file_path.exists():
        logger.warning("Aggregated Match Stats JSON file not found: %s", file_path)
        return None

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate that we have a dictionary
        if not isinstance(data, dict):
            logger.warning("Match stats JSON is not a dictionary: %s", file_path)
            return None

        # Check for required fields
        required_fields = ["idMatchIntern", "idMatchExtern", "localId", "visitId", "period", "periodDuration", "sumPeriod", "periodDurationList"]
        if not all(field in data for field in required_fields):
            logger.warning("Match stats JSON missing required fields: %s", file_path)
            return None

        # Ensure moves is a list
        if "moves" not in data:
            data["moves"] = []

        # Ensure score is a list of score entries
        if "score" not in data:
            data["score"] = []
        elif not isinstance(data["score"], list):
            logger.warning("Score is not a list in match stats: %s", file_path)
            return None

        # Validate the data using the Pydantic model
        return MatchStats.model_validate(data)

    except json.JSONDecodeError:
        logger.warning("Could not decode Match Stats JSON: %s", file_path)
        return None
    except Exception as e:
        logger.error("Error loading match stats for %s: %s", match_id, e)
        return None


def load_player_stats(
    player_id: str, team_id: str, data_dir: Path
) -> Optional[PlayerStats]:
    """Loads player statistics JSON for a given player and team and returns a PlayerStats instance.

    Args:
        player_id: The player identifier
        team_id: The team identifier
        data_dir: Directory containing the player stats JSON files

    Returns:
        PlayerStats instance if successful, None otherwise
    """
    file_path = data_dir / f"player_{player_id}_team_{team_id}.json"
    if not file_path.exists():
        logger.warning("Player stats file not found: %s", file_path)
        return None

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # The player stats are nested under the "player" key
        if isinstance(data, dict) and "player" in data:
            data = data["player"]

        return PlayerStats.model_validate(data)

    except json.JSONDecodeError as exc:
        logger.warning(
            "Could not decode player stats JSON for %s/%s: %s", player_id, team_id, exc
        )
        return None
    except Exception as e:
        logger.error("Error reading player stats file %s: %s", file_path, e)
        return None
