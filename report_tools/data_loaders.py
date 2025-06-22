from datetime import datetime
import json
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from report_tools.logger import logger
from report_tools.models.matches import MatchStats, MatchMove, Match
from report_tools.models.players import PlayerStats, Player
from report_tools.models.teams import TeamStats, Team


def load_schedule(group_id, data_dir: Path) -> pd.DataFrame:
    """Read schedule CSV → DataFrame ensuring 'Match ID' exists."""
    csv_path = data_dir / f"results_{group_id}.csv"

    if not csv_path.exists():
        logger.warning(f"Schedule file {csv_path} not found, skipping group.")
        return pd.DataFrame()

    if not data_dir.exists():
        raise FileNotFoundError(f"Schedule file not found: {data_dir}")
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Check for id column, accept either 'id' or 'match_id'
    if "id" not in df.columns:
        if "match_id" in df.columns:
            # Rename match_id to id for compatibility
            df = df.rename(columns={"match_id": "id"})
        else:
            logger.error(
                "Could not find 'id' or 'match_id' column. Columns found: %s",
                df.columns.tolist(),
            )
            raise ValueError("CSV missing required column 'id' or 'match_id'")
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


def load_team_with_players(team_id: int, season: str, data_dir: Path) -> Optional[Team]:
    """Loads team data including players from team stats JSON.

    Args:
        team_id: The team identifier
        season: The season identifier
        data_dir: Directory containing the team stats JSON files

    Returns:
        Team instance with stats and players, None otherwise
    """
    file_path = data_dir / f"team_{team_id}_season_{season}.json"
    if not file_path.exists():
        logger.warning("Team stats file not found: %s", file_path)
        return None

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "team" not in data:
            logger.warning("Invalid team stats format for %s", team_id)
            return None

        team_data = data["team"]
        players_data = data.get("players", [])

        # Create team stats
        team_stats = TeamStats.model_validate(team_data)

        # Create players from player data
        players = []
        for player_data in players_data:
            # Create basic player info from stats
            player = Player(
                id=team_id,  # Use team_id as player container id
                name=player_data.get("name", ""),
                uuid=player_data.get("uuid"),
                club=player_data.get("club"),
                club_id=player_data.get("clubId"),
                team_name=player_data.get("teamName"),
                team_id=player_data.get("teamId"),
                category=player_data.get("category"),
                dorsal=player_data.get("dorsal"),
            )
            players.append(player)

        # Create team with stats and players
        team = Team(
            id=team_stats.team_id,
            name=team_stats.team_name,
            short_name=team_stats.team_name[:3].upper(),
            stats=team_stats,
            players=players,
        )

        return team

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
        return MatchStats.model_validate(data)
    except json.JSONDecodeError:
        logger.warning("Could not decode Match Stats JSON: %s", file_path)
        return None
    except Exception as e:
        logger.error("Error loading match stats for %s: %s", match_id, e)
        return None


def load_match(match_id: str, data_dir: Path) -> Optional[Match]:
    """Loads match data from match_stats JSON.

    Args:
        match_id: The match identifier
        data_dir: Directory containing the match stats JSON files

    Returns:
        Match instance with data, None otherwise
    """
    file_path = data_dir / "match_stats" / f"{match_id}.json"
    if not file_path.exists():
        logger.warning("Match stats JSON file not found: %s", file_path)
        return None

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Load moves separately
        moves = load_match_moves(match_id, data_dir)
        data["moves"] = [move.model_dump() for move in moves]

        return Match.model_validate(data)

    except json.JSONDecodeError:
        logger.warning("Could not decode Match JSON: %s", file_path)
        return None
    except Exception as e:
        logger.error("Error loading match for %s: %s", match_id, e)
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


def load_player(player_id: str, data_dir: Path) -> Optional[Player]:
    """Loads player data from player stats JSON.

    The player stats JSON has format: {"player_id": {"generalStats": ..., "evolutiveStats": [...]}}

    Args:
        player_id: The player identifier (key in the JSON)
        data_dir: Directory containing the player stats JSON files

    Returns:
        Player instance with stats, None otherwise
    """
    # Try to find the player stats file - it might be named differently
    # Look for files containing the player_id
    player_files = list(data_dir.glob(f"*{player_id}*.json"))
    if not player_files:
        logger.warning("No player stats files found for player %s", player_id)
        return None

    file_path = player_files[0]  # Take the first match

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Data format: {"player_id": {"generalStats": ..., "evolutiveStats": [...]}}
        if player_id not in data:
            logger.warning("Player %s not found in stats file %s", player_id, file_path)
            return None

        player_data = data[player_id]

        # Create player stats
        player_stats = PlayerStats.model_validate(player_data)

        # Extract player info from general stats
        general_stats = player_stats.general_stats

        player = Player(
            id=int(player_id),
            name=general_stats.name,
            uuid=getattr(general_stats, "uuid", None),
            club=general_stats.club,
            club_id=general_stats.club_id,
            team_name=general_stats.team_name,
            team_id=general_stats.team_id,
            category=general_stats.category,
            dorsal=general_stats.dorsal,
            stats=player_stats,
        )

        return player

    except json.JSONDecodeError as exc:
        logger.warning("Could not decode player stats JSON for %s: %s", player_id, exc)
        return None
    except Exception as e:
        logger.error("Error reading player stats file %s: %s", file_path, e)
        return None
