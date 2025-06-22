"""Test fixtures for basketball reports."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import pytest

from report_tools.models.matches import Match, MatchMove, MoveType, MatchStats
from report_tools.models.teams import Team


def _create_temp_file(
    tmp_path: Path, subdir: str, filename: str, data: Any
) -> Tuple[Path, str]:
    """Helper function to create temporary test files.

    Args:
        tmp_path: Base temporary directory
        subdir: Subdirectory name
        filename: Name of the file to create
        data: Data to write to the file

    Returns:
        Tuple of (base_path, file_id)
    """
    dir_path = tmp_path / subdir
    dir_path.mkdir(exist_ok=True)
    file_path = dir_path / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return tmp_path, filename.split(".")[0]


@pytest.fixture
def sample_team():
    """Fixture providing a sample team."""
    return Team(id=1, name="Team A", short_name="TA")


@pytest.fixture
def sample_match(sample_team):
    """Fixture providing a sample match."""
    return Match(
        id="123",
        match_date="2024-03-14",
        group_name="Group 1",
        local=sample_team,
        visitor=Team(id=2, name="Team B", short_name="TB"),
        score="80-75",
        moves=[],
    )


@pytest.fixture
def sample_move():
    """Fixture providing a sample move event."""
    return MatchMove(
        id_team=1,
        actor_name="John Doe",
        actor_id=101,
        actor_shirt_number="10",
        id_move=1,
        move=MoveType.TWO_POINT_MADE,
        min=5,
        sec=30,
        period=1,
        score="2-0",
        team_action=True,
        event_uuid="abc123",
        timestamp=datetime.now(),
    )


@pytest.fixture
def match_stats_data() -> Dict[str, Dict[str, int]]:
    """Fixture providing sample match stats data."""
    return {
        "idMatchIntern": 158215,
        "idMatchExtern": 928403,
        "localId": 319130,
        "visitId": 319131,
        "period": 4,
        "periodDuration": 10,
        "sumPeriod": 30,
        "periodDurationList": [10, 10, 10, 10],
        "shotsOfTwoSuccessful": 15,
        "shotsOfTwoAttempted": 30,
        "shotsOfThreeSuccessful": 8,
        "shotsOfThreeAttempted": 20,
        "shotsOfOneSuccessful": 12,
        "shotsOfOneAttempted": 15,
        "faults": 18,
        "moves": [],
        "score": [
            {
                "local": 0,
                "visit": 0,
                "minuteQuarter": 0,
                "minuteAbsolute": 0,
                "period": 1,
            }
        ],
    }


@pytest.fixture
def match_moves_data() -> List[Dict[str, Any]]:
    """Fixture providing sample match moves data."""
    return [
        {
            "idTeam": 1,
            "actorName": "John Doe",
            "actorId": 123,
            "actorShirtNumber": "10",
            "idMove": 1,
            "move": "Cistella de 2",
            "min": 5,
            "sec": 30,
            "period": 1,
            "score": "2-0",
            "teamAction": True,
            "eventUuid": "abc123",
            "timestamp": "20240314150000",
            "foulNumber": None,
            "licenseId": None,
        },
        {
            "idTeam": 2,
            "actorName": "Jane Smith",
            "actorId": 456,
            "actorShirtNumber": "15",
            "idMove": 2,
            "move": "Cistella de 3",
            "min": 6,
            "sec": 15,
            "period": 1,
            "score": "2-3",
            "teamAction": True,
            "eventUuid": "def456",
            "timestamp": "20240314150100",
            "foulNumber": 1,
            "licenseId": 789,
        },
    ]


@pytest.fixture
def match_moves_file(
    tmp_path: Path, match_moves_data: List[Dict[str, Any]]
) -> Tuple[Path, str]:
    """Create a temporary match moves JSON file for testing."""
    return _create_temp_file(
        tmp_path, "match_moves", "test_match_123.json", match_moves_data
    )


@pytest.fixture
def fixture_match_moves_file(tmp_path: Path) -> Tuple[Path, str]:
    """Create a temporary match moves file from the fixture data."""
    match_id = "158215"  # Using the idMatchIntern from the fixture

    # Read the fixture file
    fixture_path = Path(__file__).parent / "fixtures" / "match_moves.json"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    with open(fixture_path, "r", encoding="utf-8") as f:
        moves_data = json.load(f)

    return _create_temp_file(tmp_path, "match_moves", f"{match_id}.json", moves_data)


@pytest.fixture
def team_stats_fixture_file(tmp_path: Path) -> Tuple[Path, str, str]:
    """Create a temporary team stats file from the fixture data."""
    team_id = "68454"  # From the fixture file
    season = "2024"  # Example season

    # Read the fixture file
    fixture_path = Path(__file__).parent / "fixtures" / "team_stats.json"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    with open(fixture_path, "r", encoding="utf-8") as f:
        fixture_data = json.load(f)

    base_path, _ = _create_temp_file(
        tmp_path, "team_stats", f"team_{team_id}_season_{season}.json", fixture_data
    )

    return base_path, team_id, season


@pytest.fixture
def player_stats_fixture_file(tmp_path: Path) -> Tuple[Path, str, str]:
    """Create a temporary player stats file from the fixture data."""
    player_id = "69884"  # From the fixture file
    team_id = "266"  # Example team ID

    # Read the fixture file
    fixture_path = Path(__file__).parent / "fixtures" / "player_stats.json"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    with open(fixture_path, "r", encoding="utf-8") as f:
        fixture_data = json.load(f)

    # Map the keys from the fixture file to the keys expected by PlayerStats
    player_data = fixture_data[player_id]["generalStats"]
    mapped_data = {
        "shotsOfTwoSuccessful": player_data.get("sumShotsOfTwoSuccessful"),
        "shotsOfTwoAttempted": player_data.get("sumShotsOfTwoAttempted"),
        "shotsOfThreeSuccessful": player_data.get("sumShotsOfThreeSuccessful"),
        "shotsOfThreeAttempted": player_data.get("sumShotsOfThreeAttempted"),
        "shotsOfOneSuccessful": player_data.get("sumShotsOfOneSuccessful"),
        "shotsOfOneAttempted": player_data.get("sumShotsOfOneAttempted"),
        "faults": player_data.get("sumFouls"),
    }

    base_path, _ = _create_temp_file(
        tmp_path, "player_stats", f"player_{player_id}_team_{team_id}.json", mapped_data
    )

    return base_path, player_id, team_id


@pytest.fixture
def mock_schedule_data() -> pd.DataFrame:
    """Fixture providing sample schedule data."""
    return pd.DataFrame(
        {
            "id": ["1", "2"],
            "date_time": ["2024-03-14 15:00", "2024-03-14 16:00"],
            "local_team": ["Team A", "Team B"],
            "local_team_id": [1, 2],
            "visitor_team": ["Team B", "Team A"],
            "visitor_team_id": [2, 1],
            "score": ["80-75", "70-65"],
        }
    )


@pytest.fixture
def dummy_match_stats_file(tmp_path, match_stats_data):
    """Create a temporary match stats JSON file for testing."""
    match_id = "test_match_123"
    stats_dir = tmp_path / "match_stats"
    stats_dir.mkdir()
    file_path = stats_dir / f"{match_id}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(match_stats_data, f)

    return tmp_path, match_id


@pytest.fixture
def match_stats_fixture_file(
    tmp_path: Path, match_id: Optional[str] = None
) -> Tuple[Path, str]:
    """Create a temporary match stats file from the fixture data.

    Args:
        tmp_path: Base temporary directory
        match_id: Optional match ID to use. If not provided, uses the fixture's match ID.

    Returns:
        Tuple of (base_path, match_id)
    """
    match_id = match_id or "158215"  # Using the idMatchIntern from the fixture

    # Read the fixture file
    fixture_path = Path(__file__).parent / "fixtures" / "match_stats.json"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    with open(fixture_path, "r", encoding="utf-8") as f:
        stats_data = json.load(f)

    return _create_temp_file(tmp_path, "match_stats", f"{match_id}.json", stats_data)


@pytest.fixture
def match_moves_fixture_file(
    tmp_path: Path, match_id: Optional[str] = None
) -> Tuple[Path, str]:
    """Create a temporary match moves file from the fixture data.

    Args:
        tmp_path: Base temporary directory
        match_id: Optional match ID to use. If not provided, uses the fixture's match ID.

    Returns:
        Tuple of (base_path, match_id)
    """
    match_id = match_id or "158215"  # Using the idMatchIntern from the fixture

    # Read the fixture file
    fixture_path = Path(__file__).parent / "fixtures" / "match_moves.json"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    with open(fixture_path, "r", encoding="utf-8") as f:
        moves_data = json.load(f)

    return _create_temp_file(tmp_path, "match_moves", f"{match_id}.json", moves_data)


@pytest.fixture
def advanced_stats_match_stats(
    match_stats_fixture_file: Tuple[Path, str],
) -> MatchStats:
    """Load match stats from fixture file for advanced stats testing."""
    base_path, match_id = match_stats_fixture_file
    file_path = base_path / "match_stats" / f"{match_id}.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return MatchStats(**data)


@pytest.fixture
def advanced_stats_match_moves(
    match_moves_fixture_file: Tuple[Path, str],
) -> List[MatchMove]:
    """Load match moves from fixture file for advanced stats testing."""
    base_path, match_id = match_moves_fixture_file
    file_path = base_path / "match_moves" / f"{match_id}.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [MatchMove(**move) for move in data]
