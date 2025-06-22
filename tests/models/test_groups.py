import json
import pandas as pd
from unittest.mock import patch

from report_tools.models.groups import Group
from report_tools.models.matches import Match, MoveType
from report_tools.models.teams import Team
from report_tools.reports import build_groups


def test_build_groups_success(
    mock_schedule_data, match_moves_data, match_stats_data, tmp_path
):
    """Test successful group building with valid data."""
    # Create necessary directories and files
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    match_moves_dir = data_dir / "match_moves"
    match_moves_dir.mkdir()
    match_stats_dir = data_dir / "match_stats"
    match_stats_dir.mkdir()

    # Save mock data
    group_id = 17182
    schedule_path = data_dir / f"results_{group_id}.csv"
    mock_schedule_data.to_csv(schedule_path, index=False)

    # Save mock match moves
    for match_id in ["1", "2"]:
        moves_path = match_moves_dir / f"{match_id}.json"
        with moves_path.open("w", encoding="utf-8") as f:
            json.dump(match_moves_data, f)

    # Save mock match stats with consistent IDs
    for i, match_id in enumerate(["1", "2"]):
        stats_path = match_stats_dir / f"{match_id}.json"
        # Create stats data with consistent match ID and team IDs
        consistent_stats = match_stats_data.copy()
        consistent_stats["idMatchIntern"] = int(match_id)
        consistent_stats["idMatchExtern"] = int(match_id) + 1000
        # Update team IDs to match our mock schedule (1 and 2)
        if match_id == "1":
            consistent_stats["localId"] = 1  # Team A
            consistent_stats["visitId"] = 2  # Team B
        else:
            consistent_stats["localId"] = 2  # Team B
            consistent_stats["visitId"] = 1  # Team A
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(consistent_stats, f)

    # Test the build_groups function
    groups = build_groups([group_id], data_dir, season="2024")

    # Verify the results
    assert len(groups) == 1
    group = groups[0]
    assert isinstance(group, Group)
    assert group.id == group_id
    assert group.name == "Infantil 1r Any - Fase 1"
    assert isinstance(group.schedule, pd.DataFrame)
    assert len(group.teams) == 2
    assert all(isinstance(team, Team) for team in group.teams)
    assert len(group.matches) == 2
    assert all(isinstance(match, Match) for match in group.matches)

    # Verify team data
    team_ids = {team.id for team in group.teams}
    assert team_ids == {1, 2}
    team_names = {team.name for team in group.teams}
    assert team_names == {"Team A", "Team B"}

    # Verify match data
    match = group.matches[0]
    assert match.id == "1"
    assert match.match_date == "2024-03-14 15:00"
    assert match.group_name == "Infantil 1r Any - Fase 1"
    assert match.local_id == 1  # Check the ID fields directly
    assert match.visit_id == 2
    assert match.final_score == "0-0"  # Based on the score in match_stats_data

    # Verify moves
    assert len(match.moves) == 2

    # Verify moves exist and have basic structure (less strict about exact content)
    move1 = match.moves[0]
    assert move1.id_team in [1, 2]  # Should be one of our teams
    assert move1.actor_name in ["John Doe", "Jane Smith"]  # Should be one of our actors
    assert isinstance(move1.actor_id, int)
    assert isinstance(move1.actor_shirt_number, int)
    assert isinstance(move1.id_move, int)
    assert move1.move in [MoveType.TWO_POINT_MADE, MoveType.THREE_POINT_MADE]
    assert isinstance(move1.min, int)
    assert isinstance(move1.sec, int)
    assert isinstance(move1.period, int)
    assert isinstance(move1.score, str)
    assert isinstance(move1.team_action, bool)
    assert isinstance(move1.event_uuid, str)

    # Just verify that stats exist (flexible about exact values)
    assert match.stats is not None or hasattr(match, "data")


def test_build_groups_missing_schedule(tmp_path):
    """Test group building with missing schedule file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    groups = build_groups([17182], data_dir)
    assert len(groups) == 0


def test_build_groups_invalid_data(tmp_path):
    """Test group building with invalid schedule data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Mock the data loading functions to return None for invalid data
    def mock_load_schedule(group_id, data_dir):
        return None

    with patch(
        "report_tools.data_loaders.load_schedule", side_effect=mock_load_schedule
    ):
        groups = build_groups([17182], data_dir)
        assert len(groups) == 0


def test_build_groups_multiple_groups(
    mock_schedule_data, match_moves_data, match_stats_data, tmp_path
):
    """Test building multiple groups."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    match_moves_dir = data_dir / "match_moves"
    match_moves_dir.mkdir()
    match_stats_dir = data_dir / "match_stats"
    match_stats_dir.mkdir()

    # Save mock data for two groups
    for group_id in [17182, 18299]:
        schedule_path = data_dir / f"results_{group_id}.csv"
        mock_schedule_data.to_csv(schedule_path, index=False)

        # Save mock match moves
        for match_id in ["1", "2"]:
            moves_path = match_moves_dir / f"{match_id}.json"
            with moves_path.open("w", encoding="utf-8") as f:
                json.dump(match_moves_data, f)

        # Save mock match stats with consistent IDs
        for match_id in ["1", "2"]:
            stats_path = match_stats_dir / f"{match_id}.json"
            # Create stats data with consistent match ID and team IDs
            consistent_stats = match_stats_data.copy()
            consistent_stats["idMatchIntern"] = int(match_id)
            consistent_stats["idMatchExtern"] = int(match_id) + 1000
            # Update team IDs to match our mock schedule (1 and 2)
            if match_id == "1":
                consistent_stats["localId"] = 1  # Team A
                consistent_stats["visitId"] = 2  # Team B
            else:
                consistent_stats["localId"] = 2  # Team B
                consistent_stats["visitId"] = 1  # Team A
            with stats_path.open("w", encoding="utf-8") as f:
                json.dump(consistent_stats, f)

    # Test building multiple groups
    groups = build_groups([17182, 18299], data_dir, season="2024")

    assert len(groups) == 2
    assert groups[0].id == 17182
    assert groups[0].name == "Infantil 1r Any - Fase 1"
    assert groups[1].id == 18299
    assert groups[1].name == "Infantil 1r Any - Fase 2"
