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

    # Save mock match stats
    for match_id in ["1", "2"]:
        stats_path = match_stats_dir / f"{match_id}.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(match_stats_data, f)

    # Test the build_groups function
    groups = build_groups([group_id], data_dir)

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
    assert match.local.id == 1
    assert match.visitor.id == 2
    assert match.score == "80-75"

    # Verify moves
    assert len(match.moves) == 2

    # Verify first move
    move1 = match.moves[0]
    assert move1.id_team == 1
    assert move1.actor_name == "John Doe"
    assert move1.actor_id == 123
    assert move1.actor_shirt_number == "10"
    assert move1.id_move == 1
    assert move1.move == MoveType.TWO_POINT_MADE
    assert move1.min == 5
    assert move1.sec == 30
    assert move1.period == 1
    assert move1.score == "2-0"
    assert move1.team_action is True
    assert move1.event_uuid == "abc123"
    assert move1.foul_number is None
    assert move1.license_id is None

    # Verify second move
    move2 = match.moves[1]
    assert move2.id_team == 2
    assert move2.actor_name == "Jane Smith"
    assert move2.actor_id == 456
    assert move2.actor_shirt_number == "15"
    assert move2.id_move == 2
    assert move2.move == MoveType.THREE_POINT_MADE
    assert move2.min == 6
    assert move2.sec == 15
    assert move2.period == 1
    assert move2.score == "2-3"
    assert move2.team_action is True
    assert move2.event_uuid == "def456"
    assert move2.foul_number == 1
    assert move2.license_id == 789

    assert match.stats is not None


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

        # Save mock match stats
        for match_id in ["1", "2"]:
            stats_path = match_stats_dir / f"{match_id}.json"
            with stats_path.open("w", encoding="utf-8") as f:
                json.dump(match_stats_data, f)

    # Test building multiple groups
    groups = build_groups([17182, 18299], data_dir)

    assert len(groups) == 2
    assert groups[0].id == 17182
    assert groups[0].name == "Infantil 1r Any - Fase 1"
    assert groups[1].id == 18299
    assert groups[1].name == "Infantil 1r Any - Fase 2"
