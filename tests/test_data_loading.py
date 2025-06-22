"""Tests for data loading functions."""

import json
import pytest
from pathlib import Path

from report_tools.data_loaders import (
    load_match,
    load_team_with_players,
    load_player,
)
from report_tools.models.matches import Match
from report_tools.models.teams import Team
from report_tools.models.players import Player


def test_load_match_success(match_stats_fixture_file):
    """Test successful loading of match data."""
    data_dir, match_id = match_stats_fixture_file

    # Create moves file if it doesn't exist
    moves_dir = data_dir / "match_moves"
    moves_dir.mkdir(exist_ok=True)
    moves_file = moves_dir / f"{match_id}.json"
    if not moves_file.exists():
        with moves_file.open("w") as f:
            import json

            json.dump([], f)  # Empty moves array

    match = load_match(match_id, data_dir)
    assert match is not None
    assert isinstance(match, Match)

    # Test basic match info
    assert match.id_match_intern == 158215
    assert match.id_match_extern == 928403
    assert match.local_id == 319130
    assert match.visit_id == 319131

    # Test data
    assert len(match.teams) == 2
    assert len(match.score) > 0

    # Test that moves are loaded (may be empty in fixture)
    assert isinstance(match.moves, list)

    # Test legacy compatibility
    assert match.id == str(match.id_match_intern)
    assert match.final_score == "54-102"


def test_load_match_missing_file(tmp_path):
    """Test handling of missing match file."""
    match = load_match("nonexistent", tmp_path)
    assert match is None


def test_load_match_invalid_json(tmp_path):
    """Test handling of invalid JSON in match file."""
    match_stats_dir = tmp_path / "match_stats"
    match_stats_dir.mkdir()

    file_path = match_stats_dir / "invalid.json"
    with file_path.open("w") as f:
        f.write("invalid json")

    match = load_match("invalid", tmp_path)
    assert match is None


def test_load_team_with_players_success(team_stats_fixture_file):
    """Test successful loading of team with players."""
    data_dir, team_id, season = team_stats_fixture_file

    team = load_team_with_players(int(team_id), season, data_dir / "team_stats")
    assert team is not None
    assert isinstance(team, Team)

    # Test basic team info
    assert team.id == 68454
    assert team.name == "CB TORELLÓ IFN - CUIDA'T ESTÈTICA"
    assert team.short_name == "CB "

    # Test team stats
    assert team.stats is not None
    assert team.stats.sum_matches == 23
    assert team.stats.team_results.wins == 16
    assert team.stats.team_results.losses == 7

    # Test players
    assert team.players is not None
    assert len(team.players) > 0

    # Test first player
    player1 = team.players[0]
    assert player1.name == "MARTA RIVERA SANTOS"
    assert player1.team_id == 68454
    assert player1.dorsal == 28


def test_load_team_with_players_missing_file(tmp_path):
    """Test handling of missing team stats file."""
    team = load_team_with_players(99999, "2024", tmp_path)
    assert team is None


def test_load_team_with_players_invalid_format(tmp_path):
    """Test handling of invalid team stats format."""
    file_path = tmp_path / "team_99999_season_2024.json"

    # Write invalid format (missing 'team' key)
    with file_path.open("w") as f:
        json.dump({"invalid": "format"}, f)

    team = load_team_with_players(99999, "2024", tmp_path)
    assert team is None


def test_load_player_with_fixture(player_stats_fixture_file):
    """Test player loading with fixture data."""
    data_dir, player_id, team_id = player_stats_fixture_file

    # Create a mock player stats file in the expected format
    stats_file = data_dir / f"player_{player_id}.json"

    # Read the fixture data and restructure it
    fixture_file = Path(__file__).parent / "fixtures" / "player_stats.json"
    with open(fixture_file, "r") as f:
        fixture_data = json.load(f)

    # Create the expected format for loading
    player_data = {player_id: fixture_data[player_id]}

    with stats_file.open("w") as f:
        json.dump(player_data, f)

    # Test loading
    player = load_player(player_id, data_dir)
    assert player is not None
    assert isinstance(player, Player)

    # Test basic info
    assert player.id == int(player_id)
    assert player.name == "CARLA MENDEZ TORRES"
    assert player.team_id == 69884
    assert player.dorsal == 15

    # Test stats
    assert player.stats is not None
    assert player.stats.general_stats.matches_played == 10
    assert len(player.stats.evolutive_stats) > 0  # Should have evolutive stats


def test_load_player_missing_file(tmp_path):
    """Test handling of missing player stats file."""
    player = load_player("99999", tmp_path)
    assert player is None


def test_load_player_player_not_found(tmp_path):
    """Test handling when player ID is not found in stats file."""
    stats_file = tmp_path / "player_12345.json"

    # Write data for different player
    with stats_file.open("w") as f:
        json.dump({"54321": {"generalStats": {}, "evolutiveStats": []}}, f)

    player = load_player("12345", tmp_path)
    assert player is None


@pytest.fixture
def mock_data():
    """Fixture providing mock data for testing."""
    return {
        "match_data": {
            "idMatchIntern": 12345,
            "idMatchExtern": 54321,
            "time": "2024-06-22 15:00:00",
            "localId": 100,
            "visitId": 101,
            "period": 4,
            "periodDuration": 10,
            "sumPeriod": 40,
            "periodDurationList": [10, 10, 10, 10],
            "lastMinuteUsed": 1,
            "moves": [],
            "recalculated": True,
            "score": [
                {
                    "local": 0,
                    "visit": 0,
                    "minuteQuarter": 0,
                    "minuteAbsolute": 0,
                    "period": 1,
                },
                {
                    "local": 85,
                    "visit": 80,
                    "minuteQuarter": 0,
                    "minuteAbsolute": 40,
                    "period": 4,
                },
            ],
            "teams": [
                {
                    "teamIdIntern": 100,
                    "teamIdExtern": 200,
                    "colorRgb": "#FF0000",
                    "name": "Test Team A",
                    "shortName": "TTA",
                    "fede": "test",
                    "players": [
                        {
                            "actorId": 1001,
                            "uuid": "player-uuid-1",
                            "playerIdsInterns": [1001],
                            "teamId": 200,
                            "name": "Test Player 1",
                            "dorsal": "10",
                            "starting": True,
                            "captain": True,
                            "sumPeriod": 40,
                            "period": 4,
                            "periodDuration": 10,
                            "timePlayed": 38,
                            "inOutsList": [
                                {"type": "IN_TYPE", "minuteAbsolut": 0, "pointDiff": 0}
                            ],
                            "gamePlayed": 1,
                            "inOut": 5,
                            "matchHasStartingPlayers": True,
                            "teamScore": 85,
                            "oppScore": 80,
                            "data": {
                                "type": 0,
                                "score": 20,
                                "valoration": 15,
                                "shotsOfOneAttempted": 4,
                                "shotsOfTwoAttempted": 8,
                                "shotsOfThreeAttempted": 2,
                                "shotsOfOneSuccessful": 3,
                                "shotsOfTwoSuccessful": 7,
                                "shotsOfThreeSuccessful": 1,
                            },
                            "periods": [],
                            "eventTime": {"minute": 0, "second": 0},
                        }
                    ],
                    "data": {"type": 2, "score": 85, "valoration": 40},
                    "periods": [],
                    "eventTime": {"minute": 0, "second": 0},
                }
            ],
            "data": {"type": 3, "score": 165, "valoration": 75},
            "periods": [],
            "eventTime": {"minute": 0, "second": 0},
        }
    }


def test_data_integration(tmp_path, mock_data):
    """Test integration of data loading with mock data."""
    # Setup directories
    match_stats_dir = tmp_path / "match_stats"
    match_stats_dir.mkdir()
    match_moves_dir = tmp_path / "match_moves"
    match_moves_dir.mkdir()

    match_id = "12345"

    # Write match data
    match_file = match_stats_dir / f"{match_id}.json"
    with match_file.open("w") as f:
        json.dump(mock_data["match_data"], f)

    # Write empty moves data (would normally be loaded separately)
    moves_file = match_moves_dir / f"{match_id}.json"
    with moves_file.open("w") as f:
        json.dump([], f)

    # Test loading
    match = load_match(match_id, tmp_path)
    assert match is not None

    # Test data structure
    assert match.id_match_intern == 12345
    assert len(match.teams) == 1
    assert len(match.teams[0].players) == 1

    team = match.teams[0]
    assert team.name == "Test Team A"
    assert team.short_name == "TTA"

    player = team.players[0]
    assert player.name == "Test Player 1"
    assert player.dorsal == "10"
    assert player.starting is True
    assert player.captain is True
    assert player.data.score == 20

    # Test score progression
    assert len(match.score) == 2
    assert match.score[-1].local == 85
    assert match.score[-1].visit == 80

    # Test legacy compatibility
    assert match.final_score == "85-80"
    assert match.local.name == "Test Team A"


def test_error_handling_in_loading(tmp_path):
    """Test error handling in loading functions."""

    # Test match loading with corrupted JSON
    match_stats_dir = tmp_path / "match_stats"
    match_stats_dir.mkdir()

    corrupted_file = match_stats_dir / "corrupted.json"
    with corrupted_file.open("w") as f:
        f.write('{"incomplete": json structure')

    match = load_match("corrupted", tmp_path)
    assert match is None

    # Test team loading with missing team key
    team_file = tmp_path / "team_123_season_2024.json"
    with team_file.open("w") as f:
        json.dump({"players": [], "no_team_key": True}, f)

    team = load_team_with_players(123, "2024", tmp_path)
    assert team is None

    # Test player loading with empty file
    empty_file = tmp_path / "player_456.json"
    with empty_file.open("w") as f:
        json.dump({}, f)

    player = load_player("456", tmp_path)
    assert player is None
