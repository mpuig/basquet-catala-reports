from report_tools.data_loaders import load_player_stats
from report_tools.models.players import PlayerStats


def test_load_player_stats_success(player_stats_fixture_file):
    """Test successful loading of player stats from a valid JSON file."""
    data_dir, player_id, team_id = player_stats_fixture_file
    stats = load_player_stats(player_id, team_id, data_dir / "player_stats")
    assert stats is not None
    assert isinstance(stats, PlayerStats)
    assert stats.shots_of_two_successful == 1
    assert stats.shots_of_two_attempted == 1
    assert stats.shots_of_three_successful == 0
    assert stats.shots_of_three_attempted == 0
    assert stats.shots_of_one_successful == 0
    assert stats.shots_of_one_attempted == 0
    assert stats.faults == 11


def test_load_player_stats_missing_file(tmp_path):
    """Test handling of missing player stats file."""
    player_id = "99999"
    team_id = "99999"
    stats = load_player_stats(player_id, team_id, tmp_path)
    assert stats is None


def test_load_player_stats_invalid_json(tmp_path):
    """Test handling of invalid JSON in player stats file."""
    player_id = "69884"
    team_id = "266"
    stats_dir = tmp_path / "player_stats"
    stats_dir.mkdir()
    file_path = stats_dir / f"player_{player_id}_team_{team_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("invalid json content")
    stats = load_player_stats(player_id, team_id, tmp_path)
    assert stats is None
