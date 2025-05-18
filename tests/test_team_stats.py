from report_tools.data_loaders import load_team_stats
from report_tools.models.teams import TeamStats


def test_load_team_stats_success(team_stats_fixture_file):
    """Test successful loading of team stats from a valid JSON file."""
    data_dir, team_id, season = team_stats_fixture_file
    stats = load_team_stats(team_id, season, data_dir / "team_stats")
    assert stats is not None
    assert isinstance(stats, TeamStats)
    assert stats.shots_of_two_successful == 415
    assert stats.shots_of_two_attempted == 415
    assert stats.shots_of_three_successful == 26
    assert stats.shots_of_three_attempted == 26
    assert stats.shots_of_one_successful == 231
    assert stats.shots_of_one_attempted == 511
    assert stats.faults == 318


def test_load_team_stats_missing_file(tmp_path):
    """Test handling of missing team stats file."""
    team_id = "99999"
    season = "2024"
    stats = load_team_stats(team_id, season, tmp_path)
    assert stats is None


def test_load_team_stats_invalid_json(tmp_path):
    """Test handling of invalid JSON in team stats file."""
    team_id = "68454"
    season = "2024"
    stats_dir = tmp_path / "team_stats"
    stats_dir.mkdir()
    file_path = stats_dir / f"team_{team_id}_season_{season}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("invalid json content")
    stats = load_team_stats(team_id, season, tmp_path)
    assert stats is None
