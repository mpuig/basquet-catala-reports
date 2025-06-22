from report_tools.data_loaders import load_team_stats, load_team_with_players
from report_tools.models.teams import TeamStats, Team


def test_load_team_stats_success(team_stats_fixture_file):
    """Test successful loading of team stats from a valid JSON file."""
    data_dir, team_id, season = team_stats_fixture_file
    stats = load_team_stats(team_id, season, data_dir / "team_stats")
    assert stats is not None
    assert isinstance(stats, TeamStats)

    # Test team stats
    assert stats.team_name == "CB TORELLÓ IFN - CUIDA'T ESTÈTICA"
    assert stats.team_id == 68454
    assert stats.club == "CLUB BASQUET TORELLÓ"
    assert stats.category_name == "C.C. INFANTIL FEMENÍ 1R ANY"

    # Test team results
    assert stats.team_results.wins == 16
    assert stats.team_results.losses == 7
    assert stats.sum_matches == 23

    # Test team scoring
    assert stats.team_score.score_for == 1141
    assert stats.team_score.score_against == 1100

    # Test shot statistics
    assert stats.sum_shots_of_two_successful == 415
    assert stats.sum_shots_of_two_attempted == 415
    assert stats.sum_shots_of_three_successful == 26
    assert stats.sum_shots_of_three_attempted == 26
    assert stats.sum_shots_of_one_successful == 231
    assert stats.sum_shots_of_one_attempted == 511

    # Test foul statistics
    assert stats.sum_fouls == 318
    assert stats.sum_fouls_received == 0

    # Test aggregate statistics
    assert stats.total_score == 1139
    assert stats.total_valoration == 536

    # Test per-game averages
    assert abs(stats.total_score_avg_by_match - 51.77272727272727) < 0.001
    assert abs(stats.total_fouls_avg_by_match - 14.454545454545455) < 0.001


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


def test_load_team_with_players(team_stats_fixture_file):
    """Test loading team data including players."""
    data_dir, team_id, season = team_stats_fixture_file

    team = load_team_with_players(int(team_id), season, data_dir / "team_stats")
    assert team is not None
    assert isinstance(team, Team)

    # Test basic team info
    assert team.id == 68454
    assert team.name == "CB TORELLÓ IFN - CUIDA'T ESTÈTICA"
    assert team.short_name == "CB "

    # Test team stats are loaded
    assert team.stats is not None
    assert team.stats.sum_matches == 23
    assert team.stats.team_results.wins == 16

    # Test players are loaded
    assert team.players is not None
    assert len(team.players) > 0

    # Test first player
    player1 = team.players[0]
    assert player1.name == "MARTA RIVERA SANTOS"
    assert player1.club == "CLUB BASQUET TORELLÓ"
    assert player1.team_id == 68454
    assert player1.dorsal == 28


def test_team_stats_nested_models():
    """Test nested models within TeamStats."""
    from report_tools.models.teams import TeamScore, TeamResults

    # Test TeamScore model
    team_score = TeamScore(ScoreFor=100, ScoreAgainst=80)
    assert team_score.score_for == 100
    assert team_score.score_against == 80

    # Test TeamResults model
    team_results = TeamResults(wins=15, losses=5)
    assert team_results.wins == 15
    assert team_results.losses == 5

    # Test that they can be used in TeamStats
    team_stats = TeamStats(
        teamScore=team_score,
        teamResults=team_results,
        sumMatches=20,
        sumShotsOfOneAttempted=100,
        sumShotsOfTwoAttempted=150,
        sumShotsOfThreeAttempted=50,
        sumFieldThrowOfOneAttempted=0,
        sumShotsOfOneSuccessful=80,
        sumShotsOfTwoSuccessful=120,
        sumShotsOfThreeSuccessful=30,
        sumFieldThrowOfOneSuccessful=0,
        sumShotsOfOneFailed=20,
        sumShotsOfTwoFailed=30,
        sumShotsOfThreeFailed=20,
        sumFouls=100,
        sumFoulsReceived=80,
        totalScore=230,
        totalValoration=150,
        totalScoreAvgByMatch=11.5,
        totalFoulsAvgByMatch=5.0,
        totalFoulsReceivedAvgByMatch=4.0,
        totalValorationAvgByMatch=7.5,
        shotsOfOneSuccessfulAvgByMatch=4.0,
        shotsOfTwoSuccessfulAvgByMatch=6.0,
        shotsOfThreeSuccessfulAvgByMatch=1.5,
        fieldThrowOfOneSuccessfulAvgByMatch=0.0,
        club="Test Club",
        teamName="Test Team",
        teamId=1,
        categoryName="Test Category",
    )

    assert team_stats.team_score.score_for == 100
    assert team_stats.team_results.wins == 15
    assert team_stats.sum_matches == 20
