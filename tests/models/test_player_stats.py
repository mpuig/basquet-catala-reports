import json
from pathlib import Path
from report_tools.data_loaders import load_player_stats
from report_tools.models.players import PlayerStats, Player, PlayerEvolutiveStats


def test_load_player_stats_success(player_stats_fixture_file):
    """Test successful loading of player stats from fixture."""
    # Use the fixture data
    fixture_file = Path(__file__).parent.parent / "fixtures" / "player_stats.json"
    with open(fixture_file, "r") as f:
        data = json.load(f)

    # Get the first player
    first_player_id = list(data.keys())[0]
    player_data = data[first_player_id]

    stats = PlayerStats.model_validate(player_data)
    assert stats is not None
    assert isinstance(stats, PlayerStats)

    # Test general stats
    general = stats.general_stats
    assert general.name == "CARLA MENDEZ TORRES"
    assert general.team_name == "VILABÀSQUET VILADECANS TARONJA"
    assert general.team_id == 69884
    assert general.club == "CLUB BASQUET FEMENI VILADECANS SPORTS"
    assert general.matches_played == 10
    assert general.dorsal == 15

    # Test shooting stats
    assert general.sum_shots_of_two_successful == 1
    assert general.sum_shots_of_two_attempted == 1
    assert general.sum_shots_of_three_successful == 0
    assert general.sum_shots_of_three_attempted == 0
    assert general.sum_shots_of_one_successful == 0
    assert general.sum_shots_of_one_attempted == 0
    assert general.sum_fouls == 11

    # Test evolution stats
    assert len(stats.evolutive_stats) > 0
    first_match = stats.evolutive_stats[0]
    assert first_match.opponent_team_name == "CBF CERDANYOLA U13"
    assert first_match.matches_played == 1


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


def test_player_loading(player_stats_fixture_file):
    """Test loading player data including evolution."""
    # Use the fixture data
    fixture_file = Path(__file__).parent.parent / "fixtures" / "player_stats.json"
    with open(fixture_file, "r") as f:
        data = json.load(f)

    first_player_id = list(data.keys())[0]
    player_data = data[first_player_id]

    # Create player
    stats = PlayerStats.model_validate(player_data)
    player = Player(
        id=int(first_player_id),
        name=stats.general_stats.name,
        club=stats.general_stats.club,
        club_id=stats.general_stats.club_id,
        team_name=stats.general_stats.team_name,
        team_id=stats.general_stats.team_id,
        category=stats.general_stats.category,
        dorsal=stats.general_stats.dorsal,
        stats=stats,
    )

    assert player is not None
    assert isinstance(player, Player)
    assert player.id == 69884
    assert player.name == "CARLA MENDEZ TORRES"
    assert player.team_id == 69884
    assert player.dorsal == 15

    # Test that stats are properly attached
    assert player.stats is not None
    assert player.stats.general_stats.matches_played == 10
    assert len(player.stats.evolutive_stats) == 10


def test_player_evolutive_stats_model():
    """Test PlayerEvolutiveStats model with sample data."""
    stats_data = {
        "totalScore": 10,
        "totalScorePercent": 0.1,
        "totalScoreAvgByMatch": 5.0,
        "avgTeamPercent": 0.05,
        "timeTotalScore": 120.0,
        "sumShotsOfOneSuccessful": 2,
        "sumShotsOfOneAttempted": 4,
        "sumShotsOfOneFailed": 2,
        "sumShotsOfOneSuccessfulPercent": 50.0,
        "sumShotsOfOneSuccessfulAvgByMatch": 1.0,
        "timeSumShotsOfOneSuccessful": 60.0,
        "sumShotsOfTwoSuccessful": 3,
        "sumShotsOfTwoAttempted": 5,
        "sumShotsOfTwoFailed": 2,
        "sumShotsOfTwoSuccessfulPercent": 60.0,
        "sumShotsOfTwoSuccessfulAvgByMatch": 1.5,
        "timeSumShotsOfTwoSuccessful": 80.0,
        "sumShotsOfThreeSuccessful": 1,
        "sumShotsOfThreeAttempted": 3,
        "sumShotsOfThreeFailed": 2,
        "sumShotsOfThreeSuccessfulPercent": 33.3,
        "sumShotsOfThreeSuccessfulAvgByMatch": 0.5,
        "timeSumShotsOfThreeSuccessful": 40.0,
        "sumFieldThrowOfOneSuccessful": 0,
        "sumFieldThrowOfOneAttempted": 0,
        "sumFieldThrowOfOneFailed": 0,
        "sumFieldThrowOfOneSuccessfulPercent": 0.0,
        "sumFieldThrowOfOneSuccessfulAvgByMatch": 0.0,
        "timeSumFieldThrowOfOneSuccessful": 0.0,
        "sumFoulsReceived": 2,
        "timeSumFoulsReceived": 30.0,
        "sumFoulsReceivedAvgByMatch": 1.0,
        "sumFouls": 3,
        "timeSumFouls": 45.0,
        "sumFoulsAvgByMatch": 1.5,
        "sumAssists": 2,
        "sumAssistsAvgByMatch": 1.0,
        "sumValoration": 8,
        "sumValorationAvgByMatch": 4.0,
        "sumRebounds": 5,
        "sumReboundsAvgByMatch": 2.5,
        "sumDefensiveRebounds": 3,
        "sumDefensiveReboundsAvgByMatch": 1.5,
        "sumOffensiveRebounds": 2,
        "sumOffensiveReboundsAvgByMatch": 1.0,
        "timePlayed": 240,
        "gamesStarter": 1,
        "name": "Test Player",
        "club": "Test Club",
        "clubId": 123,
        "teamName": "Test Team",
        "teamId": 456,
        "category": "Test Category",
        "matchesPlayed": 2,
        "scoreEach10Min": 2.5,
        "dorsal": 10,
    }

    stats = PlayerEvolutiveStats.model_validate(stats_data)
    assert stats.total_score == 10
    assert stats.name == "Test Player"
    assert stats.sum_shots_of_two_successful == 3
    assert stats.sum_fouls == 3
    assert stats.time_played == 240
    assert stats.dorsal == 10


def test_player_stats_nested_structure():
    """Test that PlayerStats properly handles nested general and evolutive stats."""
    # Test minimal valid data structure
    minimal_data = {
        "generalStats": {
            "totalScore": 5,
            "totalScorePercent": 0.0,
            "totalScoreAvgByMatch": 2.5,
            "avgTeamPercent": 0.0,
            "timeTotalScore": 60.0,
            "sumShotsOfOneSuccessful": 1,
            "sumShotsOfOneAttempted": 2,
            "sumShotsOfOneFailed": 1,
            "sumShotsOfOneSuccessfulPercent": 50.0,
            "sumShotsOfOneSuccessfulAvgByMatch": 0.5,
            "timeSumShotsOfOneSuccessful": 30.0,
            "sumShotsOfTwoSuccessful": 2,
            "sumShotsOfTwoAttempted": 3,
            "sumShotsOfTwoFailed": 1,
            "sumShotsOfTwoSuccessfulPercent": 66.7,
            "sumShotsOfTwoSuccessfulAvgByMatch": 1.0,
            "timeSumShotsOfTwoSuccessful": 40.0,
            "sumShotsOfThreeSuccessful": 0,
            "sumShotsOfThreeAttempted": 1,
            "sumShotsOfThreeFailed": 1,
            "sumShotsOfThreeSuccessfulPercent": 0.0,
            "sumShotsOfThreeSuccessfulAvgByMatch": 0.0,
            "timeSumShotsOfThreeSuccessful": 0.0,
            "sumFieldThrowOfOneSuccessful": 0,
            "sumFieldThrowOfOneAttempted": 0,
            "sumFieldThrowOfOneFailed": 0,
            "sumFieldThrowOfOneSuccessfulPercent": 0.0,
            "sumFieldThrowOfOneSuccessfulAvgByMatch": 0.0,
            "timeSumFieldThrowOfOneSuccessful": 0.0,
            "sumFoulsReceived": 1,
            "timeSumFoulsReceived": 15.0,
            "sumFoulsReceivedAvgByMatch": 0.5,
            "sumFouls": 2,
            "timeSumFouls": 30.0,
            "sumFoulsAvgByMatch": 1.0,
            "sumAssists": 1,
            "sumAssistsAvgByMatch": 0.5,
            "sumValoration": 3,
            "sumValorationAvgByMatch": 1.5,
            "sumRebounds": 2,
            "sumReboundsAvgByMatch": 1.0,
            "sumDefensiveRebounds": 1,
            "sumDefensiveReboundsAvgByMatch": 0.5,
            "sumOffensiveRebounds": 1,
            "sumOffensiveReboundsAvgByMatch": 0.5,
            "timePlayed": 120,
            "gamesStarter": 1,
            "name": "Test Player",
            "club": "Test Club",
            "clubId": 1,
            "teamName": "Test Team",
            "teamId": 1,
            "category": "Test Category",
            "matchesPlayed": 2,
            "scoreEach10Min": 2.5,
            "dorsal": 10,
        },
        "evolutiveStats": [],
    }

    stats = PlayerStats.model_validate(minimal_data)
    assert stats.general_stats.name == "Test Player"
    assert stats.general_stats.total_score == 5
    assert len(stats.evolutive_stats) == 0
