"""Tests for nested models within the basketball data structure."""

from report_tools.models.matches import (
    EventTime,
    InOutEntry,
    PlayerStatsData,
    MatchPlayer,
    MatchTeam,
    ScoreEntry,
)
from report_tools.models.teams import TeamScore, TeamResults
from report_tools.models.players import PlayerEvolutiveStats


def test_event_time_model():
    """Test EventTime model."""
    event_time = EventTime(minute=5, second=30)
    assert event_time.minute == 5
    assert event_time.second == 30

    # Test defaults
    default_time = EventTime()
    assert default_time.minute == 0
    assert default_time.second == 0


def test_in_out_entry_model():
    """Test InOutEntry model for player substitutions."""
    entry = InOutEntry(type="IN_TYPE", minuteAbsolut=15, pointDiff=5)
    assert entry.type == "IN_TYPE"
    assert entry.minute_absolut == 15
    assert entry.point_diff == 5

    # Test with aliases
    entry2 = InOutEntry(type="OUT_TYPE", minute_absolut=20, point_diff=-3)
    assert entry2.type == "OUT_TYPE"
    assert entry2.minute_absolut == 20
    assert entry2.point_diff == -3


def test_player_stats_data_model():
    """Test PlayerStatsData model."""
    stats = PlayerStatsData(
        type=1,
        score=15,
        valoration=12,
        shotsOfOneAttempted=5,
        shotsOfTwoAttempted=8,
        shotsOfThreeAttempted=3,
        shotsOfOneSuccessful=4,
        shotsOfTwoSuccessful=6,
        shotsOfThreeSuccessful=2,
    )

    assert stats.type == 1
    assert stats.score == 15
    assert stats.valoration == 12
    assert stats.shots_of_one_attempted == 5
    assert stats.shots_of_two_attempted == 8
    assert stats.shots_of_three_attempted == 3
    assert stats.shots_of_one_successful == 4
    assert stats.shots_of_two_successful == 6
    assert stats.shots_of_three_successful == 2

    # Test defaults
    default_stats = PlayerStatsData()
    assert default_stats.score == 0
    assert default_stats.valoration == 0


def test_match_player_model():
    """Test MatchPlayer model."""
    player_data = PlayerStatsData(score=10, valoration=8)
    event_time = EventTime(minute=0, second=0)
    in_out = InOutEntry(type="IN_TYPE", minuteAbsolut=0, pointDiff=0)

    player = MatchPlayer(
        actorId=12345,
        uuid="test-uuid-123",
        playerIdsInterns=[12345],
        teamId=1001,
        name="Test Player",
        dorsal="10",
        starting=True,
        captain=False,
        sumPeriod=40,
        period=4,
        periodDuration=10,
        timePlayed=35,
        inOutsList=[in_out],
        gamePlayed=1,
        inOut=5,
        matchHasStartingPlayers=True,
        teamScore=80,
        oppScore=75,
        data=player_data,
        periods=[],
        eventTime=event_time,
    )

    assert player.actor_id == 12345
    assert player.name == "Test Player"
    assert player.dorsal == "10"
    assert player.starting is True
    assert player.captain is False
    assert player.time_played == 35
    assert len(player.in_outs_list) == 1
    assert player.team_score == 80
    assert player.data.score == 10


def test_match_team_model():
    """Test MatchTeam model."""
    team_data = PlayerStatsData(score=85, valoration=40)
    event_time = EventTime()

    team = MatchTeam(
        teamIdIntern=1001,
        teamIdExtern=2001,
        colorRgb="#FF0000",
        name="Test Team",
        shortName="TT",
        fede="test",
        players=[],
        data=team_data,
        periods=[],
        eventTime=event_time,
    )

    assert team.team_id_intern == 1001
    assert team.team_id_extern == 2001
    assert team.color_rgb == "#FF0000"
    assert team.name == "Test Team"
    assert team.short_name == "TT"
    assert team.fede == "test"
    assert team.data.score == 85


def test_score_entry_model():
    """Test ScoreEntry model."""
    score = ScoreEntry(local=45, visit=40, minuteQuarter=8, minuteAbsolute=28, period=3)

    assert score.local == 45
    assert score.visit == 40
    assert score.minute_quarter == 8
    assert score.minute_absolute == 28
    assert score.period == 3


def test_team_score_model():
    """Test TeamScore model."""
    team_score = TeamScore(ScoreFor=1200, ScoreAgainst=1100)

    assert team_score.score_for == 1200
    assert team_score.score_against == 1100


def test_team_results_model():
    """Test TeamResults model."""
    results = TeamResults(wins=18, losses=5)
    assert results.wins == 18
    assert results.losses == 5

    # Test defaults
    default_results = TeamResults()
    assert default_results.wins == 0
    assert default_results.losses == 0


def test_match_integration():
    """Test that nested models work together in a match structure."""
    from report_tools.models.matches import Match

    # Create nested components
    event_time = EventTime(minute=0, second=0)
    player_data = PlayerStatsData(score=20, valoration=15)
    team_data = PlayerStatsData(score=90, valoration=50)

    in_out = InOutEntry(type="IN_TYPE", minuteAbsolut=0, pointDiff=0)
    player = MatchPlayer(
        actorId=1,
        uuid="test",
        playerIdsInterns=[1],
        teamId=100,
        name="Player One",
        dorsal="1",
        starting=True,
        captain=True,
        sumPeriod=40,
        period=4,
        periodDuration=10,
        timePlayed=38,
        inOutsList=[in_out],
        gamePlayed=1,
        inOut=8,
        matchHasStartingPlayers=True,
        teamScore=90,
        oppScore=85,
        data=player_data,
        periods=[],
        eventTime=event_time,
    )

    team = MatchTeam(
        teamIdIntern=100,
        teamIdExtern=200,
        colorRgb="#0000FF",
        name="Blue Team",
        shortName="BLUE",
        fede="test",
        players=[player],
        data=team_data,
        periods=[],
        eventTime=event_time,
    )

    score_entry = ScoreEntry(
        local=90, visit=85, minuteQuarter=0, minuteAbsolute=40, period=4
    )

    # Create match
    match = Match(
        idMatchIntern=12345,
        idMatchExtern=54321,
        time="2024-06-22 15:00:00",
        localId=100,
        visitId=101,
        data=PlayerStatsData(score=175, valoration=80),
        eventTime=event_time,
        teams=[team],
        score=[score_entry],
    )

    # Test that all components are properly integrated
    assert match.id_match_intern == 12345
    assert len(match.teams) == 1
    assert match.teams[0].name == "Blue Team"
    assert len(match.teams[0].players) == 1
    assert match.teams[0].players[0].name == "Player One"
    assert len(match.score) == 1
    assert match.score[0].local == 90

    # Test legacy compatibility
    assert match.id == "12345"
    assert match.final_score == "90-85"


def test_player_evolutive_stats_comprehensive():
    """Test PlayerEvolutiveStats with all fields."""
    evolution_data = {
        "totalScore": 25,
        "totalScorePercent": 15.5,
        "totalScoreAvgByMatch": 12.5,
        "avgTeamPercent": 8.2,
        "timeTotalScore": 180.5,
        "sumShotsOfOneSuccessful": 8,
        "sumShotsOfOneAttempted": 12,
        "sumShotsOfOneFailed": 4,
        "sumShotsOfOneSuccessfulPercent": 66.7,
        "sumShotsOfOneSuccessfulAvgByMatch": 4.0,
        "timeSumShotsOfOneSuccessful": 90.0,
        "sumShotsOfTwoSuccessful": 7,
        "sumShotsOfTwoAttempted": 15,
        "sumShotsOfTwoFailed": 8,
        "sumShotsOfTwoSuccessfulPercent": 46.7,
        "sumShotsOfTwoSuccessfulAvgByMatch": 3.5,
        "timeSumShotsOfTwoSuccessful": 120.0,
        "sumShotsOfThreeSuccessful": 1,
        "sumShotsOfThreeAttempted": 5,
        "sumShotsOfThreeFailed": 4,
        "sumShotsOfThreeSuccessfulPercent": 20.0,
        "sumShotsOfThreeSuccessfulAvgByMatch": 0.5,
        "timeSumShotsOfThreeSuccessful": 60.0,
        "sumFieldThrowOfOneSuccessful": 0,
        "sumFieldThrowOfOneAttempted": 0,
        "sumFieldThrowOfOneFailed": 0,
        "sumFieldThrowOfOneSuccessfulPercent": 0.0,
        "sumFieldThrowOfOneSuccessfulAvgByMatch": 0.0,
        "timeSumFieldThrowOfOneSuccessful": 0.0,
        "sumFoulsReceived": 3,
        "timeSumFoulsReceived": 45.0,
        "sumFoulsReceivedAvgByMatch": 1.5,
        "sumFouls": 4,
        "timeSumFouls": 60.0,
        "sumFoulsAvgByMatch": 2.0,
        "sumAssists": 5,
        "sumAssistsAvgByMatch": 2.5,
        "sumValoration": 18,
        "sumValorationAvgByMatch": 9.0,
        "sumRebounds": 12,
        "sumReboundsAvgByMatch": 6.0,
        "sumDefensiveRebounds": 8,
        "sumDefensiveReboundsAvgByMatch": 4.0,
        "sumOffensiveRebounds": 4,
        "sumOffensiveReboundsAvgByMatch": 2.0,
        "timePlayed": 360,
        "gamesStarter": 2,
        "name": "Comprehensive Player",
        "club": "Test Club",
        "clubId": 42,
        "teamName": "Test Team",
        "teamId": 100,
        "category": "Senior",
        "matchesPlayed": 2,
        "scoreEach10Min": 6.94,
        "dorsal": 23,
        "opponentTeamName": "Rival Team",
        "opponentTeamId": 200,
        "matchCallUuid": "match-uuid-123",
        "numMatchDay": 5,
        "matchDay": "2024-06-22 15:00:00",
    }

    stats = PlayerEvolutiveStats.model_validate(evolution_data)

    # Test basic info
    assert stats.name == "Comprehensive Player"
    assert stats.total_score == 25
    assert stats.matches_played == 2
    assert stats.dorsal == 23

    # Test shooting stats
    assert stats.sum_shots_of_one_successful == 8
    assert stats.sum_shots_of_two_successful == 7
    assert stats.sum_shots_of_three_successful == 1

    # Test other stats
    assert stats.sum_assists == 5
    assert stats.sum_rebounds == 12
    assert stats.sum_fouls == 4
    assert stats.time_played == 360

    # Test match-specific fields
    assert stats.opponent_team_name == "Rival Team"
    assert stats.match_call_uuid == "match-uuid-123"
    assert stats.num_match_day == 5
