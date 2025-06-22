import json

from report_tools.data_loaders import load_match_stats, load_match
from report_tools.models.matches import MatchStats, Match


def test_load_match_stats_success(dummy_match_stats_file):
    """Test successful loading of match stats from a valid JSON file."""
    data_dir, match_id = dummy_match_stats_file
    stats = load_match_stats(match_id, data_dir)

    assert stats is not None
    assert isinstance(stats, MatchStats)
    assert stats.shots_of_two_successful == 15
    assert stats.shots_of_two_attempted == 30
    assert stats.shots_of_three_successful == 8
    assert stats.shots_of_three_attempted == 20
    assert stats.shots_of_one_successful == 12
    assert stats.shots_of_one_attempted == 15
    assert stats.faults == 18


def test_load_match_stats_missing_file(tmp_path):
    """Test handling of missing match stats file."""
    match_id = "non_existent_match"
    stats = load_match_stats(match_id, tmp_path)
    assert stats is None


def test_load_match_stats_invalid_json(tmp_path):
    """Test handling of invalid JSON in match stats file."""
    match_id = "invalid_json_match"
    stats_dir = tmp_path / "match_stats"
    stats_dir.mkdir()
    file_path = stats_dir / f"{match_id}.json"

    # Write invalid JSON
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("invalid json content")

    stats = load_match_stats(match_id, tmp_path)
    assert stats is None


def test_load_match_stats_missing_data_field(tmp_path):
    """Test handling of JSON missing the 'data' field."""
    match_id = "missing_data_match"
    stats_dir = tmp_path / "match_stats"
    stats_dir.mkdir()
    file_path = stats_dir / f"{match_id}.json"

    # Write JSON without 'data' field
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"someOtherField": "value"}, f)

    stats = load_match_stats(match_id, tmp_path)
    assert stats is None


def test_match_stats_optional_fields():
    """Test that MatchStats fields are optional."""
    stats = MatchStats(id_match_intern=1, local_id=1, visit_id=2)
    assert stats.shots_of_two_successful == 0
    assert stats.shots_of_two_attempted == 0
    assert stats.shots_of_three_successful == 0
    assert stats.shots_of_three_attempted == 0
    assert stats.shots_of_one_successful == 0
    assert stats.shots_of_one_attempted == 0
    assert stats.faults == 0


def test_match_optional_fields():
    """Test that Match fields have proper defaults."""
    from report_tools.models.matches import PlayerStatsData, EventTime

    match = Match(
        id_match_intern=1,
        id_match_extern=1,
        time="2024-01-01",
        local_id=1,
        visit_id=2,
        data=PlayerStatsData(),
        event_time=EventTime(),
    )

    # Test defaults
    assert match.period == 4
    assert match.period_duration == 10
    assert match.sum_period == 40
    assert match.period_duration_list == []
    assert match.last_minute_used == 1
    assert match.recalculated is False
    assert len(match.moves) == 0
    assert len(match.score) == 0
    assert len(match.teams) == 0
    assert len(match.periods) == 0

    # Test legacy compatibility
    assert match.id == "1"
    assert match.match_date == ""
    assert match.group_name == ""
    assert match.final_score == ""
    assert match.local is None
    assert match.visitor is None


def test_fixture_match_stats_basic_info(match_stats_fixture_file):
    """Test basic match information from the fixture file."""
    data_dir, match_id = match_stats_fixture_file

    # Load match stats using the fixture file
    stats = load_match_stats(match_id, data_dir)
    assert stats is not None

    # Test basic match information
    assert stats.id_match_intern == 158215
    assert stats.id_match_extern == 928403
    assert stats.local_id == 319130
    assert stats.visit_id == 319131
    assert stats.period == 4
    assert stats.period_duration == 10


def test_fixture_match_stats_score_progression(match_stats_fixture_file):
    """Test score progression in the fixture file."""
    data_dir, match_id = match_stats_fixture_file

    # Load match stats using the fixture file
    stats = load_match_stats(match_id, data_dir)
    assert stats is not None

    # Test score progression
    scores = stats.score
    assert len(scores) > 0

    # Test first score
    assert scores[0]["local"] == 0
    assert scores[0]["visit"] == 0
    assert scores[0]["period"] == 1
    assert scores[0]["minuteQuarter"] == 0
    assert scores[0]["minuteAbsolute"] == 0

    # Test score progression
    for i in range(1, len(scores)):
        prev_score = scores[i - 1]
        curr_score = scores[i]

        # Score should never decrease
        assert curr_score["local"] >= prev_score["local"]
        assert curr_score["visit"] >= prev_score["visit"]

        # Minutes should be in order
        assert curr_score["minuteAbsolute"] >= prev_score["minuteAbsolute"]

        # Period should be valid
        assert 1 <= curr_score["period"] <= 4


def test_fixture_match_stats_periods(match_stats_fixture_file):
    """Test period information in the fixture file."""
    data_dir, match_id = match_stats_fixture_file

    # Load match stats using the fixture file
    stats = load_match_stats(match_id, data_dir)
    assert stats is not None

    # Test period information
    assert stats.period == 4  # Total number of period
    assert stats.period_duration == 10  # Minutes per period


def test_fixture_match_stats_moves(match_stats_fixture_file):
    """Test moves information in the fixture file."""
    data_dir, match_id = match_stats_fixture_file

    # Load match stats using the fixture file
    stats = load_match_stats(match_id, data_dir)
    assert stats is not None

    # Test moves array
    moves = stats.moves
    assert isinstance(moves, list)
    # Note: The fixture file has an empty moves array, but the structure is correct
    # The actual moves are in a separate file (match_moves.json)


def test_match_loading(match_stats_fixture_file):
    """Test loading match data from fixture."""
    data_dir, match_id = match_stats_fixture_file

    # Load match data
    match = load_match(match_id, data_dir)
    assert match is not None
    assert isinstance(match, Match)

    # Test basic match info
    assert match.id_match_intern == 158215
    assert match.id_match_extern == 928403
    assert match.local_id == 319130
    assert match.visit_id == 319131

    # Test teams data
    assert len(match.teams) == 2
    team1 = match.teams[0]
    assert team1.name == "LLUÏSOS DE GRÀCIA B"
    assert team1.short_name == "LLU"
    assert len(team1.players) > 0

    team2 = match.teams[1]
    assert team2.name == "FC MARTINENC BÀSQUET B"
    assert team2.short_name == "FCM"

    # Test score progression
    assert len(match.score) > 0
    final_score = match.score[-1]
    assert final_score.local == 54
    assert final_score.visit == 102

    # Test legacy compatibility
    assert match.id == str(match.id_match_intern)
    assert match.final_score == "54-102"

    # Test that local/visitor Team objects are created
    assert match.local is not None
    assert match.visitor is not None
    assert match.local.name == team1.name
    assert match.visitor.name == team2.name


def test_match_player_stats(match_stats_fixture_file):
    """Test player statistics within match data."""
    data_dir, match_id = match_stats_fixture_file

    match = load_match(match_id, data_dir)
    assert match is not None

    # Test first team's first player
    team1 = match.teams[0]
    player1 = team1.players[0]

    # Test first player (name may vary in fixture data)
    assert len(player1.name) > 0
    assert len(player1.dorsal) > 0
    assert isinstance(player1.starting, bool)
    assert isinstance(player1.captain, bool)
    assert player1.time_played >= 0

    # Test player stats data
    assert player1.data.score >= 0
    assert isinstance(player1.data.valoration, int)
    assert player1.data.shots_of_one_successful >= 0
    assert player1.data.shots_of_two_successful >= 0
    assert player1.data.shots_of_three_successful >= 0

    # Test in/out substitutions
    assert len(player1.in_outs_list) > 0
    first_sub = player1.in_outs_list[0]
    assert first_sub.type in ["IN_TYPE", "OUT_TYPE"]
    assert isinstance(first_sub.minute_absolut, int)
    assert isinstance(first_sub.point_diff, int)
