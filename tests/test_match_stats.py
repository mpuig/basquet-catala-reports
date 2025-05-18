import json
from pathlib import Path

import pytest

from report_tools.data_loaders import load_match_stats
from report_tools.models.matches import MatchStats


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
    stats = MatchStats()
    assert stats.shots_of_two_successful is None
    assert stats.shots_of_two_attempted is None
    assert stats.shots_of_three_successful is None
    assert stats.shots_of_three_attempted is None
    assert stats.shots_of_one_successful is None
    assert stats.shots_of_one_attempted is None
    assert stats.faults is None


def test_fixture_match_stats_basic_info(fixture_match_stats_file):
    """Test basic match information from the fixture file."""
    data_dir, match_id = fixture_match_stats_file

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


def test_fixture_match_stats_score_progression(fixture_match_stats_file):
    """Test score progression in the fixture file."""
    data_dir, match_id = fixture_match_stats_file

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
        prev_score = scores[i-1]
        curr_score = scores[i]

        # Score should never decrease
        assert curr_score["local"] >= prev_score["local"]
        assert curr_score["visit"] >= prev_score["visit"]

        # Minutes should be in order
        assert curr_score["minuteAbsolute"] >= prev_score["minuteAbsolute"]

        # Period should be valid
        assert 1 <= curr_score["period"] <= 4


def test_fixture_match_stats_periods(fixture_match_stats_file):
    """Test period information in the fixture file."""
    data_dir, match_id = fixture_match_stats_file

    # Load match stats using the fixture file
    stats = load_match_stats(match_id, data_dir)
    assert stats is not None

    # Test period information
    assert stats.period == 4  # Total number of period
    assert stats.period_duration == 10  # Minutes per period



def test_fixture_match_stats_moves(fixture_match_stats_file):
    """Test moves information in the fixture file."""
    data_dir, match_id = fixture_match_stats_file

    # Load match stats using the fixture file
    stats = load_match_stats(match_id, data_dir)
    assert stats is not None

    # Test moves array
    moves = stats.moves
    assert isinstance(moves, list)
    # Note: The fixture file has an empty moves array, but the structure is correct
    # The actual moves are in a separate file (match_moves.json)
