from datetime import datetime
from typing import Dict, Set, Tuple, Any

from report_tools.data_loaders import load_match_stats, load_match_moves
from report_tools.models.matches import MatchMove, MoveType, Match
from report_tools.models.teams import Team
from report_tools.stats import process_match_events
from report_tools.stats import (
    update_time_metrics,
    handle_score,
    handle_player_event,
    finalize_game_metrics,
)


def test_match_stats(match_stats_fixture_file, match_moves_fixture_file, sample_team):
    data_dir, match_id = match_stats_fixture_file
    stats = load_match_stats(match_id, data_dir)
    assert stats is not None

    moves_dir, moves_match_id = match_moves_fixture_file
    moves = load_match_moves(moves_match_id, moves_dir)
    assert moves is not None

    match = Match(
        id_match_intern=int(match_id),
        id_match_extern=928403,
        local_id=1,
        visit_id=2,
        time="2024-03-14 15:00:00",
        score=[
            {
                "local": 80,
                "visit": 75,
                "period": 4,
                "minuteQuarter": 0,
                "minuteAbsolute": 40,
            }
        ],
        teams=[],
        moves=moves,
        # Legacy compatibility
        id=match_id,
        match_date="2024-03-14",
        group_name="Group 1",
        local=sample_team,
        visitor=Team(id=2, name="Team B", short_name="TB"),
        final_score="80-75",
    )

    match_stats = process_match_events(match)
    print(
        f"Team points: {match_stats.team_points_for} - {match_stats.team_points_against}"
    )
    print(
        f"Shots: T2 {match_stats.shots_of_two_successful}/{match_stats.shots_of_two_attempted}, "
        f"T3 {match_stats.shots_of_three_successful}/{match_stats.shots_of_three_attempted}, "
        f"T1 {match_stats.shots_of_one_successful}/{match_stats.shots_of_one_attempted}"
    )
    print(f"Fouls: {match_stats.faults}")


def test_process_match_events(sample_move):
    # Create a match with required fields
    sample_match = Match(
        id_match_intern=123,
        id_match_extern=456,
        local_id=1,
        visit_id=2,
        time="2024-03-14 15:00:00",
        score=[
            {
                "local": 5,
                "visit": 0,
                "period": 1,
                "minuteQuarter": 0,
                "minuteAbsolute": 10,
            }
        ],
        teams=[],
        moves=[],
        # Legacy compatibility
        id="123",
        match_date="2024-03-14",
        group_name="Group 1",
        local=Team(id=1, name="Team A", short_name="TA"),
        visitor=Team(id=2, name="Team B", short_name="TB"),
        final_score="80-75",
    )
    # Add some moves to the match
    sample_match.moves = [
        sample_move,
        MatchMove(
            id_team=1,
            actor_name="Jane Smith",
            actor_id=102,
            actor_shirt_number="11",
            id_move=2,
            move=MoveType.THREE_POINT_MADE,
            min=6,
            sec=0,
            period=1,
            score="5-0",
            team_action=True,
            event_uuid="def456",
            timestamp=datetime.now(),
        ),
        MatchMove(
            id_team=2,
            actor_name="Bob Wilson",
            actor_id=201,
            actor_shirt_number="20",
            id_move=3,
            move=MoveType.PERSONAL_FOUL_FIRST,
            min=6,
            sec=15,
            period=1,
            score="5-0",
            team_action=False,
            event_uuid="ghi789",
            timestamp=datetime.now(),
        ),
    ]

    stats = process_match_events(sample_match)

    assert stats.id_match_intern == "123"
    assert stats.local_id == 1
    assert stats.visit_id == 2
    assert stats.shots_of_two_successful == 1
    assert stats.shots_of_three_successful == 1
    assert stats.faults == 1


def test_update_time_metrics():
    on_court: Set[str] = {"Player1", "Player2", "Player3", "Player4", "Player5"}
    lineup_stats: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    # Test with 5 players on court
    current_lineup = update_time_metrics(60.0, on_court, lineup_stats)
    assert current_lineup is not None
    assert len(current_lineup) == 5
    assert lineup_stats[current_lineup]["secs"] == 60.0

    # Test with less than 5 players
    on_court.remove("Player5")
    current_lineup = update_time_metrics(30.0, on_court, lineup_stats)
    assert current_lineup is None

    # Test with zero delta seconds
    current_lineup = update_time_metrics(0.0, on_court, lineup_stats)
    assert current_lineup is None


def test_handle_score():
    on_court: Set[str] = {"Player1", "Player2", "Player3"}
    lineup_stats: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    team_pts_f = 0
    team_pts_a = 0
    on_pts_f: Dict[str, int] = {}
    on_pts_a: Dict[str, int] = {}
    plus_minus: Dict[str, int] = {}

    # Test local team scoring
    team_pts_f, team_pts_a, on_pts_f, on_pts_a, plus_minus = handle_score(
        2,
        True,
        on_court,
        lineup_stats,
        None,
        team_pts_f,
        team_pts_a,
        on_pts_f,
        on_pts_a,
        plus_minus,
    )
    assert team_pts_f == 2
    assert team_pts_a == 0
    assert all(on_pts_f[p] == 2 for p in on_court)
    assert all(plus_minus[p] == 2 for p in on_court)

    # Test visitor team scoring
    team_pts_f, team_pts_a, on_pts_f, on_pts_a, plus_minus = handle_score(
        3,
        False,
        on_court,
        lineup_stats,
        None,
        team_pts_f,
        team_pts_a,
        on_pts_f,
        on_pts_a,
        plus_minus,
    )
    assert team_pts_f == 2
    assert team_pts_a == 3
    assert all(on_pts_a[p] == 3 for p in on_court)
    assert all(plus_minus[p] == -1 for p in on_court)

    # Test with lineup
    lineup_tuple = tuple(sorted(on_court))
    lineup_stats[lineup_tuple] = {"secs": 0, "pts_for": 0, "pts_against": 0}
    team_pts_f, team_pts_a, on_pts_f, on_pts_a, plus_minus = handle_score(
        2,
        True,
        on_court,
        lineup_stats,
        lineup_tuple,
        team_pts_f,
        team_pts_a,
        on_pts_f,
        on_pts_a,
        plus_minus,
    )
    assert lineup_stats[lineup_tuple]["pts_for"] == 2


def test_handle_player_event(sample_move):
    player_state: Dict[str, Dict] = {}
    on_court: Set[str] = set()
    minutes_played: Dict[str, float] = {}
    points: Dict[str, int] = {}
    fouls: Dict[str, int] = {}

    # Test scoring event
    player_state, on_court, minutes_played, points, fouls = handle_player_event(
        sample_move, 330.0, player_state, on_court, minutes_played, points, fouls
    )
    assert points[sample_move.actor_name] == 2
    assert sample_move.actor_name in on_court

    # Test substitution out
    exit_move = MatchMove(
        id_team=1,
        actor_name="John Doe",
        actor_id=101,
        actor_shirt_number="10",
        id_move=4,
        move=MoveType.PLAYER_EXITS,
        min=6,
        sec=0,
        period=1,
        score="5-0",
        team_action=False,
        event_uuid="jkl012",
        timestamp=datetime.now(),
    )
    player_state, on_court, minutes_played, points, fouls = handle_player_event(
        exit_move, 360.0, player_state, on_court, minutes_played, points, fouls
    )
    assert sample_move.actor_name not in on_court
    assert minutes_played[sample_move.actor_name] == 0.5  # 30 seconds = 0.5 minutes

    # Test foul
    foul_move = MatchMove(
        id_team=1,
        actor_name="John Doe",
        actor_id=101,
        actor_shirt_number="10",
        id_move=5,
        move=MoveType.PERSONAL_FOUL_FIRST,
        min=7,
        sec=0,
        period=1,
        score="5-0",
        team_action=False,
        event_uuid="mno345",
        timestamp=datetime.now(),
    )
    player_state, on_court, minutes_played, points, fouls = handle_player_event(
        foul_move, 420.0, player_state, on_court, minutes_played, points, fouls
    )
    assert fouls[sample_move.actor_name] == 1


def test_finalize_game_metrics():
    on_court: Set[str] = {"Player1", "Player2", "Player3", "Player4", "Player5"}
    player_state: Dict[str, Dict] = {
        "Player1": {"status": "in", "since": 0.0, "number": "1"},
        "Player2": {"status": "in", "since": 300.0, "number": "2"},
        "Player3": {"status": "out", "since": 600.0, "number": "3"},
    }
    lineup_stats: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    minutes_played: Dict[str, float] = {}

    # Test with remaining time
    minutes_played, lineup_stats = finalize_game_metrics(
        720.0, 600.0, on_court, player_state, lineup_stats, minutes_played
    )
    assert all(minutes_played[p] == 2.0 for p in on_court)  # 120 seconds = 2 minutes
    assert len(lineup_stats) == 1
    lineup_tuple = tuple(sorted(on_court))
    assert lineup_stats[lineup_tuple]["secs"] == 120.0

    # Test with no remaining time
    minutes_played, lineup_stats = finalize_game_metrics(
        600.0, 600.0, on_court, player_state, lineup_stats, minutes_played
    )
    assert all(minutes_played[p] == 2.0 for p in on_court)  # No additional time
    assert lineup_stats[lineup_tuple]["secs"] == 120.0
