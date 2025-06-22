from datetime import datetime

from report_tools.data_loaders import load_match_moves
from report_tools.models.matches import MatchMove, MoveType


def test_load_match_moves_success(match_moves_file):
    """Test successful loading of match moves from a valid JSON file."""
    data_dir, match_id = match_moves_file
    moves = load_match_moves(match_id, data_dir)

    assert len(moves) == 2
    assert all(isinstance(move, MatchMove) for move in moves)

    # Test first move
    move1 = moves[0]
    assert move1.id_team == 1
    assert move1.actor_name == "John Doe"
    assert move1.actor_id == 123
    assert move1.actor_shirt_number == 10
    assert move1.id_move == 1
    assert move1.move == MoveType.TWO_POINT_MADE
    assert move1.min == 5
    assert move1.sec == 30
    assert move1.period == 1
    assert move1.score == "2-0"
    assert move1.team_action is True
    assert move1.event_uuid == "abc123"
    assert isinstance(move1.timestamp, datetime)
    assert move1.foul_number is None
    assert move1.license_id is None

    # Test second move
    move2 = moves[1]
    assert move2.id_team == 2
    assert move2.actor_name == "Jane Smith"
    assert move2.actor_id == 456
    assert move2.actor_shirt_number == 15
    assert move2.id_move == 2
    assert move2.move == MoveType.THREE_POINT_MADE
    assert move2.min == 6
    assert move2.sec == 15
    assert move2.period == 1
    assert move2.score == "2-3"
    assert move2.team_action is True
    assert move2.event_uuid == "def456"
    assert isinstance(move2.timestamp, datetime)
    assert move2.foul_number == 1
    assert move2.license_id == 789


def test_load_match_moves_missing_file(tmp_path):
    """Test handling of missing match moves file."""
    match_id = "non_existent_match"
    moves = load_match_moves(match_id, tmp_path)
    assert moves == []


def test_load_match_moves_invalid_json(tmp_path):
    """Test handling of invalid JSON in match moves file."""
    match_id = "invalid_json_match"
    moves_dir = tmp_path / "match_moves"
    moves_dir.mkdir()
    file_path = moves_dir / f"{match_id}.json"

    # Write invalid JSON
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("invalid json content")

    moves = load_match_moves(match_id, tmp_path)
    assert moves == []


def test_match_move_get_absolute_seconds():
    """Test the get_absolute_seconds method of MatchMove (elapsed time from start of match)."""
    move = MatchMove(
        id_team=1,
        actor_name="Test Player",
        actor_id=123,
        actor_shirt_number="10",
        id_move=1,
        move=MoveType.TWO_POINT_MADE,
        min=5,
        sec=30,
        period=1,
        score="2-0",
        team_action=True,
        event_uuid="test123",
    )

    # For period 1, min 5, sec 30, with PERIOD_LENGTH_SEC = 600 (10 minutes)
    # In basketball, the clock counts DOWN. min=5, sec=30 means 5:30 remaining
    # Elapsed time = 600 - (5*60 + 30) = 600 - 330 = 270 seconds
    # Plus elapsed from prior periods: (1-1) * 600 = 0
    # Total: 0 + 270 = 270
    assert move.get_absolute_seconds() == 270


def test_load_match_moves_from_fixture(fixture_match_moves_file):
    """Test loading match moves from the fixture file."""
    data_dir, match_id = fixture_match_moves_file
    moves = load_match_moves(match_id, data_dir)

    # The fixture file contains real match moves data
    assert len(moves) > 0
    assert all(isinstance(move, MatchMove) for move in moves)

    # Test first move from fixture
    move1 = moves[0]
    assert move1.id_team == 319131
    assert move1.actor_name == "SARA CASTILLO HERRERA"
    assert move1.actor_id == 4966665
    assert move1.actor_shirt_number == 28
    assert move1.id_move == 178
    assert move1.move == MoveType.JUMP_BALL_WON
    assert move1.min == 10
    assert move1.sec == 0
    assert move1.period == 1
    assert move1.score == "0-0"
    assert move1.team_action is False
    assert isinstance(move1.timestamp, datetime)
    assert move1.license_id == 1051490
    assert move1.event_uuid == "63aaf5b0-4dde-412b-9107-91742512126d"


def test_fixture_match_moves_sequence(fixture_match_moves_file):
    """Test the sequence of moves in the fixture file."""
    data_dir, match_id = fixture_match_moves_file
    moves = load_match_moves(match_id, data_dir)

    # Test the sequence of moves
    assert len(moves) > 0

    # Test first few moves in sequence
    assert moves[0].move == MoveType.JUMP_BALL_WON
    assert moves[1].move == MoveType.JUMP_BALL_LOST
    assert moves[2].move == MoveType.PERSONAL_FOUL_TWO_FREE_THROWS_FIRST
    assert moves[3].move == MoveType.FREE_THROW_MADE
    assert moves[4].move == MoveType.FREE_THROW_MADE

    # Test score progression
    assert moves[0].score == "0-0"
    assert moves[3].score == "0-1"
    assert moves[4].score == "0-2"


def test_fixture_match_moves_teams(fixture_match_moves_file):
    """Test team-related information in the fixture moves."""
    data_dir, match_id = fixture_match_moves_file
    moves = load_match_moves(match_id, data_dir)

    # Get unique team IDs
    team_ids = {move.id_team for move in moves}

    # Team ID 0 is used for special game events (e.g., period endings, timeouts)
    # The other two IDs should be the local and visitor teams
    assert 319130 in team_ids  # Local team
    assert 319131 in team_ids  # Visitor team
    assert len(team_ids) in (2, 3)  # Either just the two teams, or including team 0

    # If team 0 is present, verify it's used for special events
    if 0 in team_ids:
        team_0_moves = [move for move in moves if move.id_team == 0]
        for move in team_0_moves:
            # These moves should be special game events like period endings
            assert move.move in [
                MoveType.PERIOD_END,
                MoveType.TIMEOUT,
            ]  # Add other special moves if needed
            # Special events can be team actions (e.g., period endings affect both teams)
            # or individual actions (e.g., technical fouls)


def test_fixture_match_moves_timestamps(fixture_match_moves_file):
    """Test timestamp handling in the fixture moves."""
    data_dir, match_id = fixture_match_moves_file
    moves = load_match_moves(match_id, data_dir)

    # All moves should have valid timestamps
    for move in moves:
        assert isinstance(move.timestamp, datetime)
        assert move.timestamp.year == 2024
        assert move.timestamp.month == 9
        assert move.timestamp.day == 14

    # Moves should be in chronological order
    for i in range(len(moves) - 1):
        assert moves[i].timestamp <= moves[i + 1].timestamp


def test_fixture_match_moves_periods(fixture_match_moves_file):
    """Test period information in the fixture moves."""
    data_dir, match_id = fixture_match_moves_file
    moves = load_match_moves(match_id, data_dir)

    # Get unique period
    periods = {move.period for move in moves}
    assert len(periods) > 0
    assert all(1 <= period <= 4 for period in periods)  # Valid period are 1-4

    # Test period sequence
    for i in range(len(moves) - 1):
        if moves[i].period == moves[i + 1].period:
            # If same period, minutes should be in order
            assert moves[i].min >= moves[i + 1].min
        else:
            # If different period, should be sequential
            assert moves[i].period < moves[i + 1].period
