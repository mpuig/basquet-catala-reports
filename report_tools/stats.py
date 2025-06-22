from typing import Dict, Set, Optional, Tuple, Any

from report_tools.models.matches import (
    Match,
    MatchStats,
    MoveType,
    MatchMove,
    FOUL_MOVES,
)


def process_match_events(match: Match) -> MatchStats:
    """Process all events in a match and return match statistics.

    Args:
        match: Match object containing all events

    Returns:
        MatchStats object with processed statistics
    """
    # Initialize MatchStats with match info
    stats = MatchStats(
        id_match_intern=match.id,  # Use the string directly, do not cast to int
        id_match_extern=None,
        local_id=match.local.id,
        visit_id=match.visitor.id,
        period=4,  # Default to 4 periods
        period_duration=10,  # Default to 10 minutes per period
        sum_period=40,  # 4 periods * 10 minutes
        period_duration_list=[10, 10, 10, 10],  # 10 minutes for each period
        score=[],
        moves=[],
        shots_of_two_successful=0,
        shots_of_two_attempted=0,
        shots_of_three_successful=0,
        shots_of_three_attempted=0,
        shots_of_one_successful=0,
        shots_of_one_attempted=0,
        faults=0,
    )

    # Track team points
    team_points_for = 0
    team_points_against = 0

    # Process each move
    for move in match.moves:
        if move.move in [
            MoveType.TWO_POINT_MADE,
            MoveType.TWO_POINT_MISSED,
            MoveType.DUNK,
        ]:
            stats.shots_of_two_attempted += 1
            if move.move in [MoveType.TWO_POINT_MADE, MoveType.DUNK]:
                stats.shots_of_two_successful += 1
                if move.id_team == match.local.id:
                    team_points_for += 2
                else:
                    team_points_against += 2
        elif move.move in [
            MoveType.THREE_POINT_MADE,
            MoveType.THREE_POINT_MISSED,
            MoveType.THREE_POINTER,
        ]:
            stats.shots_of_three_attempted += 1
            if move.move in [MoveType.THREE_POINT_MADE, MoveType.THREE_POINTER]:
                stats.shots_of_three_successful += 1
                if move.id_team == match.local.id:
                    team_points_for += 3
                else:
                    team_points_against += 3
        elif move.move in [MoveType.FREE_THROW_MADE, MoveType.FREE_THROW_MISSED]:
            stats.shots_of_one_attempted += 1
            if move.move == MoveType.FREE_THROW_MADE:
                stats.shots_of_one_successful += 1
                if move.id_team == match.local.id:
                    team_points_for += 1
                else:
                    team_points_against += 1
        elif move.move in FOUL_MOVES:
            stats.faults += 1

    # Add team points to stats
    stats.team_points_for = team_points_for
    stats.team_points_against = team_points_against

    # Compute per-team stats for HTML table
    local_stats = dict(points=0, t2=0, t3=0, t1=0, fouls=0)
    visitor_stats = dict(points=0, t2=0, t3=0, t1=0, fouls=0)

    for move in match.moves:
        if move.move in [MoveType.TWO_POINT_MADE, MoveType.DUNK]:
            if move.id_team == match.local.id:
                local_stats["t2"] += 1
                local_stats["points"] += 2
            elif move.id_team == match.visitor.id:
                visitor_stats["t2"] += 1
                visitor_stats["points"] += 2
        elif move.move in [MoveType.THREE_POINT_MADE, MoveType.THREE_POINTER]:
            if move.id_team == match.local.id:
                local_stats["t3"] += 1
                local_stats["points"] += 3
            elif move.id_team == match.visitor.id:
                visitor_stats["t3"] += 1
                visitor_stats["points"] += 3
        elif move.move == MoveType.FREE_THROW_MADE:
            if move.id_team == match.local.id:
                local_stats["t1"] += 1
                local_stats["points"] += 1
            elif move.id_team == match.visitor.id:
                visitor_stats["t1"] += 1
                visitor_stats["points"] += 1
        elif move.move in FOUL_MOVES:
            if move.id_team == match.local.id:
                local_stats["fouls"] += 1
            elif move.id_team == match.visitor.id:
                visitor_stats["fouls"] += 1

    stats.local_stats = local_stats
    stats.visitor_stats = visitor_stats

    return stats


def update_time_metrics(
    delta_secs: float,
    on_court: Set[str],
    lineup_stats: Dict[Tuple[str, ...], Dict[str, Any]],
) -> Optional[Tuple[str, ...]]:
    """Updates player on-court seconds and lineup seconds based on time delta.

    Args:
        delta_secs: Time elapsed since last event
        on_court: Set of players currently on court
        lineup_stats: Dictionary tracking lineup statistics

    Returns:
        Current lineup tuple if 5 players are on court, None otherwise
    """
    current_lineup_tuple = None
    if delta_secs > 0:
        if len(on_court) == 5:
            lineup_tuple = tuple(sorted(on_court))
            if lineup_tuple not in lineup_stats:
                lineup_stats[lineup_tuple] = {"secs": 0}
            lineup_stats[lineup_tuple]["secs"] += delta_secs
            current_lineup_tuple = lineup_tuple
    return current_lineup_tuple


def handle_score(
    pts: int,
    is_local_team: bool,
    on_court: Set[str],
    lineup_stats: Dict[Tuple[str, ...], Dict[str, Any]],
    current_lineup_tuple: Optional[Tuple[str, ...]],
    team_pts_f: int,
    team_pts_a: int,
    on_pts_f: Dict[str, int],
    on_pts_a: Dict[str, int],
    plus_minus: Dict[str, int],
) -> Tuple[int, int, Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Updates plus/minus, on/off points, lineup points, and opponent points for a scoring event.

    Args:
        pts: Points scored
        is_local_team: Whether the scoring team is the local team
        on_court: Set of players currently on court
        lineup_stats: Dictionary tracking lineup statistics
        current_lineup_tuple: Current lineup tuple if 5 players are on court
        team_pts_f: Team points for
        team_pts_a: Team points against
        on_pts_f: Points scored while each player is on court
        on_pts_a: Points scored against while each player is on court
        plus_minus: Plus/minus rating for each player

    Returns:
        Updated team_pts_f, team_pts_a, on_pts_f, on_pts_a, plus_minus
    """
    point_diff = pts if is_local_team else -pts
    for p in on_court:
        if p not in plus_minus:
            plus_minus[p] = 0
        if p not in on_pts_f:
            on_pts_f[p] = 0
        if p not in on_pts_a:
            on_pts_a[p] = 0
        plus_minus[p] += point_diff
        if is_local_team:
            on_pts_f[p] += pts
        else:
            on_pts_a[p] += pts

    if is_local_team:
        team_pts_f += pts
    else:
        team_pts_a += pts

    if current_lineup_tuple and current_lineup_tuple in lineup_stats:
        if "pts_for" not in lineup_stats[current_lineup_tuple]:
            lineup_stats[current_lineup_tuple]["pts_for"] = 0
        if "pts_against" not in lineup_stats[current_lineup_tuple]:
            lineup_stats[current_lineup_tuple]["pts_against"] = 0
        if is_local_team:
            lineup_stats[current_lineup_tuple]["pts_for"] += pts
        else:
            lineup_stats[current_lineup_tuple]["pts_against"] += pts

    return team_pts_f, team_pts_a, on_pts_f, on_pts_a, plus_minus


def handle_player_event(
    move: MatchMove,
    abs_seconds: float,
    player_state: Dict[str, Dict],
    on_court: Set[str],
    minutes_played: Dict[str, float],
    points: Dict[str, int],
    fouls: Dict[str, int],
) -> Tuple[Dict[str, Dict], Set[str], Dict[str, float], Dict[str, int], Dict[str, int]]:
    """Handles player-specific stats (points, fouls) and substitutions.

    Args:
        move: The move event to process
        abs_seconds: Absolute time in seconds
        player_state: Dictionary tracking player state
        on_court: Set of players currently on court
        minutes_played: Minutes played by each player
        points: Points scored by each player
        fouls: Fouls committed by each player

    Returns:
        Updated player_state, on_court, minutes_played, points, fouls
    """
    player = move.actor_name
    if player not in player_state:
        player_state[player] = {
            "status": "out",
            "since": 0.0,
            "number": move.actor_shirt_number,
        }
        minutes_played[player] = 0.0
        points[player] = 0
        fouls[player] = 0

    if move.move == MoveType.PLAYER_ENTERS:
        player_state[player]["status"] = "in"
        player_state[player]["since"] = abs_seconds
        on_court.add(player)
    elif move.move == MoveType.PLAYER_EXITS:
        if player_state[player]["status"] == "in":
            minutes_played[player] += (abs_seconds - player_state[player]["since"]) / 60
        player_state[player]["status"] = "out"
        player_state[player]["since"] = abs_seconds
        on_court.discard(player)
    elif move.move in [
        MoveType.TWO_POINT_MADE,
        MoveType.THREE_POINT_MADE,
        MoveType.FREE_THROW_MADE,
    ]:
        # If player is not on court, add them
        if player not in on_court:
            player_state[player]["status"] = "in"
            player_state[player]["since"] = abs_seconds
            on_court.add(player)
        points[player] += (
            2
            if move.move == MoveType.TWO_POINT_MADE
            else 3 if move.move == MoveType.THREE_POINT_MADE else 1
        )
    elif move.move in [
        MoveType.PERSONAL_FOUL_FIRST,
        MoveType.PERSONAL_FOUL_SECOND,
        MoveType.PERSONAL_FOUL_THIRD,
        MoveType.PERSONAL_FOUL_FOURTH,
        MoveType.PERSONAL_FOUL_FIFTH,
    ]:
        # If player is not on court, add them
        if player not in on_court:
            player_state[player]["status"] = "in"
            player_state[player]["since"] = abs_seconds
            on_court.add(player)
        fouls[player] += 1

    return player_state, on_court, minutes_played, points, fouls


def finalize_game_metrics(
    game_end_sec: float,
    last_event_abs_seconds: float,
    on_court: Set[str],
    player_state: Dict[str, Dict],
    lineup_stats: Dict[Tuple[str, ...], Dict[str, Any]],
    minutes_played: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[Tuple[str, ...], Dict[str, Any]]]:
    """Calculates remaining metrics at game end.

    Args:
        game_end_sec: End time of the game in seconds
        last_event_abs_seconds: Time of last event in seconds
        on_court: Set of players currently on court
        player_state: Dictionary tracking player state
        lineup_stats: Dictionary tracking lineup statistics
        minutes_played: Minutes played by each player

    Returns:
        Updated minutes_played and lineup_stats
    """
    # Handle remaining time for players still on court
    remaining_secs = game_end_sec - last_event_abs_seconds
    if remaining_secs > 0:
        for p in on_court:
            if p not in minutes_played:
                minutes_played[p] = 0.0
            minutes_played[p] += remaining_secs / 60
        if len(on_court) == 5:
            lineup_tuple = tuple(sorted(on_court))
            if lineup_tuple not in lineup_stats:
                lineup_stats[lineup_tuple] = {"secs": 0}
            lineup_stats[lineup_tuple]["secs"] += remaining_secs

    return minutes_played, lineup_stats
