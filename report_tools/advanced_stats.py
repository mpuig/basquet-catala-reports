from collections import defaultdict

import pandas as pd

from report_tools.models.matches import Match, MoveType, FOUL_MOVES


def _get_internal_team_ids(match: Match):
    """Get the correct internal team IDs for local and visitor teams."""
    moves_team_ids = (
        set(move.id_team for move in match.moves[:20]) if match.moves else set()
    )
    unique_team_ids = list(moves_team_ids)

    if len(unique_team_ids) < 2:
        return None, None

    # Map internal team IDs to external team IDs by checking player names
    local_team_id = None
    visitor_team_id = None

    # Sample players from each team to determine mapping
    team_samples = {tid: [] for tid in unique_team_ids}
    for move in match.moves[:50]:  # Check more moves for better mapping
        if move.id_team in team_samples and len(team_samples[move.id_team]) < 5:
            team_samples[move.id_team].append(move.actor_name)

    # Get player names from match teams data (which has correct external team mapping)
    local_players_in_match = set()
    visitor_players_in_match = set()

    if hasattr(match, "teams") and match.teams:
        for team_data in match.teams:
            if hasattr(team_data, "players") and hasattr(team_data, "teamIdExtern"):
                for player in team_data.players:
                    player_name = getattr(player, "name", None) or getattr(
                        player, "playerName", None
                    )
                    if player_name:
                        if team_data.teamIdExtern == match.local.id:
                            local_players_in_match.add(player_name)
                        elif team_data.teamIdExtern == match.visitor.id:
                            visitor_players_in_match.add(player_name)

    # Map internal team IDs based on player name overlap
    for tid in unique_team_ids:
        tid_players = set(team_samples[tid])
        local_overlap = len(tid_players.intersection(local_players_in_match))
        visitor_overlap = len(tid_players.intersection(visitor_players_in_match))

        if local_overlap > visitor_overlap:
            local_team_id = tid
        elif visitor_overlap > local_overlap:
            visitor_team_id = tid

    # Fallback: assign remaining team ID
    if local_team_id and not visitor_team_id:
        visitor_team_id = [tid for tid in unique_team_ids if tid != local_team_id][0]
    elif visitor_team_id and not local_team_id:
        local_team_id = [tid for tid in unique_team_ids if tid != visitor_team_id][0]
    elif not local_team_id and not visitor_team_id:
        # Last resort: use order from unique_team_ids
        local_team_id = unique_team_ids[0]
        visitor_team_id = unique_team_ids[1]

    return local_team_id, visitor_team_id


def calculate_player_aggregate_stats(match: Match) -> pd.DataFrame:
    """
    Returns a DataFrame with player aggregate stats:
    Player, #, Mins, PTS, T3, T2, T1, Fouls, +/-, team_id
    """
    moves = match.moves
    if not moves:
        return pd.DataFrame()

    # Get correct internal team IDs
    local_internal_id, visitor_internal_id = _get_internal_team_ids(match)
    if not local_internal_id or not visitor_internal_id:
        return pd.DataFrame()

    player_stats = defaultdict(
        lambda: {
            "#": None,
            "Mins": 0.0,
            "PTS": 0,
            "T3": 0,
            "T2": 0,
            "T1": 0,
            "Fouls": 0,
            "+/-": 0,
            "team_id": None,
        }
    )

    # Track player time using proper 4-period basketball structure
    player_total_seconds = defaultdict(float)
    on_court_local = set()
    on_court_visit = set()

    # Group moves by period to handle period boundaries correctly
    moves_by_period = defaultdict(list)
    for move in moves:
        period = getattr(move, "period", 1)
        moves_by_period[period].append(move)

    # Process each period separately (basketball has 4 periods)
    for period in range(1, 5):
        if period not in moves_by_period:
            continue

        period_moves = moves_by_period[period]
        if not period_moves:
            continue

        # Track players on court during this period
        period_on_court_local = on_court_local.copy()  # Carry over from previous period
        period_on_court_visit = on_court_visit.copy()
        player_enter_times = {}  # Reset enter times for this period

        # Set starting times for players who carried over from previous period
        # In basketball, periods count down from 10:00 to 0:00
        # So period start time in absolute seconds is (period-1)*600
        period_start_time = (period - 1) * 600  # Absolute seconds for period start
        for player in period_on_court_local | period_on_court_visit:
            player_enter_times[player] = period_start_time

        # Process moves in this period
        for move in period_moves:
            current_time = (
                move.get_absolute_seconds()
                if hasattr(move, "get_absolute_seconds")
                else period_start_time
            )

            name = move.actor_name
            player_stats[name]["#"] = move.actor_shirt_number
            # Only set team_id if not already set or is None
            if player_stats[name]["team_id"] is None:
                player_stats[name]["team_id"] = move.id_team

            # Infer starting lineup from first actions in period 1
            if period == 1 and move.move not in {
                MoveType.PLAYER_ENTERS,
                MoveType.PLAYER_EXITS,
            }:
                if (
                    name not in period_on_court_local
                    and name not in period_on_court_visit
                ):
                    # Player doing action in period 1 without ENTER event = starting lineup
                    player_enter_times[name] = 0.0  # Started at game time 0
                    if move.id_team == local_internal_id:
                        period_on_court_local.add(name)
                    elif move.id_team == visitor_internal_id:
                        period_on_court_visit.add(name)

            # Track points for statistics
            if move.move in {MoveType.THREE_POINT_MADE, MoveType.THREE_POINTER}:
                player_stats[name]["T3"] += 1
                player_stats[name]["PTS"] += 3
                pts = 3
            elif move.move in {MoveType.TWO_POINT_MADE, MoveType.DUNK}:
                player_stats[name]["T2"] += 1
                player_stats[name]["PTS"] += 2
                pts = 2
            elif move.move == MoveType.FREE_THROW_MADE:
                player_stats[name]["T1"] += 1
                player_stats[name]["PTS"] += 1
                pts = 1
            else:
                pts = 0

            if move.move in FOUL_MOVES:
                player_stats[name]["Fouls"] += 1

            # Handle substitutions
            if move.move == MoveType.PLAYER_ENTERS:
                player_enter_times[name] = current_time
                if move.id_team == local_internal_id:
                    period_on_court_local.add(name)
                elif move.id_team == visitor_internal_id:
                    period_on_court_visit.add(name)
            elif move.move == MoveType.PLAYER_EXITS:
                if name in player_enter_times:
                    # Calculate time for this stint in this period
                    stint_time = current_time - player_enter_times[name]
                    player_total_seconds[name] += stint_time
                    del player_enter_times[name]

                if move.id_team == local_internal_id:
                    period_on_court_local.discard(name)
                elif move.id_team == visitor_internal_id:
                    period_on_court_visit.discard(name)

            # Plus/minus calculation
            if pts > 0:
                if move.id_team == local_internal_id:
                    for p in period_on_court_local:
                        player_stats[p]["+/-"] += pts
                    for p in period_on_court_visit:
                        player_stats[p]["+/-"] -= pts
                elif move.id_team == visitor_internal_id:
                    for p in period_on_court_visit:
                        player_stats[p]["+/-"] += pts
                    for p in period_on_court_local:
                        player_stats[p]["+/-"] -= pts

        # End of period: add remaining time for players still on court
        # Period end time is start of next period (or game end for period 4)
        period_end_time = period * 600  # Absolute seconds for period end
        for player_name, enter_time in player_enter_times.items():
            stint_time = period_end_time - enter_time
            player_total_seconds[player_name] += stint_time

        # Update global on_court sets for next period
        on_court_local = period_on_court_local.copy()
        on_court_visit = period_on_court_visit.copy()

    # Convert seconds to minutes and assign to player stats
    for player_name, total_seconds in player_total_seconds.items():
        if player_name in player_stats:
            # Ensure minutes are non-negative (handle any timing calculation issues)
            minutes = max(0, round(total_seconds / 60, 1))
            player_stats[player_name]["Mins"] = minutes

    df = pd.DataFrame([{"Player": k, **v} for k, v in player_stats.items()])

    return df


def calculate_lineup_stats(match: Match, team_id: int = None) -> pd.DataFrame:
    """
    Returns a DataFrame with lineup stats for both teams:
    lineup, mins, usage_%, NetRtg
    """
    moves = match.moves
    if not moves:
        return pd.DataFrame()

    # Get correct internal team IDs
    local_internal_id, visitor_internal_id = _get_internal_team_ids(match)
    if not local_internal_id or not visitor_internal_id:
        return pd.DataFrame()

    # Track lineups over time
    lineup_stats = defaultdict(lambda: {"mins": 0.0, "pts_for": 0, "pts_against": 0})
    on_court = {
        local_internal_id: set(),
        visitor_internal_id: set(),
    }  # Track by internal team ID
    last_time = 0.0

    for move in moves:
        current_time = (
            move.get_absolute_seconds()
            if hasattr(move, "get_absolute_seconds")
            else 0.0
        )
        time_delta = current_time - last_time

        # Update lineup time for current lineup
        if time_delta > 0:
            for team_id_key in [local_internal_id, visitor_internal_id]:
                if len(on_court[team_id_key]) >= 3:  # Minimum lineup size
                    lineup = " - ".join(sorted(on_court[team_id_key]))
                    lineup_stats[lineup]["mins"] += time_delta / 60

        # Handle substitutions
        if move.move == MoveType.PLAYER_ENTERS:
            if move.id_team in on_court:
                on_court[move.id_team].add(move.actor_name)
        elif move.move == MoveType.PLAYER_EXITS:
            if move.id_team in on_court:
                on_court[move.id_team].discard(move.actor_name)

        # Track scoring for lineups
        pts = 0
        if move.move in {MoveType.THREE_POINT_MADE, MoveType.THREE_POINTER}:
            pts = 3
        elif move.move in {MoveType.TWO_POINT_MADE, MoveType.DUNK}:
            pts = 2
        elif move.move == MoveType.FREE_THROW_MADE:
            pts = 1

        if pts > 0:
            scoring_team = move.id_team
            other_team = (
                visitor_internal_id
                if scoring_team == local_internal_id
                else local_internal_id
            )

            # Add points for scoring team's current lineup
            if len(on_court[scoring_team]) >= 3:
                lineup = " - ".join(sorted(on_court[scoring_team]))
                lineup_stats[lineup]["pts_for"] += pts

            # Add points against for other team's current lineup
            if len(on_court[other_team]) >= 3:
                lineup = " - ".join(sorted(on_court[other_team]))
                lineup_stats[lineup]["pts_against"] += pts

        last_time = current_time

    # Convert to DataFrame
    data = []
    total_game_time = (
        max([move.get_absolute_seconds() for move in moves]) / 60 if moves else 40
    )

    for lineup, stats in lineup_stats.items():
        if stats["mins"] > 0:  # Only include lineups that played
            usage_pct = (stats["mins"] / total_game_time) * 100
            net_rtg = (
                ((stats["pts_for"] - stats["pts_against"]) / stats["mins"]) * 40
                if stats["mins"] > 0
                else 0
            )

            data.append(
                {
                    "lineup": lineup,
                    "mins": round(stats["mins"], 1),
                    "usage_%": round(usage_pct, 1),
                    "NetRtg": round(net_rtg, 1),
                }
            )

    df = pd.DataFrame(data)
    return df.sort_values("NetRtg", ascending=False) if not df.empty else df


def calculate_on_off_stats(match: Match, team_id: int) -> pd.DataFrame:
    """
    Returns a DataFrame with on/off stats for players from the target team only:
    Player, Mins_ON, On_Net, Off_Net, ON-OFF
    """
    moves = match.moves
    if not moves:
        return pd.DataFrame()

    # Get correct internal team IDs
    local_internal_id, visitor_internal_id = _get_internal_team_ids(match)
    if not local_internal_id or not visitor_internal_id:
        return pd.DataFrame()

    # Use the provided team_id (should be one of the internal team IDs)
    target_team_id = team_id

    player_stats = defaultdict(
        lambda: {"on_seconds": 0.0, "on_pts_for": 0, "on_pts_against": 0}
    )

    # Track players on court using proper 4-period basketball structure
    on_court = set()
    team_total_pts_for = 0
    team_total_pts_against = 0

    # Group moves by period to handle period boundaries correctly
    moves_by_period = defaultdict(list)
    for move in moves:
        period = getattr(move, "period", 1)
        moves_by_period[period].append(move)

    # Process each period separately (basketball has 4 periods)
    for period in range(1, 5):
        if period not in moves_by_period:
            continue

        period_moves = moves_by_period[period]
        if not period_moves:
            continue

        # Track players on court during this period
        period_on_court = on_court.copy()  # Carry over from previous period
        player_enter_times = {}  # Reset enter times for this period

        # Set starting times for players who carried over from previous period
        period_start_time = (period - 1) * 600  # Absolute seconds for period start
        for player in period_on_court:
            player_enter_times[player] = period_start_time

        # Process moves in this period
        for move in period_moves:
            current_time = (
                move.get_absolute_seconds()
                if hasattr(move, "get_absolute_seconds")
                else period_start_time
            )

            # Infer starting lineup from first actions in period 1
            if period == 1 and move.move not in {
                MoveType.PLAYER_ENTERS,
                MoveType.PLAYER_EXITS,
            }:
                if (
                    move.id_team == target_team_id
                    and move.actor_name not in period_on_court
                ):
                    # Player doing action in period 1 without ENTER event = starting lineup
                    player_enter_times[move.actor_name] = 0.0  # Started at game time 0
                    period_on_court.add(move.actor_name)

            # Handle scoring
            pts = 0
            if move.move in {MoveType.THREE_POINT_MADE, MoveType.THREE_POINTER}:
                pts = 3
            elif move.move in {MoveType.TWO_POINT_MADE, MoveType.DUNK}:
                pts = 2
            elif move.move == MoveType.FREE_THROW_MADE:
                pts = 1

            if pts > 0:
                if move.id_team == target_team_id:
                    team_total_pts_for += pts
                    # Points for the team - add to on-court players
                    for player in period_on_court:
                        player_stats[player]["on_pts_for"] += pts
                else:
                    team_total_pts_against += pts
                    # Points against the team - add to on-court players
                    for player in period_on_court:
                        player_stats[player]["on_pts_against"] += pts

            # Handle substitutions
            if move.move == MoveType.PLAYER_ENTERS and move.id_team == target_team_id:
                player_enter_times[move.actor_name] = current_time
                period_on_court.add(move.actor_name)
            elif move.move == MoveType.PLAYER_EXITS and move.id_team == target_team_id:
                if move.actor_name in player_enter_times:
                    # Calculate time for this stint in this period
                    stint_time = current_time - player_enter_times[move.actor_name]
                    player_stats[move.actor_name]["on_seconds"] += stint_time
                    del player_enter_times[move.actor_name]
                period_on_court.discard(move.actor_name)

        # End of period: add remaining time for players still on court
        period_end_time = period * 600  # Absolute seconds for period end
        for player_name, enter_time in player_enter_times.items():
            stint_time = period_end_time - enter_time
            player_stats[player_name]["on_seconds"] += stint_time

        # Update global on_court set for next period
        on_court = period_on_court.copy()

    # Calculate total game time (standard 40 minutes)
    total_game_seconds = 2400  # 40 minutes * 60 seconds

    # Build DataFrame
    if not player_stats:
        return pd.DataFrame()

    data = []
    for player, stats in player_stats.items():
        if stats["on_seconds"] > 0:  # Only include players who played
            mins_on = round(stats["on_seconds"] / 60, 2)
            on_net = (
                round((stats["on_pts_for"] - stats["on_pts_against"]) * 40 / mins_on, 1)
                if mins_on > 0
                else 0.0
            )

            # Calculate off stats
            mins_off = round((total_game_seconds - stats["on_seconds"]) / 60, 2)
            if mins_off > 0:
                off_pts_for = team_total_pts_for - stats["on_pts_for"]
                off_pts_against = team_total_pts_against - stats["on_pts_against"]
                off_net = round((off_pts_for - off_pts_against) * 40 / mins_off, 1)
                on_off = round(on_net - off_net, 1)
            else:
                off_net = "—"
                on_off = "—"

            data.append(
                {
                    "Player": player,
                    "Mins_ON": mins_on,
                    "On_Net": on_net,
                    "Off_Net": off_net,
                    "ON-OFF": on_off,
                }
            )

    return pd.DataFrame(data)


def calculate_pairwise_minutes(match: Match) -> pd.DataFrame:
    """
    Returns a DataFrame (matrix) of pairwise minutes played together.
    """
    moves = match.moves
    pairwise_secs = defaultdict(lambda: defaultdict(float))
    on_court = set()
    last_time = 0.0
    for move in moves:
        current_time = (
            move.get_absolute_seconds()
            if hasattr(move, "get_absolute_seconds")
            else 0.0
        )
        delta = current_time - last_time
        if delta > 0:
            for p1 in on_court:
                for p2 in on_court:
                    pairwise_secs[p1][p2] += delta
        if move.move == MoveType.PLAYER_ENTERS:
            on_court.add(move.actor_name)
        elif move.move == MoveType.PLAYER_EXITS:
            on_court.discard(move.actor_name)
        last_time = current_time
    players = sorted(pairwise_secs.keys())
    matrix = pd.DataFrame(index=players, columns=players, dtype=int)
    for p1 in players:
        for p2 in players:
            matrix.loc[p1, p2] = int(round(pairwise_secs[p1][p2] / 60))
    return matrix.astype(int)


def calculate_game_log(match: Match) -> pd.DataFrame:
    """
    Returns a DataFrame with the game log:
    game_idx, player, mins, pts, +/-, drtg
    """
    moves = match.moves
    per_game_minutes = defaultdict(float)
    per_game_points = defaultdict(int)
    per_game_plusminus = defaultdict(int)
    on_court = set()
    last_time = 0.0
    for move in moves:
        current_time = (
            move.get_absolute_seconds()
            if hasattr(move, "get_absolute_seconds")
            else 0.0
        )
        delta = current_time - last_time
        if delta > 0:
            for p in on_court:
                per_game_minutes[p] += delta
        pts = 0
        if move.move == MoveType.FREE_THROW_MADE:
            pts = 1
        elif move.move in {MoveType.TWO_POINT_MADE, MoveType.DUNK}:
            pts = 2
        elif move.move in {MoveType.THREE_POINT_MADE, MoveType.THREE_POINTER}:
            pts = 3
        if pts > 0:
            per_game_points[move.actor_name] += pts
            if move.id_team == match.local.id:
                for p_on_court in on_court:
                    per_game_plusminus[p_on_court] += pts
            else:
                for p_on_court in on_court:
                    per_game_plusminus[p_on_court] -= pts
        if move.move == MoveType.PLAYER_ENTERS:
            on_court.add(move.actor_name)
        elif move.move == MoveType.PLAYER_EXITS:
            on_court.discard(move.actor_name)
        last_time = current_time
    records = []
    for pname, mins in per_game_minutes.items():
        records.append(
            {
                "game_idx": 1,
                "player": pname,
                "mins": round(mins / 60, 1),
                "pts": per_game_points.get(pname, 0),
                "+/-": per_game_plusminus.get(pname, 0),
                "drtg": None,  # Not enough info for DRtg in this simple version
            }
        )
    return pd.DataFrame(records)


def calculate_player_evolution(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with rolling averages for each player:
    game_idx, player, roll_pts, roll_mins, roll_pm, roll_drtg
    """
    if game_logs.empty:
        return pd.DataFrame()
    df = game_logs.copy()
    df["roll_pts"] = df.groupby("player")["pts"].transform(
        lambda s: s.rolling(3, 1).mean()
    )
    df["roll_mins"] = df.groupby("player")["mins"].transform(
        lambda s: s.rolling(3, 1).mean()
    )
    df["roll_pm"] = df.groupby("player")["+/-"].transform(
        lambda s: s.rolling(3, 1).mean()
    )
    if "drtg" in df.columns:
        df["roll_drtg"] = df.groupby("player")["drtg"].transform(
            lambda s: s.rolling(window=3, min_periods=1).mean()
        )
    return df
