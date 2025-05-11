from collections import defaultdict
from typing import Dict, List, Any, Sequence, Optional, Set

import pandas as pd

from report_tools.logger import logger
from report_tools.models import PlayerAggregate
from report_tools.utils import get_absolute_seconds, shorten_name


def _match_external_team_id(team_info: dict, target_team_id: str) -> bool:
    return str(team_info.get("teamIdExtern")) == target_team_id


class StatsCalculator:
    """Calculates and stores statistics for a single team over a set of matches.

    Processes play-by-play event data to compute player aggregates, on/off court
    metrics, pairwise minutes, and lineup performance.

    Designed to be instantiated once per report generation scope (e.g., per match
    in this script, or per group if used for aggregation).
    """

    points_map = {"Cistella de 1": 1, "Cistella de 2": 2, "Cistella de 3": 3}
    foul_keywords = ("Personal", "Tècnica", "Antiesportiva", "Desqualificant")

    def __init__(self, team_id: str):
        self.team_id = team_id
        self.players: Dict[str, PlayerAggregate] = defaultdict(PlayerAggregate)
        self.pairwise_secs: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # --- On/Off tracking ---
        self.on_secs: Dict[str, float] = defaultdict(
            float
        )  # seconds with player ON court
        self.on_pts_f: Dict[str, int] = defaultdict(int)  # points for while ON
        self.on_pts_a: Dict[str, int] = defaultdict(int)  # points against while ON
        self.team_pts_f: int = 0  # total points for (team)
        self.team_pts_a: int = 0  # total points against
        # --- Evolution & lineup tracking ---
        self.game_log: List[dict] = []  # one row per player per game
        self.lineup_stats: Dict[tuple, Dict[str, Any]] = defaultdict(
            lambda: {"secs": 0.0, "pts_for": 0, "pts_against": 0}
        )
        self.plus_minus: Dict[str, int] = defaultdict(int)  # +/- por jugadora
        # --- Game-level summary ---
        self.game_summaries: List[dict] = (
            []
        )  # Store {'game_idx': int, 'opponent_pts': int}
        self.processed_matches: set[str] = (
            set()
        )  # Track processed match IDs for summaries

    # ==================================================================
    # Public API
    # ==================================================================

    def process(
        self,
        schedule: pd.DataFrame,
        all_moves: Dict[str, Sequence[dict]],
        all_stats: Dict[str, dict],
    ) -> None:
        """Processes all matches found in the schedule data.

        Iterates through the provided schedule, loads corresponding moves and stats
        data, maps the target team ID, and processes events for each match using
        internal helper methods.

        Args:
            schedule: DataFrame containing schedule information (must include 'match_id').
            all_moves: Dictionary mapping match_id to a list of play-by-play events.
            all_stats: Dictionary mapping match_id to aggregated match stats JSON data.
        """
        processed_matches = 0
        skipped_matches_no_moves = 0
        skipped_matches_no_stats = 0
        skipped_matches_id_mapping = 0

        for index, row_series in schedule.iterrows():
            match_id = str(row_series.get("match_id", ""))
            if not match_id or match_id.lower() == "nan":
                continue

            events = all_moves.get(match_id)
            stats_data = all_stats.get(match_id)

            if not events or not stats_data:
                if not events:
                    skipped_matches_no_moves += 1
                if not stats_data:
                    skipped_matches_no_stats += 1
                continue

            internal_id, target_team_name = self._get_internal_id_and_name(
                match_id, stats_data
            )
            if internal_id is None:
                skipped_matches_id_mapping += 1
                continue  # Mapping failed

            # Process the single game using the mapped internal ID
            self._process_single_game_events(
                match_id, events, internal_id, target_team_name
            )
            processed_matches += 1

    # ==================================================================
    # Internal Processing Logic - Single Game
    # ==================================================================

    def _get_internal_id_and_name(
        self, match_id: str, stats_data: dict
    ) -> tuple[Optional[int], Optional[str]]:
        """Finds the internal ID and name for the target team using match_stats data."""
        target_team_schedule_id: str = self.team_id
        match_stats_teams: list = stats_data.get("teams", [])
        internal_id: Optional[int] = None
        team_name: str = "Unknown"

        for team_info in match_stats_teams:
            if _match_external_team_id(team_info, target_team_schedule_id):
                internal_id = team_info.get("teamIdIntern")
                team_name = team_info.get("name", team_name)
                break

        if internal_id is None:
            logger.debug(
                "Skipping match %s for schedule team ID %s – Cannot map to internal idTeam using match_stats data.",
                match_id,
                target_team_schedule_id,
            )
            return None, None

        return internal_id, team_name

    def _process_single_game_events(
        self,
        match_id: str,
        events: Sequence[dict],
        internal_id: int,  # Target team's internal ID for this match
        target_team_name: str,  # Target team's name for this match
    ) -> None:
        """Core loop through events for a single game, updating state and stats."""
        # ---- Per-game accumulators and state ----
        per_game_minutes: Dict[str, float] = defaultdict(float)
        per_game_points: Dict[str, int] = defaultdict(int)
        per_game_plusminus: Dict[str, int] = defaultdict(int)
        per_game_on_pts_a: Dict[str, int] = defaultdict(int)  # For per-game DRtg
        game_opponent_pts: int = 0
        player_state: Dict[str, Dict] = {}  # Tracks {'status', 'since', 'number'}
        on_court: Set[str] = set()  # Players currently on court
        last_event_abs_seconds = 0.0
        game_end_sec = 0.0

        for event in events:
            event_time_info = {
                "period": event.get("period", 0),
                "minute": event.get("min", 0),
                "second": event.get("sec", 0),
            }
            event_abs_seconds = get_absolute_seconds(**event_time_info)
            delta_secs = event_abs_seconds - last_event_abs_seconds

            actor_name = event.get("actorName", "")
            event_team_id = event.get("idTeam")
            is_target_team_event = event_team_id == internal_id

            # 1. Update On/Off and Lineup time metrics based on elapsed time
            current_lineup_tuple = self._update_time_metrics(delta_secs, on_court)

            # 2. Handle scoring events
            pts = self.points_map.get(event.get("move", ""), 0)
            if pts > 0:
                game_opponent_pts = self._handle_score(
                    pts,
                    is_target_team_event,
                    on_court,
                    per_game_plusminus,
                    per_game_on_pts_a,
                    current_lineup_tuple,
                    game_opponent_pts,
                )

            # 3. Update game end time tracker
            last_event_abs_seconds = event_abs_seconds
            game_end_sec = max(game_end_sec, event_abs_seconds)

            # 4. Handle player-specific events (stats, substitutions)
            if is_target_team_event and actor_name and actor_name != target_team_name:
                self._handle_player_event(
                    event,
                    actor_name,
                    event_abs_seconds,
                    player_state,
                    on_court,
                    per_game_minutes,
                    per_game_points,
                )

        # --- End of Game Processing ---
        self._finalize_game_metrics(
            game_end_sec,
            last_event_abs_seconds,
            on_court,
            player_state,
            per_game_minutes,
            per_game_points,
            per_game_plusminus,
            per_game_on_pts_a,
            game_opponent_pts,
            match_id,
        )

    def _update_time_metrics(
        self, delta_secs: float, on_court: Set[str]
    ) -> Optional[tuple]:
        """Updates player on-court seconds and lineup seconds based on time delta."""
        current_lineup_tuple = None
        if delta_secs > 0:
            for p in on_court:
                self.on_secs[p] += delta_secs
            if len(on_court) == 5:
                lineup_tuple = tuple(sorted(on_court))
                lu_stats = self.lineup_stats[lineup_tuple]
                lu_stats["secs"] = lu_stats.get("secs", 0) + delta_secs
                current_lineup_tuple = lineup_tuple
        return current_lineup_tuple

    def _handle_score(
        self,
        pts: int,
        is_target_team_event: bool,
        on_court: Set[str],
        per_game_plusminus: Dict[str, int],
        per_game_on_pts_a: Dict[str, int],
        current_lineup_tuple: Optional[tuple],
        game_opponent_pts: int,
    ) -> int:
        """Updates plus/minus, on/off points, lineup points, and opponent points for a scoring event."""
        point_diff = pts if is_target_team_event else -pts
        for p in on_court:
            self.plus_minus[p] += point_diff
            per_game_plusminus[p] += point_diff

        if is_target_team_event:
            self.team_pts_f += pts
            for p in on_court:
                self.on_pts_f[p] += pts
        else:
            self.team_pts_a += pts
            game_opponent_pts += pts
            for p in on_court:
                self.on_pts_a[p] += pts
                per_game_on_pts_a[p] += pts

        if current_lineup_tuple:
            lu_stats = self.lineup_stats[current_lineup_tuple]
            if is_target_team_event:
                lu_stats["pts_for"] = lu_stats.get("pts_for", 0) + pts
            else:
                lu_stats["pts_against"] = lu_stats.get("pts_against", 0) + pts

        return game_opponent_pts  # Return updated opponent points

    def _handle_player_event(
        self,
        event: dict,
        actor_name: str,
        event_abs_seconds: float,
        player_state: Dict[str, Dict],
        on_court: Set[str],
        per_game_minutes: Dict[str, float],
        per_game_points: Dict[str, int],
    ) -> None:
        """Handles player-specific stats (points, fouls) and substitutions."""
        event_type = event.get("move", "")
        actor_num = event.get("actorShirtNumber")

        # Ensure player is tracked
        if actor_name not in self.players:
            self.players[actor_name] = PlayerAggregate(number=actor_num)
        elif self.players[actor_name].number is None and actor_num is not None:
            self.players[actor_name].number = actor_num

        pa = self.players[actor_name]
        st = player_state.setdefault(
            actor_name, {"status": "out", "since": 0.0, "number": actor_num}
        )

        # --- Handle Player Stats ---
        if event_type in ["Cistella de 3", "Triple"]:
            pa.t3 += 1
            pa.pts += 3
            per_game_points[actor_name] += 3
        elif event_type in ["Cistella de 2", "Esmaixada"]:
            pa.t2 += 1
            pa.pts += 2
            per_game_points[actor_name] += 2
        elif event_type == "Cistella de 1":
            pa.t1 += 1
            pa.pts += 1
            per_game_points[actor_name] += 1
        elif event_type.startswith("Personal"):  # Corrected check
            pa.fouls += 1

        # --- Handle Substitutions ---
        if event_type == "Entra al camp":
            if st["status"] == "out":
                st["status"] = "in"
                st["since"] = event_abs_seconds
                for other_player in on_court:  # Update pairwise for new player
                    self._credit_pairwise_secs(actor_name, other_player, 0)
                on_court.add(actor_name)
        elif event_type == "Surt del camp":
            if st["status"] == "in":
                duration = event_abs_seconds - st["since"]
                self._credit_minutes(actor_name, duration)
                per_game_minutes[actor_name] += duration
                on_court.remove(actor_name)
                for other_player in on_court:  # Update pairwise for leaving player
                    self._credit_pairwise_secs(actor_name, other_player, duration)
                st["status"] = "out"
                st["since"] = event_abs_seconds

    def _finalize_game_metrics(
        self,
        game_end_sec: float,
        last_event_abs_seconds: float,
        on_court: Set[str],
        player_state: Dict[str, Dict],
        per_game_minutes: Dict[str, float],
        per_game_points: Dict[str, int],
        per_game_plusminus: Dict[str, int],
        per_game_on_pts_a: Dict[str, int],
        game_opponent_pts: int,
        match_id: str,
    ) -> None:
        """Calculates remaining metrics at game end and logs summary/game data."""
        # Flush remaining On/Off/Lineup seconds
        remaining_secs = game_end_sec - last_event_abs_seconds
        if remaining_secs > 0:
            for p in on_court:
                self.on_secs[p] += remaining_secs
            if len(on_court) == 5:
                lineup_tuple = tuple(sorted(on_court))
                self.lineup_stats[lineup_tuple]["secs"] += remaining_secs

        # Credit remaining time for players still on court
        for p, st in player_state.items():
            if st["status"] == "in":
                duration = game_end_sec - st["since"]
                self._credit_minutes(p, duration)
                per_game_minutes[p] += duration
                temp_on_court = on_court.copy()
                temp_on_court.remove(p)
                for other_player in temp_on_court:
                    self._credit_pairwise_secs(p, other_player, duration)

        # Set GP to 1 for participating players (single match report context)
        for player_name in player_state:
            if player_name in self.players:
                self.players[player_name].gp = 1

        # Save game summary (opponent points)
        game_idx = 1  # Always 1 for single match report
        if player_state and match_id not in self.processed_matches:
            self.game_summaries.append(
                {"game_idx": game_idx, "opponent_pts": game_opponent_pts}
            )
            self.processed_matches.add(match_id)

        # Save per-player game log entry
        for pname, secs in per_game_minutes.items():
            if pname in self.players:
                self.game_log.append(
                    {
                        "game_idx": game_idx,
                        "player": pname,
                        "mins": round(secs / 60, 1),
                        "pts": per_game_points.get(pname, 0),
                        "+/-": per_game_plusminus.get(pname, 0),
                        "drtg": (
                            round(
                                (per_game_on_pts_a.get(pname, 0) * 40 / (secs / 60)), 1
                            )
                            if secs >= 60
                            else None
                        ),
                    }
                )

    # ==================================================================
    # Table Generation & Accessors (Mostly unchanged)
    # ==================================================================
    # def evolution_table(...) - Removed as not applicable

    def lineup_table(
        self,
        min_usage: float = 0.0,
        names_to_exclude: set[str] = set(),  # Lower default usage for single game
    ) -> pd.DataFrame:
        """Generates a DataFrame summarizing lineup performance.

        Calculates minutes played, usage percentage (relative to total valid lineup time),
        and Net Rating (per 40 min) for each 5-player lineup that played.
        Filters out lineups with less than min_usage and those containing excluded players.

        Args:
            min_usage: Minimum usage percentage (0.0 to 1.0) required for a lineup to be included.
                       Defaults to 0.0 for single-match context.
            names_to_exclude: A set of player names to exclude from calculations.

        Returns:
            pandas.DataFrame: Sorted by NetRtg (desc), including lineup, mins, usage_%, NetRtg.
        """
        # Calculate total seconds across all tracked lineups with positive time
        # Filter lineups *before* calculating total seconds for usage %
        valid_lineups = {
            lu: v
            for lu, v in self.lineup_stats.items()
            if v.get("secs", 0) > 0 and not any(p in names_to_exclude for p in lu)
        }

        total_lineup_secs = sum(stats["secs"] for stats in valid_lineups.values())
        if total_lineup_secs == 0:
            logger.info("No lineup data with positive time found after exclusions.")
            return pd.DataFrame()

        records = []
        # Iterate through the pre-filtered valid lineups
        for lu, v in valid_lineups.items():
            # Calculate usage % relative to total *valid* lineup seconds
            usage_percent = (
                (v["secs"] / total_lineup_secs) * 100 if total_lineup_secs > 0 else 0
            )

            # Filter based on the calculated percentage (min_usage is 0.0 to 1.0)
            if (usage_percent / 100.0) < min_usage:
                continue

            # Original NetRtg calculation
            net = (
                (v.get("pts_for", 0) - v.get("pts_against", 0))
                * 40
                / (v.get("secs", 0) / 60)
                if v.get("secs", 0) > 0
                else 0
            )

            records.append(
                {
                    "lineup": " - ".join(map(shorten_name, lu)),
                    "mins": round(v["secs"] / 60, 1),
                    "usage_%": round(usage_percent, 1),
                    "NetRtg": round(net, 1),
                }
            )

        df = pd.DataFrame(records)
        # Sort by NetRtg by default, but ensure columns exist
        if df.empty or "NetRtg" not in df.columns:
            return df  # nothing to sort
        return df.sort_values("NetRtg", ascending=False).reset_index(drop=True)

    def _credit_minutes(self, player: str, secs: float) -> None:
        pa = self.players[player]
        pa.merge_secs(secs)

    def _credit_pairwise_secs(self, p1: str, p2: str, secs: float) -> None:
        """Adds played time to the pairwise matrix (for p1<->p2)."""
        self.pairwise_secs[p1][p2] += secs
        if p1 != p2:  # Avoid double-counting self-time
            self.pairwise_secs[p2][p1] += secs

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def player_table(self, names_to_exclude: set[str] = set()) -> pd.DataFrame:
        """Generates a DataFrame summarizing aggregate player statistics.

        Args:
            names_to_exclude: A set of player names to exclude from the table.

        Returns:
            pandas.DataFrame: Player stats (Player, #, Mins, PTS, T3, T2, T1, Fouls, +/-),
                              sorted by PTS descending.
        """
        data = {
            "Player": [],
            "#": [],
            "Mins": [],
            "PTS": [],
            "T3": [],
            "T2": [],
            "T1": [],
            "Fouls": [],
            "+/-": [],
        }
        for name, aggr in sorted(
            self.players.items(), key=lambda it: it[1].pts, reverse=True
        ):
            if name not in names_to_exclude:
                data["Player"].append(shorten_name(name))
                data["#"].append(aggr.number)
                data["Mins"].append(aggr.minutes)
                data["PTS"].append(aggr.pts)
                data["T3"].append(aggr.t3)
                data["T2"].append(aggr.t2)
                data["T1"].append(aggr.t1)
                data["Fouls"].append(aggr.fouls)
                data["+/-"].append(self.plus_minus.get(name, 0))
        return pd.DataFrame(data)

    def pairwise_minutes(self, names_to_exclude: set[str] = set()) -> pd.DataFrame:
        """Generates a matrix DataFrame showing total minutes pairs of players played together.

        Args:
            names_to_exclude: A set of player names to exclude from the matrix.

        Returns:
            pandas.DataFrame: A square matrix where index/columns are shortened player names
                              (sorted by total minutes played) and values are total integer
                              minutes played together.
        """
        # Get players sorted by total minutes played (descending)
        # Filter players *before* sorting and creating matrix
        valid_players = {
            name: data
            for name, data in self.players.items()
            if name not in names_to_exclude
        }
        if not valid_players:
            return pd.DataFrame()  # Return empty if no players left

        sorted_players_by_minutes = sorted(
            valid_players.items(),
            key=lambda item: item[1].minutes,
            reverse=True,
        )
        sorted_names = [name for name, _ in sorted_players_by_minutes]

        # Create matrix using the sorted names of *valid* players
        matrix = pd.DataFrame(index=sorted_names, columns=sorted_names, dtype=int)
        for p1 in sorted_names:
            for p2 in sorted_names:
                matrix.loc[p1, p2] = int(round(self.pairwise_secs[p1].get(p2, 0) / 60))

        # Rename index and columns to shortened names
        matrix.index = matrix.index.map(shorten_name)
        matrix.columns = matrix.columns.map(shorten_name)

        # Ensure all values are integers before returning
        return matrix.astype(int)

    # ------------------------------------------------------------------
    # On/Off Net Rating table
    # ------------------------------------------------------------------
    def on_off_table(self, names_to_exclude: set[str] = set()) -> pd.DataFrame:
        rows = []
        # Filter on_secs before summing total_secs and iterating
        valid_on_secs = {
            p: s for p, s in self.on_secs.items() if p not in names_to_exclude
        }
        # Calculate total game seconds from the maximum player on_secs if available
        # This assumes at least one player was on court for the whole game duration processed
        total_game_secs = max(valid_on_secs.values()) if valid_on_secs else 0

        if total_game_secs == 0:
            # Fallback: estimate from game log if no valid_on_secs
            if self.game_log:
                total_game_secs = max(row["mins"] for row in self.game_log) * 60
            else:
                logger.warning(
                    "Could not determine total game seconds for On/Off calculation."
                )
                return pd.DataFrame()  # no data or way to calc off

        # Iterate through filtered players
        for player, secs_on in valid_on_secs.items():
            mins_on = secs_on / 60
            if mins_on < 0.01:  # Use 2-decimal precision threshold
                continue  # skip very small samples

            mins_off = (total_game_secs - secs_on) / 60
            on_net = (
                (self.on_pts_f[player] - self.on_pts_a[player]) * 40 / mins_on
                if mins_on > 0
                else 0
            )

            if mins_off >= 0.01:  # Use 2-decimal precision threshold
                off_pts_f = self.team_pts_f - self.on_pts_f[player]
                off_pts_a = self.team_pts_a - self.on_pts_a[player]
                # Protect division by zero if mins_off is effectively zero
                off_net = (off_pts_f - off_pts_a) * 40 / mins_off if mins_off > 0 else 0
                diff = on_net - off_net
            else:
                off_net = None
                diff = None

            rows.append(
                {
                    "Player": shorten_name(player),
                    "Mins_ON": round(mins_on, 2),
                    "On_Net": round(on_net, 1),
                    "Off_Net": round(off_net, 1) if off_net is not None else "—",
                    "ON-OFF": round(diff, 1) if diff is not None else "—",
                }
            )
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Ensure ON-OFF is numeric for sorting, coercing errors to NaN
        df["ON-OFF"] = pd.to_numeric(df["ON-OFF"], errors="coerce")

        return df.sort_values(
            "ON-OFF", ascending=False, na_position="last"
        ).reset_index(drop=True)
