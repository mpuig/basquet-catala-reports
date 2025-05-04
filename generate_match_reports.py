"""
Generate Individual Match HTML Reports
-------------------------------------

Generates detailed HTML reports for individual basketball matches based on data
downloaded by `fetch_data.py`. For each match involving the specified target team
found in the provided group schedules, this script:

1.  Calculates per-match statistics (Player aggregates, On/Off, Lineups) for the target team.
2.  Generates a markdown summary (`summary.md`) including team stat comparison.
3.  (Optional) Generates an AI-powered narrative summary using an LLM (`llm_summary.md`, cached).
4.  Generates plots (PNG): Score timeline, Pairwise minutes heatmap, Player On/Off chart, Lineup NetRtg chart.
5.  Combines the summaries, tables, and plots into a final HTML report (`report.html`).
6.  Creates an `index.html` file listing all generated reports, sorted by date.

Dependencies:
-------------
*   Python 3.10+
*   pandas >= 2.0
*   matplotlib
*   seaborn
*   numpy
*   jinja2
*   markdown2
*   (Optional) litellm (and an API key for the chosen LLM, e.g., OPENAI_API_KEY)

CLI Examples:
-------------
1.  Generate reports for Team 69630 in groups 17182 and 18299:
    $ python generate_match_reports.py --team 69630 --groups 17182 18299

2.  Specify data and output directories:
    $ python generate_match_reports.py --team 69630 --groups 17182 --data-dir ../data --output-dir ../output/reports

3.  Use a different season identifier (affects official stats link):
    $ python generate_match_reports.py --team 69630 --groups 17182 --season 2023

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Any, Optional

import litellm
import markdown2  # For converting summary.md to HTML
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Environment, select_autoescape, Template

################################################################################
# Configuration & logging                                                       #
################################################################################

# --- Game Settings ---
PERIOD_LENGTH_SEC: int = 600  # 10‑minute quarters in Catalan U13 competition

# --- Plotting Settings ---
PLOT_DPI = 100
SCORE_TIMELINE_COLOR_LOCAL = "#003366"
SCORE_TIMELINE_COLOR_VISITOR = "#7FFFD4"
PAIRWISE_CMAP = "Reds"
ON_NET_PALETTE = "coolwarm"
LINEUP_PALETTE = "viridis"

# --- LLM Settings ---
LLM_MODEL = "gpt-4.1"
LLM_TEMPERATURE = 0.5
LLM_MAX_TOKENS = 250

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,  # Default to INFO, can be overridden if needed
    format="%(levelname)s | %(name)s | %(asctime)s | %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("match_report_generator")


################################################################################
# Utility functions                                                             #
################################################################################


def get_absolute_seconds(
    period: int, minute: int, second: int, *, period_len: int = PERIOD_LENGTH_SEC
) -> int:
    """Convert (period, remaining mm:ss) → absolute seconds since game start."""
    return (period - 1) * period_len + (period_len - (minute * 60 + second))


def shorten_name(full_name: str) -> str:
    """Removes the last part of a name (e.g., 'FIRST MIDDLE LAST' -> 'FIRST MIDDLE')."""
    parts = full_name.split()
    if len(parts) > 1:  # Only modify if there's more than one part
        return " ".join(parts[:-1])  # Join all parts except the last one
    return full_name  # Return original if it's just one word or empty


################################################################################
# Dataclasses                                                                   #
################################################################################


@dataclass
class PlayerAggregate:
    """Holds aggregated statistics for a single player."""

    number: str = "??"
    gp: int = 0
    secs_played: float = 0.0
    pts: int = 0
    t3: int = 0
    t2: int = 0
    t1: int = 0
    fouls: int = 0

    def merge_secs(self, secs: float) -> None:
        self.secs_played += secs

    @property
    def minutes(self) -> int:
        return round(self.secs_played / 60)


################################################################################
# IO helpers                                                                    #
################################################################################


def load_schedule(path: Path) -> pd.DataFrame:
    """Read schedule CSV → DataFrame ensuring 'Match ID' exists."""
    if not path.exists():
        raise FileNotFoundError(f"Schedule file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    if "match_id" not in df.columns:
        logger.error(
            "Could not find 'match_id' column. Columns found: %s", df.columns.tolist()
        )
        raise ValueError("CSV missing required column 'match_id' or equivalent")
    return df


def load_match_moves(match_id: str, moves_dir: Path) -> List[dict]:
    """Return list of event dicts for a match or empty list if not found/invalid."""
    file_path = moves_dir / f"{match_id}.json"
    if not file_path.exists():
        logger.warning("Match moves JSON not found for %s", match_id)
        return []
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Could not decode JSON for %s – %s", match_id, exc)
        return []


def load_team_stats(team_id: str, season: str, team_stats_dir: Path) -> dict | None:
    """Loads team statistics JSON for a given team and season."""
    file_path = team_stats_dir / f"team_{team_id}_season_{season}.json"
    if not file_path.exists():
        logger.warning("Team stats file not found: %s", file_path)
        return None
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning(
            "Could not decode team stats JSON for %s/%s: %s", team_id, season, exc
        )
        return None
    except Exception as e:
        logger.error("Error reading team stats file %s: %s", file_path, e)
        return None


def load_match_stats(match_id: str, stats_dir: Path) -> dict | None:
    """Loads and parses the aggregated match stats JSON data."""
    if not match_id or not isinstance(match_id, str) or len(match_id) < 5:
        logger.debug("Skipping load_match_stats for invalid match_id: %s", match_id)
        return None

    json_filepath = stats_dir / f"{match_id}.json"
    if not json_filepath.exists():
        logger.warning("Aggregated Match Stats JSON file not found: %s", json_filepath)
        return None

    try:
        with open(json_filepath, "r", encoding="utf-8") as f:
            stats_data = json.load(f)
            return stats_data
    except json.JSONDecodeError:
        logger.warning("Could not decode Match Stats JSON: %s", json_filepath)
        return None
    except Exception as e:
        logger.error("Error reading Match Stats JSON file %s: %s", json_filepath, e)
        return None


################################################################################
# Core calculator (Copied directly from process_data.py)                        #
# We will instantiate this per match for the target team.                      #
################################################################################


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
        all_stats: Dict[str, dict],  # Add match stats data
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

            if not events:
                skipped_matches_no_moves += 1
                continue
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
        target_team_schedule_id = self.team_id
        match_stats_teams = stats_data.get("teams", [])
        internal_id = None
        team_name = "Unknown"

        for team_info in match_stats_teams:
            if str(team_info.get("teamIdExtern")) == target_team_schedule_id:
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
                    "Mins_ON": round(mins_on, 2),  # Changed rounding to 2 decimal places
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


################################################################################
# Plotting Functions (Adapted for single match reports)                         #
################################################################################


def plot_score_timeline(
    match_moves: List[dict],
    match_stats: dict,
    match_output_dir: Path,
    target_team_id: str,
) -> Optional[str]:
    """Generates a line plot showing the score progression over time."""
    if not match_moves:
        return None

    times = []
    local_scores = []
    visitor_scores = []
    current_local_score = 0
    current_visitor_score = 0
    # Determine internal IDs
    local_internal_id = None
    visitor_internal_id = None
    for team_info in match_stats.get("teams", []):
        if str(team_info.get("teamIdExtern")) == target_team_id:
            # This logic assumes target team is one of the two teams in match_stats
            # We need to figure out if target was local or visitor *in this match*
            # Let's find both internal IDs first
            pass  # Need a robust way to map local/visitor schedule ID to internal ID

    # Simplified: Assume team 0 in match_stats is local, team 1 is visitor (based on structure)
    # This might be fragile if the order isn't guaranteed.
    if len(match_stats.get("teams", [])) == 2:
        local_internal_id = match_stats["teams"][0].get("teamIdIntern")
        visitor_internal_id = match_stats["teams"][1].get("teamIdIntern")
    else:
        logger.warning("Could not determine internal IDs for score timeline plot.")
        return None

    if not local_internal_id or not visitor_internal_id:
        logger.warning("Internal IDs missing for score timeline plot.")
        return None

    # Get team names from match_stats for legend
    local_name = match_stats["teams"][0].get("name", "Local")
    visitor_name = match_stats["teams"][1].get("name", "Visitor")

    times.append(0)
    local_scores.append(0)
    visitor_scores.append(0)

    points_map = {"Cistella de 1": 1, "Cistella de 2": 2, "Cistella de 3": 3}

    for event in sorted(
        match_moves,
        key=lambda x: get_absolute_seconds(
            x.get("period", 0), x.get("min", 0), x.get("sec", 0)
        ),
    ):
        event_time = get_absolute_seconds(
            event.get("period", 0), event.get("min", 0), event.get("sec", 0)
        )
        pts = points_map.get(event.get("move", ""), 0)
        if pts > 0:
            event_team_id = event.get("idTeam")
            if event_team_id == local_internal_id:
                current_local_score += pts
            elif event_team_id == visitor_internal_id:
                current_visitor_score += pts
            else:
                continue  # Skip points from unknown team IDs if any

            times.append(event_time / 60)  # Convert to minutes
            local_scores.append(current_local_score)
            visitor_scores.append(current_visitor_score)

    filename = match_output_dir / "score_timeline.png"
    relative_path = filename.name

    try:
        plt.figure(figsize=(10, 5))
        plt.plot(
            times, local_scores, label=f"{local_name}", color=SCORE_TIMELINE_COLOR_LOCAL
        )
        plt.plot(
            times,
            visitor_scores,
            label=f"{visitor_name}",
            color=SCORE_TIMELINE_COLOR_VISITOR,
        )
        plt.xlabel("Time (Minutes)")
        plt.ylabel("Score")
        plt.title("Score Progression")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(filename, dpi=PLOT_DPI)
        plt.close()
        logger.debug(f"Saved score timeline plot: {filename}")
        return relative_path
    except Exception as e:
        logger.error(f"Failed to generate score timeline plot: {e}")
        plt.close()  # Attempt to close figure on error
        return None


def plot_pairwise_heatmap(
    pairwise_df: pd.DataFrame, match_output_dir: Path
) -> Optional[str]:
    """Generates and saves a heatmap for pairwise minutes for a single match."""
    if pairwise_df.empty:
        logger.info("Pairwise DataFrame is empty, skipping heatmap.")
        return None
    if pairwise_df.shape[0] < 2 or pairwise_df.shape[1] < 2:
        logger.info("Insufficient players for pairwise heatmap.")
        return None

    filename = match_output_dir / "pairwise_heatmap.png"
    relative_path = filename.name

    try:
        plt.figure(
            figsize=(
                max(6, pairwise_df.shape[1] * 0.8),
                max(4, pairwise_df.shape[0] * 0.6),
            )
        )  # Dynamic size
        sns.heatmap(
            pairwise_df, annot=True, fmt="d", cmap=PAIRWISE_CMAP, linewidths=0.5
        )
        plt.title("Pairwise Minutes Played Together")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        # Removed figtext description for single match context
        plt.tight_layout()
        plt.savefig(filename, dpi=PLOT_DPI)
        plt.close()
        logger.debug(f"Saved pairwise heatmap: {filename}")
        return relative_path
    except Exception as e:
        logger.error(f"Failed to generate/save pairwise heatmap: {e}")
        plt.close()  # Attempt to close figure on error
        return None


def plot_player_on_net(onoff_df: pd.DataFrame, match_output_dir: Path) -> Optional[str]:
    """Generates and saves a bar chart for Player On_Net rating for a single match."""
    if onoff_df.empty or "On_Net" not in onoff_df.columns:
        logger.info("On/Off DataFrame is empty or missing 'On_Net', skipping plot.")
        return None

    # Ensure 'On_Net' is numeric, coercing errors
    onoff_df["On_Net"] = pd.to_numeric(onoff_df["On_Net"], errors="coerce")
    onoff_df = onoff_df.dropna(subset=["On_Net"])

    if onoff_df.empty:
        logger.info("No valid numeric 'On_Net' data found, skipping plot.")
        return None

    # Sort by On_Net for plotting
    onoff_df_sorted = onoff_df.sort_values("On_Net", ascending=False)

    filename = match_output_dir / "on_net_chart.png"
    relative_path = filename.name

    try:
        plt.figure(figsize=(max(8, len(onoff_df_sorted) * 0.6), 5))  # Dynamic width
        barplot = sns.barplot(
            x="Player",
            y="On_Net",
            hue="Player",  # Use hue for consistency, but legend=False
            data=onoff_df_sorted,
            palette=ON_NET_PALETTE,
            legend=False,
        )
        # Add value labels on bars - adjust text positioning slightly
        for i, row in onoff_df_sorted.iterrows():
            value = row.On_Net
            pos = i  # Use index for bar position
            barplot.text(
                pos,
                value + (np.sign(value) * 1 if value != 0 else 1),  # Adjust offset
                f"{value:.1f}",
                color="black",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )

        plt.title("Player On-Court Net Rating (per 40 min)")
        plt.xlabel("Player")
        plt.ylabel("On_Net Rating")
        plt.xticks(rotation=45, ha="right")
        # Removed figtext description
        plt.tight_layout()
        plt.savefig(filename, dpi=PLOT_DPI)
        plt.close()
        logger.debug(f"Saved player On_Net plot: {filename}")
        return relative_path
    except Exception as e:
        logger.error(f"Failed to generate/save player On_Net plot: {e}")
        plt.close()
        return None


def plot_lineup_netrtg(
    lineup_df: pd.DataFrame, match_output_dir: Path
) -> Optional[str]:
    """Generates and saves a bar chart for Lineup NetRtg for a single match."""
    if lineup_df.empty or "NetRtg" not in lineup_df.columns:
        logger.info("Lineup DataFrame empty or missing 'NetRtg', skipping plot.")
        return None

    # Ensure 'NetRtg' is numeric
    lineup_df["NetRtg"] = pd.to_numeric(lineup_df["NetRtg"], errors="coerce")
    lineup_df = lineup_df.dropna(subset=["NetRtg"])

    if lineup_df.empty:
        logger.info("No valid numeric 'NetRtg' data for lineups, skipping plot.")
        return None

    top_n = 5  # Changed from 10 to 5
    # Make a copy before modifying for plotting
    lineup_df_plot = lineup_df.sort_values("NetRtg", ascending=False).head(top_n).copy()

    # Replace hyphens with newlines *specifically for plot labels*
    lineup_df_plot['lineup'] = lineup_df_plot['lineup'].str.replace(' - ', '\n', regex=False)

    if lineup_df_plot.empty:  # Check again after head(top_n)
        logger.info("No lineups found after filtering top N, skipping plot.")
        return None

    filename = match_output_dir / "lineup_chart.png"
    relative_path = filename.name

    try:
        plt.figure(figsize=(max(10, len(lineup_df_plot) * 1.2), 6))  # Dynamic width
        barplot = sns.barplot(
            x="lineup",  # This now has newlines
            y="NetRtg",
            hue="lineup",  # Use hue for consistency, but legend=False
            data=lineup_df_plot,
            palette=LINEUP_PALETTE,
            legend=False,
        )
        # Add value labels
        for i, row in lineup_df_plot.iterrows():  # Use plot-specific df
            value = row.NetRtg
            pos = lineup_df_plot.index.get_loc(i)  # Get numerical position
            barplot.text(
                pos,
                value + (np.sign(value) * 1 if value != 0 else 1),
                f"{value:.1f}",
                color="black",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )
        plt.title(f"Top {len(lineup_df_plot)} Lineup Net Rating (per 40 min)")
        plt.xlabel("Lineup")
        plt.ylabel("Net Rating")
        plt.xticks(rotation=0, ha="center")
        # Removed figtext
        plt.tight_layout()
        plt.savefig(filename, dpi=PLOT_DPI)
        logger.debug(f"Saved Lineup NetRtg Bar Chart: {filename}")
        return relative_path
    except Exception as e:
        logger.error(f"Failed to generate/save lineup NetRtg plot: {e}")
        return None
    finally:
        plt.close()  # Ensure plot is closed


################################################################################
# Report Generation Functions                                                  #
################################################################################


def generate_summary_md(
    match_info: pd.Series, match_stats: dict, target_team_id: str
) -> str:
    """Generates a markdown summary of the match."""
    # Extract basic info from schedule row
    local_name = match_info.get("local_team", "Local")
    visitor_name = match_info.get("visitor_team", "Visitor")
    local_id = str(match_info.get("local_team_id", ""))
    match_id = match_info.get("match_id", "Unknown")

    is_target_local = target_team_id == local_id
    target_team_name = local_name if is_target_local else visitor_name
    opponent_team_name = visitor_name if is_target_local else local_name

    summary_lines = []

    if "teams" in match_stats and len(match_stats["teams"]) == 2:
        team1_stats = match_stats["teams"][0]
        team2_stats = match_stats["teams"][1]

        # Ensure we correctly identify target and opponent stats within match_stats
        target_stats = None
        opponent_stats = None
        if str(team1_stats.get("teamIdExtern")) == target_team_id:
            target_stats = team1_stats
            opponent_stats = team2_stats
        elif str(team2_stats.get("teamIdExtern")) == target_team_id:
            target_stats = team2_stats
            opponent_stats = team1_stats

        if target_stats and opponent_stats:
            summary_lines.append("## Team Stat Comparison")
            summary_lines.append(
                f"| Stat         | {target_team_name} | {opponent_team_name} |"
            )
            summary_lines.append(
                "|:-------------|-------------------:|-------------------:|"
            )

            # Example stats - add more as needed from match_stats structure
            # Access nested 'data' dictionary safely with .get('data', {})
            ts_data = target_stats.get("data", {})
            os_data = opponent_stats.get("data", {})

            summary_lines.append(
                f"| Points       | {ts_data.get('score', '?')} | {os_data.get('score', '?')} |"
            )
            summary_lines.append(
                f"| T2 Made/Att  | {ts_data.get('shotsOfTwoSuccessful', '?')}/{ts_data.get('shotsOfTwoAttempted', '?')} | {os_data.get('shotsOfTwoSuccessful', '?')}/{os_data.get('shotsOfTwoAttempted', '?')} |"
            )
            summary_lines.append(
                f"| T3 Made/Att  | {ts_data.get('shotsOfThreeSuccessful', '?')}/{ts_data.get('shotsOfThreeAttempted', '?')} | {os_data.get('shotsOfThreeSuccessful', '?')}/{os_data.get('shotsOfThreeAttempted', '?')} |"
            )
            summary_lines.append(
                f"| T1 Made/Att  | {ts_data.get('shotsOfOneSuccessful', '?')}/{ts_data.get('shotsOfOneAttempted', '?')} | {os_data.get('shotsOfOneSuccessful', '?')}/{os_data.get('shotsOfOneAttempted', '?')} |"
            )
            summary_lines.append(
                f"| Fouls        | {ts_data.get('faults', '?')} | {os_data.get('faults', '?')} |"
            )
            summary_lines.append("")
        else:
            logger.warning(
                f"Could not properly match teams in match_stats for {match_id}"
            )

    return "\n".join(summary_lines)


def generate_llm_summary(
    match_info: pd.Series,
    match_stats: dict,
    target_team_id: str,
    player_stats_df: pd.DataFrame,
) -> Optional[str]:
    """Generates a narrative match summary using an LLM via litellm."""
    logger.info("Attempting to generate LLM summary...")

    # --- Prepare data for prompt ---
    local_name = match_info.get("local_team", "Local")
    visitor_name = match_info.get("visitor_team", "Visitor")
    local_id = str(match_info.get("local_team_id", ""))
    is_target_local = target_team_id == local_id
    target_team_name = local_name if is_target_local else visitor_name
    opponent_team_name = visitor_name if is_target_local else local_name
    score = match_info.get("score", "-")
    match_date_time = match_info.get("date_time", "Unknown")

    # Team stats comparison (extract from generate_summary_md logic)
    team_stats_summary = "(Team stats comparison not available)"
    if "teams" in match_stats and len(match_stats["teams"]) == 2:
        team1_stats = match_stats["teams"][0]
        team2_stats = match_stats["teams"][1]
        target_stats = None
        opponent_stats = None
        if str(team1_stats.get("teamIdExtern")) == target_team_id:
            target_stats = team1_stats
            opponent_stats = team2_stats
        elif str(team2_stats.get("teamIdExtern")) == target_team_id:
            target_stats = team2_stats
            opponent_stats = team1_stats

        if target_stats and opponent_stats:
            ts_data = target_stats.get("data", {})
            os_data = opponent_stats.get("data", {})
            team_stats_lines = [
                f"- Points: {ts_data.get('score', '?')} vs {os_data.get('score', '?')}",
                f"- T2: {ts_data.get('shotsOfTwoSuccessful', '?')}/{ts_data.get('shotsOfTwoAttempted', '?')} vs {os_data.get('shotsOfTwoSuccessful', '?')}",
                f"- T3: {ts_data.get('shotsOfThreeSuccessful', '?')}/{ts_data.get('shotsOfThreeAttempted', '?')} vs {os_data.get('shotsOfThreeSuccessful', '?')}",
                f"- T1: {ts_data.get('shotsOfOneSuccessful', '?')}/{ts_data.get('shotsOfOneAttempted', '?')} vs {os_data.get('shotsOfOneSuccessful', '?')}",
                f"- Fouls: {ts_data.get('faults', '?')} vs {os_data.get('faults', '?')}",
            ]
            team_stats_summary = "\n".join(team_stats_lines)

    # Top player stats (e.g., top 3 scorers for target team)
    top_players_summary = "(No player stats available)"
    if not player_stats_df.empty:
        top_scorers = player_stats_df.nlargest(3, "PTS")[["Player", "PTS"]]
        top_players_lines = [
            f"- {row['Player']}: {row['PTS']} PTS" for _, row in top_scorers.iterrows()
        ]
        top_players_summary = "\n".join(top_players_lines)

    # Optional momentum data – inject if available
    momentum_section = ""
    if momentum_info := match_stats.get("momentum"):
        momentum_section = f"""
    Momentum Insights:
    - Lead changes: {momentum_info.get('lead_changes', '?')}
    - Ties: {momentum_info.get('ties', '?')}
    - Largest lead held: {momentum_info.get('max_lead', '?')} points
    """

    # Optional per-quarter scores
    quarters_section = ""
    if "quarters" in match_stats:
        quarters = match_stats["quarters"]
        quarters_str = " | ".join(
            f"Q{i + 1}: {q.get('home', '?')}-{q.get('away', '?')}"
            for i, q in enumerate(quarters)
        )
        quarters_section = f"\nScoring by Quarter:\n{quarters_str}\n"

    # Final Prompt
    prompt = f"""
    Generate a concise and informative 2-paragraph summary of a youth basketball match, focusing on the team "{target_team_name}".

    Match Context:
    - Date: {match_date_time}
    - Home: {local_name}
    - Opponent: {visitor_name}
    - Final Score: {score} ({local_name} vs {visitor_name})

    Team Comparison ({target_team_name} vs {opponent_team_name}):
    {team_stats_summary}

    Top Scorers ({target_team_name}):
    {top_players_summary}
    {quarters_section}
    {momentum_section}
    Instructions:
    - Use a neutral and analytical tone, like a coach's or media summary.
    - Emphasize "{target_team_name}" performance: scoring rhythm, defense, substitutions.
    - Mention key momentum phases (e.g., big runs, fouls, FT%, lead changes).
    - Highlight top scorer contributions in-game context.
    - Keep output to 2 short but informative paragraphs (~150 words total).
    """

    # --- Call LLM via litellm ---
    # Check for API key (assuming OpenAI for default)
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY environment variable not set. Skipping LLM summary."
        )
        return None

    try:
        # Use a common default model, consider making this configurable
        response = litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=LLM_MAX_TOKENS,  # Limit output length
            temperature=LLM_TEMPERATURE,  # Lower temperature for more factual summary
        )
        summary_text = response.choices[0].message.content.strip()
        logger.info("LLM summary generated successfully.")
        return summary_text
    except Exception as e:
        logger.error(f"Failed to generate LLM summary using litellm: {e}")
        return None


################################################################################
# HTML Template (Basic Placeholder)                                            #
################################################################################

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Match Report: {{ match_id }} - {{ local_name }} vs {{ visitor_name }}</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; margin-bottom: 20px; width: auto; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .summary { background-color: #eee; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .plot { margin-bottom: 30px; text-align: center; }
        .plot img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        .plot-caption { font-size: 0.9em; color: #555; margin-top: 5px; }
    </style>
</head>
<body>
    <h1>Match Report: {{ local_name }} vs {{ visitor_name }}</h1>
    <p><strong>Match ID:</strong> <a href="https://www.basquetcatala.cat/estadistiques/{{ season }}/{{ match_id }}" target="_blank" title="View official stats page">{{ match_id }}</a> | <strong>Date:</strong> {{ match_date }} | <strong>Group:</strong> {{ group_name }}</p>

    <div class="summary">
        {{ summary_html | safe }}
    </div>

    {% if llm_summary %}
    <h2>AI Generated Summary</h2>
    <div class="summary">
        <p>{{ llm_summary }}</p>
    </div>
    {% endif %}

    <h2>{{ target_team_name }} Statistics</h2>

    <h3>Player Aggregates</h3>
    {{ player_table_html | safe }}

    <h3>On/Off Net Rating</h3>
    {{ on_off_table_html | safe }}

    <h3>Top Lineups (by Net Rating)</h3>
    {{ lineup_table_html | safe }}

    <h2>Charts</h2>

    <!-- Placeholder for charts -->
    {% if score_timeline_path %}
    <div class="plot">
        <h3>Score Timeline</h3>
        <img src="{{ score_timeline_path }}" alt="Score Timeline">
        <p class="plot-caption">Score progression throughout the match.</p>
    </div>
    {% endif %}

    {% if pairwise_heatmap_path %}
    <div class="plot">
        <h3>Pairwise Minutes Heatmap</h3>
        <img src="{{ pairwise_heatmap_path }}" alt="Pairwise Minutes Heatmap">
         <p class="plot-caption">Minutes players spent on court together.</p>
   </div>
    {% endif %}

    {% if on_net_chart_path %}
    <div class="plot">
        <h3>Player On-Court Net Rating</h3>
        <img src="{{ on_net_chart_path }}" alt="Player On Net Rating Chart">
        <p class="plot-caption">Team point differential per 40 mins while player was on court.</p>
    </div>
    {% endif %}

     {% if lineup_chart_path %}
    <div class="plot">
        <h3>Top Lineup Net Rating</h3>
        <img src="{{ lineup_chart_path }}" alt="Lineup Net Rating Chart">
        <p class="plot-caption">Point differential per 40 mins for top lineups.</p>
    </div>
    {% endif %}


</body>
</html>
"""

INDEX_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Match Report Index</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        a { text-decoration: none; color: #0066cc; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Match Report Index (Team: {{ target_team_id }})</h1>

    {% if reports %}
    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th>Group</th>
                <th>Local</th>
                <th>Visitor</th>
                <th>Score</th>
                <th>Report Link</th>
            </tr>
        </thead>
        <tbody>
            {% for report in reports %} {# Use already sorted reports #}
            <tr>
                <td>{{ report.match_date }}</td>
                <td>{{ report.group_name }}</td>
                <td>{{ report.local_name }}</td>
                <td>{{ report.visitor_name }}</td>
                <td>{{ report.score }}</td>
                <td><a href="{{ report.report_path }}">{{ report.match_id }}</a></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p>No match reports were generated for Team {{ target_team_id }}.</p>
    {% endif %}

</body>
</html>
"""


################################################################################
# CLI entrypoint                                                               #
################################################################################


def _generate_single_report(
    match_info_row: pd.Series,
    target_team_id: str,
    gid: str,
    moves_dir: Path,
    stats_dir: Path,
    output_dir: Path,
    env: Environment,
    template: Template,
    args: argparse.Namespace,
) -> Optional[dict]:
    """Loads data, calculates stats, generates components, and renders HTML for ONE match."""
    match_id = str(match_info_row.get("match_id", ""))
    # Assume team ID conversion already happened before calling this function
    local_team_id = str(int(float(match_info_row.get("local_team_id", ""))))

    logger.info(f"Processing Match ID: {match_id} (Group: {gid})")

    # --- Create output directory for this match ---
    match_output_dir = output_dir / f"match_{match_id}"
    match_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data for this specific match ---
    match_moves = load_match_moves(match_id, moves_dir)
    match_stats = load_match_stats(match_id, stats_dir)

    if not match_moves:
        logger.warning(
            f"Missing moves JSON for match {match_id}. Cannot generate full report."
        )
        return None  # Indicate failure
    if not match_stats:
        logger.warning(
            f"Missing stats JSON for match {match_id}. Cannot generate full report."
        )
        return None  # Indicate failure

    # --- Calculate Stats for the target team in this match ---
    calc = StatsCalculator(target_team_id)
    single_match_schedule_df = pd.DataFrame([match_info_row])
    calc.process(
        single_match_schedule_df, {match_id: match_moves}, {match_id: match_stats}
    )

    # --- Generate Report Components ---

    # 1. Summary MD
    summary_content = generate_summary_md(match_info_row, match_stats, target_team_id)
    summary_md_path = match_output_dir / "summary.md"
    summary_md_path.write_text(summary_content, encoding="utf-8")
    summary_html = markdown2.markdown(summary_content, extras=["tables"])

    # 2. Stats Tables
    player_df = calc.player_table()
    onoff_df = calc.on_off_table()
    lineup_df = calc.lineup_table()
    player_table_html = (
        player_df.to_html(index=False, classes="stats-table", border=0)
        if not player_df.empty
        else "<p>(No player data)</p>"
    )
    on_off_table_html = (
        onoff_df.to_html(index=False, classes="stats-table", border=0)
        if not onoff_df.empty
        else "<p>(No On/Off data)</p>"
    )
    lineup_table_html = (
        lineup_df.head(5).to_html(index=False, classes="stats-table", border=0)
        if not lineup_df.empty
        else "<p>(No lineup data)</p>"
    )

    # 3. AI Summary (Cached)
    llm_summary_text = None
    llm_summary_md_path = match_output_dir / "llm_summary.md"
    if llm_summary_md_path.exists():
        try:
            llm_summary_text = llm_summary_md_path.read_text(encoding="utf-8").strip()
            if llm_summary_text:
                logger.info(f"Using cached LLM summary from: {llm_summary_md_path}")
            else:
                logger.warning(
                    f"Cached LLM summary file is empty: {llm_summary_md_path}. Will attempt regeneration."
                )
                llm_summary_text = None
        except Exception as e:
            logger.error(
                f"Error reading cached LLM summary {llm_summary_md_path}: {e}. Will attempt regeneration."
            )
            llm_summary_text = None

    if llm_summary_text is None:
        logger.info("LLM summary cache not found or invalid. Generating...")
        llm_summary_text = generate_llm_summary(
            match_info_row, match_stats, target_team_id, player_df
        )
        if llm_summary_text:
            try:
                llm_summary_md_path.write_text(llm_summary_text, encoding="utf-8")
                logger.info(
                    f"Saved newly generated LLM summary to: {llm_summary_md_path}"
                )
            except Exception as e:
                logger.error(
                    f"Error writing LLM summary cache {llm_summary_md_path}: {e}"
                )

    # 4. Charts
    score_timeline_path_rel = plot_score_timeline(
        match_moves, match_stats, match_output_dir, target_team_id
    )
    pairwise_heatmap_path_rel = plot_pairwise_heatmap(
        calc.pairwise_minutes(), match_output_dir
    )
    on_net_chart_path_rel = plot_player_on_net(onoff_df, match_output_dir)
    lineup_chart_path_rel = plot_lineup_netrtg(lineup_df, match_output_dir)

    # --- Get Context for Rendering ---
    local_name = match_info_row.get("local_team", "Local Team")
    visitor_name = match_info_row.get("visitor_team", "Visitor Team")
    target_team_name = local_name if target_team_id == local_team_id else visitor_name
    match_date = match_info_row.get("date_time", "Unknown Date")
    score = match_info_row.get("score", "-")
    group_name = f"Group {gid}"

    # --- Render HTML Report ---
    try:
        html_content = template.render(
            match_id=match_id,
            group_name=group_name,
            match_date=match_date,
            target_team_name=target_team_name,
            local_name=local_name,
            visitor_name=visitor_name,
            summary_html=summary_html,
            player_table_html=player_table_html,
            on_off_table_html=on_off_table_html,
            lineup_table_html=lineup_table_html,
            score_timeline_path=score_timeline_path_rel,
            pairwise_heatmap_path=pairwise_heatmap_path_rel,
            on_net_chart_path=on_net_chart_path_rel,
            lineup_chart_path=lineup_chart_path_rel,
            season=args.season,
            llm_summary=llm_summary_text,
        )
        report_html_path = match_output_dir / "report.html"
        report_html_path.write_text(html_content, encoding="utf-8")
        logger.info(f"-> Successfully generated report: {report_html_path}")
    except Exception as e:
        logger.error(f"Error rendering or writing HTML for match {match_id}: {e}")
        return None  # Indicate failure

    # --- Return data for index page ---
    return {
        "match_id": match_id,
        "local_name": local_name,
        "visitor_name": visitor_name,
        "match_date": match_date,
        "group_name": group_name,
        "score": score,
        "report_path": str(report_html_path.relative_to(output_dir)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate individual match HTML reports."
    )
    parser.add_argument("--team", required=True, help="Team ID to focus reports on")
    parser.add_argument(
        "--groups",
        nargs="+",
        required=True,
        help="Competition group IDs to scan for matches",
    )
    parser.add_argument(
        "--data-dir", default="data", help="Root data folder (CSV & JSON)"
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to save generated match reports",
    )
    parser.add_argument(
        "--season", default="2024", help="Season identifier (default: 2024)"
    )
    # parser.add_argument(
    #     "--exclude-players", # Maybe add later if needed for individual reports
    #     nargs='+',
    #     default=[],
    #     help="List of player UUIDs to exclude from reports",
    # )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    moves_dir = data_dir / "match_moves"
    stats_dir = data_dir / "match_stats"
    output_dir = Path(args.output_dir)
    target_team_id = args.team

    env = Environment(
        loader=None, autoescape=select_autoescape(["html", "xml"])
    )  # Loader=None as template is string
    template = env.from_string(HTML_TEMPLATE)

    logger.info(f"Starting report generation for Team ID: {target_team_id}")
    logger.info(f"Data Source: {data_dir.resolve()}")
    logger.info(f"Output Directory: {output_dir.resolve()}")

    processed_match_count = 0
    skipped_match_count = 0
    report_links_data = []  # Initialize list to store report link info

    for gid in args.groups:
        logger.info(f"--- Processing Group: {gid} ---")
        csv_path = data_dir / f"results_{gid}.csv"

        if not csv_path.exists():
            logger.warning(
                f"Schedule file not found: {csv_path}. Skipping group {gid}."
            )
            continue

        try:
            schedule_df = load_schedule(csv_path)
        except (FileNotFoundError, ValueError) as e:
            logger.error(
                f"Error loading schedule {csv_path}: {e}. Skipping group {gid}."
            )
            continue

        # Iterate through each match in the group schedule
        for index, match_info_row in schedule_df.iterrows():
            match_id = str(match_info_row.get("match_id", ""))

            # --- Robust Team ID Conversion ---
            try:
                # Attempt float -> int -> str conversion to strip ".0"
                local_team_id = str(int(float(match_info_row.get("local_team_id", ""))))
            except (ValueError, TypeError):
                # Fallback if ID is not numeric (e.g., 'Descansa')
                local_team_id = str(match_info_row.get("local_team_id", ""))

            try:
                visitor_team_id = str(
                    int(float(match_info_row.get("visitor_team_id", "")))
                )
            except (ValueError, TypeError):
                visitor_team_id = str(match_info_row.get("visitor_team_id", ""))
            # --- End Robust Team ID Conversion ---

            local_name = match_info_row.get("local_team", "?")
            visitor_name = match_info_row.get("visitor_team", "?")

            if not match_id or match_id.lower() == "nan":  # Also check for 'nan' string
                logger.warning(
                    "Skipping row with missing or invalid match_id in %s", csv_path
                )
                continue

            # --- Debugging Log ---
            logger.debug(
                f"Match {match_id}: Local={local_name} ({local_team_id}), Visitor={visitor_name} ({visitor_team_id}), Target={target_team_id}"
            )
            # --- End Debugging Log ---

            # Check if the target team played in this match
            if target_team_id != local_team_id and target_team_id != visitor_team_id:
                continue

            # Call the helper function to generate the report for this match
            report_data = _generate_single_report(
                match_info_row=match_info_row,
                target_team_id=target_team_id,
                gid=gid,
                moves_dir=moves_dir,
                stats_dir=stats_dir,
                output_dir=output_dir,
                env=env,
                template=template,
                args=args,
            )

            if report_data:
                processed_match_count += 1
                report_links_data.append(report_data)
            else:
                skipped_match_count += (
                    1  # Increment skipped if helper indicated failure
                )

        logger.info(f"--- Finished Group: {gid} ---")

    logger.info("Report generation complete.")
    logger.info(f"Successfully processed {processed_match_count} matches.")
    if skipped_match_count > 0:
        logger.warning(f"Skipped {skipped_match_count} matches due to missing data.")

    # --- Generate Index HTML ---
    if report_links_data:
        logger.info("Generating index.html...")
        try:
            index_template = env.from_string(INDEX_HTML_TEMPLATE)

            # Sort reports by date before rendering
            # Handle potential errors if date format is inconsistent or missing
            def sort_key(report):
                try:
                    # Attempt to parse date assuming DD-MM-YYYY HH:MM format
                    return pd.to_datetime(
                        report.get("match_date", "1900-01-01"),
                        format="%d-%m-%Y %H:%M",
                        errors="coerce",
                    )
                except ValueError:
                    # Fallback for unexpected formats or unparseable dates
                    return pd.Timestamp("1900-01-01")

            sorted_reports = sorted(report_links_data, key=sort_key)

            index_html_content = index_template.render(
                reports=sorted_reports,
                target_team_id=target_team_id,  # Pass team ID for title
            )
            index_html_path = output_dir / "index.html"
            index_html_path.write_text(index_html_content, encoding="utf-8")
            logger.info(f"Successfully generated index file: {index_html_path}")
        except Exception as e:
            logger.error(f"Failed to generate index.html: {e}")
            import traceback

            traceback.print_exc()  # Print stack trace for debugging Jinja errors
    else:
        logger.info(
            f"No reports were generated for Team {target_team_id}, skipping index.html creation."
        )


if __name__ == "__main__":
    main()
