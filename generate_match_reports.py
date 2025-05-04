"""
Generate Individual Match HTML Reports
-------------------------------------
• Python 3.10+
• Pandas ≥ 2.0
• Jinja2
• markdown2

Usage (CLI example)
-------------------
$ python generate_match_reports.py --team 69630 --groups 17182 18299 --data-dir data --output-dir reports

This script generates an HTML report for each match found in the specified group schedules
where the specified team played, including stats, summaries, and charts.
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
from jinja2 import Environment, select_autoescape

################################################################################
# Configuration & logging                                                       #
################################################################################

PERIOD_LENGTH_SEC: int = 600  # 10‑minute quarters in Catalan U13 competition

logging.basicConfig(
    level=logging.DEBUG,
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
    """Encapsulates all per‑team computations (minutes, points, lineups…)."""

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
        self.game_summaries: List[dict] = []  # Store {'game_idx': int, 'opponent_pts': int}
        self.processed_matches: set[str] = set()  # Track processed match IDs for summaries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        schedule: pd.DataFrame,
        all_moves: Dict[str, Sequence[dict]],
        all_stats: Dict[str, dict],  # Add match stats data
    ) -> None:
        """Iterate schedule rows & their events to populate aggregates."""
        # Iterate using iterrows() to reliably access columns by name
        processed_matches = 0
        skipped_matches_no_moves = 0
        skipped_matches_no_stats = 0
        skipped_matches_id_mapping = 0

        for index, row_series in schedule.iterrows():
            match_id = str(row_series.get("match_id", ""))
            if not match_id:
                continue  # Skip rows without match ID

            events = all_moves.get(match_id)
            stats_data = all_stats.get(match_id)

            if not events:
                # log.debug("No moves data found for match %s, skipping.", match_id)
                skipped_matches_no_moves += 1
                continue
            if not stats_data:
                # log.debug("No stats data found for match %s, skipping ID mapping.", match_id)
                skipped_matches_no_stats += 1
                continue

            # Pass the stats data to the processing function
            mapped = self._process_single_game(match_id, row_series, events, stats_data)
            if mapped:
                processed_matches += 1
            else:
                skipped_matches_id_mapping += 1

        # For single match report, we don't need the summary logs here
        # logger.info("Processing complete. Total matches in schedule: %d", len(schedule))
        # ...

    # ------------------------------------------------------------------
    # Private helpers – one game
    # ------------------------------------------------------------------

    def _process_single_game(
        self,
        match_id: str,
        row_series: pd.Series,
        events: Sequence[dict],
        stats_data: dict,  # Add stats data
    ) -> bool:  # Return True if processed, False if skipped due to mapping
        """Core loop; translates original long logic into smaller steps."""

        target_team_schedule_id = (
            self.team_id
        )  # ID from the schedule CSV (e.g., '69630')
        match_stats_teams = stats_data.get("teams", [])
        internal_id = None  # Internal ID used in moves/stats JSON (e.g., 319033)
        target_team_name = "Unknown"

        # Find internal_id using the mapping in match_stats_teams
        for team_info in match_stats_teams:
            if str(team_info.get("teamIdExtern")) == target_team_schedule_id:
                internal_id = team_info.get("teamIdIntern")
                target_team_name = team_info.get("name", target_team_name)
                break

        if internal_id is None:
            logger.debug(
                "Skipping match %s for schedule team ID %s – Cannot map to internal idTeam using match_stats data.",
                match_id,
                target_team_schedule_id,
            )
            return False  # Indicate mapping failure
        # else:
        #     logger.debug(
        #         "Mapped schedule team %s (%s) to internal ID %s for match %s",
        #         target_team_schedule_id,
        #         target_team_name,
        #         internal_id,
        #         match_id,
        #     )

        # --- The rest of the processing logic remains largely the same ---
        # ---- per‑game accumulators ------------------------------------------------
        per_game_minutes: Dict[str, float] = defaultdict(float)
        per_game_points: Dict[str, int] = defaultdict(int)
        per_game_plusminus: Dict[str, int] = defaultdict(int)
        per_game_on_pts_a: Dict[str, int] = defaultdict(int)  # Added for per-game DRtg
        game_opponent_pts: int = 0  # Total opponent points in this game
        current_lineup_tuple: tuple | None = None  # Track lineup for event attribution
        player_state: Dict[str, Dict] = {}
        on_court: Set[str] = set()
        last_event_abs_seconds = 0.0
        game_end_sec = 0.0
        last_processed_period = 0

        for event in events:
            event_type = event.get("move", "")
            period = event.get("period", 0)
            minute = event.get("minute", 0)
            second = event.get("second", 0)
            actor_name = event.get("actorName", "")
            actor_num = event.get("actorShirtNumber")
            event_team_id = event.get("idTeam")
            is_target_team_event = event_team_id == internal_id

            # --- Fix for absolute seconds calculation ---
            # Use the utility function which assumes 'min' is time remaining in period
            event_abs_seconds = get_absolute_seconds(period, minute, second)
            # --- End Fix ---

            # --- On/Off tracking -------------------------------------------------
            delta = event_abs_seconds - last_event_abs_seconds
            # current_lineup_tuple retains its value from the previous event unless delta > 0
            # Do NOT reset current_lineup_tuple here

            if delta > 0:
                for p in on_court:
                    self.on_secs[p] += delta
                # ----- accumulate lineup seconds and update current_lineup_tuple -----
                if len(on_court) == 5:
                    lineup_tuple = tuple(sorted(on_court))
                    # Use .get to safely access/initialize lineup stats dictionary
                    lu_stats = self.lineup_stats[lineup_tuple]
                    lu_stats["secs"] = lu_stats.get("secs", 0) + delta
                    current_lineup_tuple = lineup_tuple  # Store the active lineup tuple for subsequent delta=0 events
                else:
                    current_lineup_tuple = (
                        None  # Reset if time elapsed but not 5 players
                    )
                # ------------------------------------------------------------------
            # If delta == 0, current_lineup_tuple remains unchanged from the previous event

            # Check for points scored in the CURRENT event
            pts = self.points_map.get(event_type, 0)
            if pts:
                # --- ADD Player Plus/Minus Update ---
                point_diff = pts if is_target_team_event else -pts
                for p in on_court:
                    self.plus_minus[p] += point_diff
                    per_game_plusminus[p] += point_diff
                # --- END Player Plus/Minus Update ---

                # Update team and player On/Off points
                if is_target_team_event:
                    self.team_pts_f += pts
                    for p in on_court:
                        self.on_pts_f[p] += pts
                else:
                    self.team_pts_a += pts
                    game_opponent_pts += pts  # Accumulate opponent points for the game
                    for p in on_court:
                        self.on_pts_a[p] += pts
                        per_game_on_pts_a[p] += pts

                # --- Original Lineup point accumulation ---
                # Use current_lineup_tuple which is based on the state *before* this event
                if current_lineup_tuple is not None:
                    if is_target_team_event:  # Target team scored
                        self.lineup_stats[current_lineup_tuple]["pts_for"] = (
                            self.lineup_stats[current_lineup_tuple].get("pts_for", 0)
                            + pts
                        )
                    elif not is_target_team_event:  # Opponent scored
                        self.lineup_stats[current_lineup_tuple]["pts_against"] = (
                            self.lineup_stats[current_lineup_tuple].get(
                                "pts_against", 0
                            )
                            + pts
                        )
                # --- END Lineup point accumulation ---

            # --------------------------------------------------------------------
            last_event_abs_seconds = event_abs_seconds
            game_end_sec = max(
                game_end_sec, event_abs_seconds
            )  # Keep track of game end time

            # Process only if it's a target team event, has an actor name,
            # and the actor name is NOT the team name itself (i.e., ignore team actions like Timeouts)
            if is_target_team_event and actor_name and actor_name != target_team_name:
                # Ensure player is in our main tracker
                if actor_name not in self.players:
                    self.players[actor_name] = PlayerAggregate(number=actor_num)
                elif self.players[actor_name].number is None and actor_num is not None:
                    self.players[actor_name].number = (
                        actor_num  # Fill in number if missing
                    )

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
                elif event_type == "Cistella de 1":  # Correct event for FT made
                    pa.t1 += 1
                    pa.pts += 1
                    per_game_points[actor_name] += 1
                elif event_type.startswith(
                    "Personal"
                ):  # Check if it starts with Personal
                    pa.fouls += 1

                # --- Handle Minutes Played & Pairwise ---
                if event_type == "Entra al camp":
                    if st["status"] == "out":
                        st["status"] = "in"
                        st["since"] = event_abs_seconds
                        # Update pairwise for newly entered player
                        for other_player in on_court:
                            self._credit_pairwise_secs(actor_name, other_player, 0)
                        on_court.add(actor_name)
                elif event_type == "Surt del camp":
                    if st["status"] == "in":
                        duration = event_abs_seconds - st["since"]
                        self._credit_minutes(actor_name, duration)
                        per_game_minutes[actor_name] += duration
                        # Credit pairwise time for player leaving
                        on_court.remove(actor_name)
                        for other_player in on_court:
                            self._credit_pairwise_secs(
                                actor_name, other_player, duration
                            )
                        st["status"] = "out"
                        st["since"] = event_abs_seconds

        # --- End of Game Processing ---
        # Flush remaining On/Off seconds from the last event until the final horn
        remaining = game_end_sec - last_event_abs_seconds
        if remaining > 0:
            for p in on_court:
                self.on_secs[p] += remaining
        if len(on_court) == 5 and remaining > 0:
            lineup_tuple = tuple(sorted(on_court))
            self.lineup_stats[lineup_tuple]["secs"] += remaining
        # Credit remaining time for players still on court
        for p, st in player_state.items():
            if st["status"] == "in":
                duration = game_end_sec - st["since"]
                self._credit_minutes(p, duration)
                per_game_minutes[p] += duration
                # self.on_secs[p] += duration
                # Credit remaining pairwise time
                temp_on_court = on_court.copy()
                temp_on_court.remove(p)
                for other_player in temp_on_court:
                    self._credit_pairwise_secs(p, other_player, duration)

        # Increment GP for players who participated in this game
        # For single match report, GP will always be 1 if player played
        for player_name in player_state:
            self.players[player_name].gp = 1  # Set GP to 1

        # ---- save game summary (opponent points) --------------------------------
        # We only process one game, so game_idx is always 1 for logs/summaries
        game_idx = 1
        if player_state:
            # Add summary if match hasn't been processed before (safeguard)
            if match_id not in self.processed_matches:
                self.game_summaries.append({
                    'game_idx': game_idx,  # Always 1 for single match context
                    'opponent_pts': game_opponent_pts
                })
                self.processed_matches.add(match_id)
        # else:
        #     game_idx = None # No game index if no players/actions for target team

        # ---- save per‑player game row (for evolution-like data within the match) ---
        # This will be a single point per player if they played
        if game_idx is not None:  # Only log if game_idx was assigned (i.e., target team played)
            for pname, secs in per_game_minutes.items():
                if pname in self.players:  # Only log if player is tracked
                    self.game_log.append(
                        {
                            "game_idx": game_idx,  # Always 1
                            "player": pname,
                            "mins": round(secs / 60, 1),
                            "pts": per_game_points.get(pname, 0),
                            "+/-": per_game_plusminus.get(pname, 0),
                            "drtg": (  # DRtg doesn't make sense for single game, maybe remove later
                                round(
                                    (per_game_on_pts_a.get(pname, 0) * 40 / (secs / 60)), 1
                                )
                                if secs >= 60
                                else None
                            ),
                        }
                    )

        return True  # Indicate successful processing

    # ------------------------------------------------------------------
    # Evolution (rolling) table - Not applicable for single match report
    # ------------------------------------------------------------------
    # def evolution_table(self, names_to_exclude: set[str] = set()) -> pd.DataFrame:
    #     ...

    # ------------------------------------------------------------------
    # Lineup summary table
    # ------------------------------------------------------------------
    def lineup_table(
        self, min_usage: float = 0.0, names_to_exclude: set[str] = set()  # Lower default usage for single game
    ) -> pd.DataFrame:
        """Return DataFrame of lineups filtered by usage with Net Rating.

        Avoids KeyError when no qualifying lineups exist.
        Adjusted min_usage default for single-game context.
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
                    "lineup": '\n'.join(map(shorten_name, lu)),
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
        data = {
            "Player": [],
            "#": [],
            # "GP": [], # GP is always 1 for single match
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
                # data["GP"].append(aggr.gp)
                data["Mins"].append(aggr.minutes)
                data["PTS"].append(aggr.pts)
                data["T3"].append(aggr.t3)
                data["T2"].append(aggr.t2)
                data["T1"].append(aggr.t1)
                data["Fouls"].append(aggr.fouls)
                data["+/-"].append(self.plus_minus.get(name, 0))
        return pd.DataFrame(data)

    def pairwise_minutes(self, names_to_exclude: set[str] = set()) -> pd.DataFrame:
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
                total_game_secs = max(row['mins'] for row in self.game_log) * 60
            else:
                logger.warning("Could not determine total game seconds for On/Off calculation.")
                return pd.DataFrame()  # no data or way to calc off

        # Iterate through filtered players
        for player, secs_on in valid_on_secs.items():
            mins_on = secs_on / 60
            if mins_on < 0.1:  # Lower threshold for single game
                continue  # skip very small samples

            mins_off = (total_game_secs - secs_on) / 60
            on_net = (self.on_pts_f[player] - self.on_pts_a[player]) * 40 / mins_on if mins_on > 0 else 0

            if mins_off >= 0.1:  # Lower threshold for single game
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
                    "Mins_ON": round(mins_on, 1),
                    "On_Net": round(on_net, 1),
                    "Off_Net": round(off_net, 1) if off_net is not None else "—",
                    "ON-OFF": round(diff, 1) if diff is not None else "—",
                }
            )
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Ensure ON-OFF is numeric for sorting, coercing errors to NaN
        df['ON-OFF'] = pd.to_numeric(df['ON-OFF'], errors='coerce')

        return df.sort_values("ON-OFF", ascending=False, na_position="last").reset_index(drop=True)


################################################################################
# Plotting Functions (Adapted for single match reports)                         #
################################################################################

def plot_score_timeline(match_moves: List[dict], match_stats: dict, match_output_dir: Path, target_team_id: str) -> \
Optional[str]:
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
    for team_info in match_stats.get('teams', []):
        if str(team_info.get('teamIdExtern')) == target_team_id:
            # This logic assumes target team is one of the two teams in match_stats
            # We need to figure out if target was local or visitor *in this match*
            # Let's find both internal IDs first
            pass  # Need a robust way to map local/visitor schedule ID to internal ID

    # Simplified: Assume team 0 in match_stats is local, team 1 is visitor (based on structure)
    # This might be fragile if the order isn't guaranteed.
    if len(match_stats.get('teams', [])) == 2:
        local_internal_id = match_stats['teams'][0].get('teamIdIntern')
        visitor_internal_id = match_stats['teams'][1].get('teamIdIntern')
    else:
        logger.warning("Could not determine internal IDs for score timeline plot.")
        return None

    if not local_internal_id or not visitor_internal_id:
        logger.warning("Internal IDs missing for score timeline plot.")
        return None

    # Get team names from match_stats for legend
    local_name = match_stats['teams'][0].get('name', 'Local')
    visitor_name = match_stats['teams'][1].get('name', 'Visitor')

    times.append(0)
    local_scores.append(0)
    visitor_scores.append(0)

    points_map = {"Cistella de 1": 1, "Cistella de 2": 2, "Cistella de 3": 3}

    for event in sorted(match_moves,
                        key=lambda x: get_absolute_seconds(x.get('period', 0), x.get('min', 0), x.get('sec', 0))):
        event_time = get_absolute_seconds(event.get('period', 0), event.get('min', 0), event.get('sec', 0))
        pts = points_map.get(event.get('move', ''), 0)
        if pts > 0:
            event_team_id = event.get('idTeam')
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
        plt.plot(times, local_scores, label=f"{local_name}", color='#003366')
        plt.plot(times, visitor_scores, label=f"{visitor_name}", color='#7FFFD4')
        plt.xlabel("Time (Minutes)")
        plt.ylabel("Score")
        plt.title("Score Progression")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(filename, dpi=100)
        plt.close()
        logger.debug(f"Saved score timeline plot: {filename}")
        return relative_path
    except Exception as e:
        logger.error(f"Failed to generate score timeline plot: {e}")
        plt.close()  # Attempt to close figure on error
        return None


def plot_pairwise_heatmap(pairwise_df: pd.DataFrame, match_output_dir: Path) -> Optional[str]:
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
        plt.figure(figsize=(max(6, pairwise_df.shape[1] * 0.8), max(4, pairwise_df.shape[0] * 0.6)))  # Dynamic size
        sns.heatmap(pairwise_df, annot=True, fmt="d", cmap="Reds", linewidths=0.5)
        plt.title(f"Pairwise Minutes Played Together")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        # Removed figtext description for single match context
        plt.tight_layout()
        plt.savefig(filename, dpi=100)
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
            palette="coolwarm",
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
                fontsize=9
            )

        plt.title(f"Player On-Court Net Rating (per 40 min)")
        plt.xlabel("Player")
        plt.ylabel("On_Net Rating")
        plt.xticks(rotation=45, ha="right")
        # Removed figtext description
        plt.tight_layout()
        plt.savefig(filename, dpi=100)
        plt.close()
        logger.debug(f"Saved player On_Net plot: {filename}")
        return relative_path
    except Exception as e:
        logger.error(f"Failed to generate/save player On_Net plot: {e}")
        plt.close()
        return None


def plot_lineup_netrtg(lineup_df: pd.DataFrame, match_output_dir: Path) -> Optional[str]:
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

    top_n = 10  # Show fewer lineups for single match
    lineup_df_sorted = lineup_df.sort_values("NetRtg", ascending=False).head(top_n)

    if lineup_df_sorted.empty:  # Check again after head(top_n)
        logger.info("No lineups found after filtering top N, skipping plot.")
        return None

    filename = match_output_dir / "lineup_chart.png"
    relative_path = filename.name

    try:
        plt.figure(figsize=(max(10, len(lineup_df_sorted) * 1.2), 6))  # Dynamic width
        barplot = sns.barplot(
            x="lineup",
            y="NetRtg",
            hue="lineup",  # Use hue for consistency, but legend=False
            data=lineup_df_sorted,
            palette="viridis",
            legend=False,
        )
        # Add value labels
        for i, row in lineup_df_sorted.iterrows():
            value = row.NetRtg
            pos = lineup_df_sorted.index.get_loc(i)  # Get numerical position
            barplot.text(
                pos,
                value + (np.sign(value) * 1 if value != 0 else 1),
                f"{value:.1f}",
                color="black",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9
            )
        plt.title(f"Top {len(lineup_df_sorted)} Lineup Net Rating (per 40 min)")
        plt.xlabel("Lineup")
        plt.ylabel("Net Rating")
        plt.xticks(rotation=0, ha="center")
        # Removed figtext
        plt.tight_layout()
        plt.savefig(filename, dpi=100)
        plt.close()
        logger.debug(f"Saved Lineup NetRtg Bar Chart: {filename}")
        return relative_path
    except Exception as e:
        logger.error(f"Failed to generate/save lineup NetRtg plot: {e}")
        plt.close()
        return None


################################################################################
# Report Generation Functions                                                  #
################################################################################

def generate_summary_md(match_info: pd.Series, match_stats: dict, target_team_id: str) -> str:
    """Generates a markdown summary of the match."""
    # Extract basic info from schedule row
    local_name = match_info.get("local_team", "Local")
    visitor_name = match_info.get("visitor_team", "Visitor")
    local_id = str(match_info.get("local_team_id", ""))
    visitor_id = str(match_info.get("visitor_team_id", ""))
    score = match_info.get("score", "-")
    match_date_time = match_info.get("date_time", "Unknown")
    match_id = match_info.get("match_id", "Unknown")

    # Determine target and opponent
    is_target_local = (target_team_id == local_id)
    target_team_name = local_name if is_target_local else visitor_name
    opponent_team_name = visitor_name if is_target_local else local_name

    # Extract scores if possible
    try:
        local_score, visitor_score = map(int, score.split('-'))
    except ValueError:
        local_score, visitor_score = "?", "?"

    target_score = local_score if is_target_local else visitor_score
    opponent_score = visitor_score if is_target_local else local_score

    # Extract detailed stats from match_stats if available
    # Remove redundant header info, keep only the stats table
    summary_lines = []
    # summary_lines = [
    #     f"# Match Report: {target_team_name} vs {opponent_team_name}",
    #     f"",
    #     f"**Date:** {match_date_time}",
    #     f"**Match ID:** {match_id}",
    #     f"## Final Score: {local_name} {local_score} - {visitor_score} {visitor_name}",
    #     f"",
    # ]

    # Add team stats comparison if possible
    if 'teams' in match_stats and len(match_stats['teams']) == 2:
        team1_stats = match_stats['teams'][0]
        team2_stats = match_stats['teams'][1]

        # Ensure we correctly identify target and opponent stats within match_stats
        target_stats = None
        opponent_stats = None
        if str(team1_stats.get('teamIdExtern')) == target_team_id:
            target_stats = team1_stats
            opponent_stats = team2_stats
        elif str(team2_stats.get('teamIdExtern')) == target_team_id:
            target_stats = team2_stats
            opponent_stats = team1_stats

        if target_stats and opponent_stats:
            summary_lines.append("## Team Stat Comparison")
            summary_lines.append(f"| Stat         | {target_team_name} | {opponent_team_name} |")
            summary_lines.append("|:-------------|-------------------:|-------------------:|")

            # Example stats - add more as needed from match_stats structure
            # Access nested 'data' dictionary safely with .get('data', {})
            ts_data = target_stats.get('data', {})
            os_data = opponent_stats.get('data', {})

            summary_lines.append(f"| Points       | {ts_data.get('score', '?')} | {os_data.get('score', '?')} |")
            summary_lines.append(
                f"| T2 Made/Att  | {ts_data.get('shotsOfTwoSuccessful', '?')}/{ts_data.get('shotsOfTwoAttempted', '?')} | {os_data.get('shotsOfTwoSuccessful', '?')}/{os_data.get('shotsOfTwoAttempted', '?')} |")
            summary_lines.append(
                f"| T3 Made/Att  | {ts_data.get('shotsOfThreeSuccessful', '?')}/{ts_data.get('shotsOfThreeAttempted', '?')} | {os_data.get('shotsOfThreeSuccessful', '?')}/{os_data.get('shotsOfThreeAttempted', '?')} |")
            summary_lines.append(
                f"| T1 Made/Att  | {ts_data.get('shotsOfOneSuccessful', '?')}/{ts_data.get('shotsOfOneAttempted', '?')} | {os_data.get('shotsOfOneSuccessful', '?')}/{os_data.get('shotsOfOneAttempted', '?')} |")
            summary_lines.append(f"| Fouls        | {ts_data.get('faults', '?')} | {os_data.get('faults', '?')} |")
            summary_lines.append(f"")
        else:
            logger.warning(f"Could not properly match teams in match_stats for {match_id}")

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
    if 'teams' in match_stats and len(match_stats['teams']) == 2:
        team1_stats = match_stats['teams'][0]
        team2_stats = match_stats['teams'][1]
        target_stats = None
        opponent_stats = None
        if str(team1_stats.get('teamIdExtern')) == target_team_id:
            target_stats = team1_stats
            opponent_stats = team2_stats
        elif str(team2_stats.get('teamIdExtern')) == target_team_id:
            target_stats = team2_stats
            opponent_stats = team1_stats

        if target_stats and opponent_stats:
            ts_data = target_stats.get('data', {})
            os_data = opponent_stats.get('data', {})
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
        top_scorers = player_stats_df.nlargest(3, 'PTS')[['Player', 'PTS']]
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
    if 'quarters' in match_stats:
        quarters = match_stats['quarters']
        quarters_str = " | ".join(
            f"Q{i + 1}: {q.get('home', '?')}-{q.get('away', '?')}" for i, q in enumerate(quarters))
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
        logger.warning("OPENAI_API_KEY environment variable not set. Skipping LLM summary.")
        return None

    try:
        # Use a common default model, consider making this configurable
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,  # Limit output length
            temperature=0.5,  # Lower temperature for more factual summary
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate individual match HTML reports.")
    parser.add_argument("--team", required=True, help="Team ID to focus reports on")
    parser.add_argument(
        "--groups", nargs='+', required=True, help="Competition group IDs to scan for matches"
    )
    parser.add_argument(
        "--data-dir", default="data", help="Root data folder (CSV & JSON)"
    )
    parser.add_argument(
        "--output-dir", default="reports", help="Directory to save generated match reports"
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
    team_stats_dir = data_dir / "team_stats"
    output_dir = Path(args.output_dir)
    target_team_id = args.team

    # Jinja2 Environment
    # For simplicity, embedding template here. Could load from file.
    # env = Environment(loader=FileSystemLoader('.'), autoescape=select_autoescape(['html', 'xml'])) # Example if loading from file
    env = Environment(loader=None, autoescape=select_autoescape(['html', 'xml']))  # Loader=None as template is string
    template = env.from_string(HTML_TEMPLATE)

    logger.info(f"Starting report generation for Team ID: {target_team_id}")
    logger.info(f"Data Source: {data_dir.resolve()}")
    logger.info(f"Output Directory: {output_dir.resolve()}")

    processed_match_count = 0
    skipped_match_count = 0
    report_links_data = [] # Initialize list to store report link info

    for gid in args.groups:
        logger.info(f"--- Processing Group: {gid} ---")
        csv_path = data_dir / f"results_{gid}.csv"

        if not csv_path.exists():
            logger.warning(f"Schedule file not found: {csv_path}. Skipping group {gid}.")
            continue

        try:
            schedule_df = load_schedule(csv_path)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error loading schedule {csv_path}: {e}. Skipping group {gid}.")
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
                visitor_team_id = str(int(float(match_info_row.get("visitor_team_id", ""))))
            except (ValueError, TypeError):
                visitor_team_id = str(match_info_row.get("visitor_team_id", ""))
            # --- End Robust Team ID Conversion ---

            local_name = match_info_row.get("local_team", "?")
            visitor_name = match_info_row.get("visitor_team", "?")

            if not match_id or match_id.lower() == 'nan': # Also check for 'nan' string
                logger.warning("Skipping row with missing or invalid match_id in %s", csv_path)
                continue

            # --- Debugging Log ---
            logger.debug(
                f"Match {match_id}: Local={local_name} ({local_team_id}), Visitor={visitor_name} ({visitor_team_id}), Target={target_team_id}")
            # --- End Debugging Log ---

            # Check if the target team played in this match
            if target_team_id != local_team_id and target_team_id != visitor_team_id:
                # logger.debug(f"Skipping match {match_id}: Target team {target_team_id} did not play.")
                continue

            logger.info(f"Processing Match ID: {match_id} (Group: {gid})")

            # --- Create output directory for this match ---
            match_output_dir = output_dir / f"match_{match_id}"
            match_output_dir.mkdir(parents=True, exist_ok=True)

            # --- Load data for this specific match ---
            match_moves = load_match_moves(match_id, moves_dir)
            match_stats = load_match_stats(match_id, stats_dir)

            if not match_moves:
                logger.warning(f"Missing moves JSON for match {match_id}. Cannot generate full report.")
                # Optionally create a minimal report or skip
                skipped_match_count += 1
                continue
            if not match_stats:
                logger.warning(f"Missing stats JSON for match {match_id}. Cannot generate full report.")
                # Optionally create a minimal report or skip
                skipped_match_count += 1
                continue

            # --- Calculate Stats for the target team in this match ---
            # Instantiate calculator for THIS team and THIS match
            calc = StatsCalculator(target_team_id)
            # We need a DataFrame-like structure for process, even for one row
            single_match_schedule_df = pd.DataFrame([match_info_row])
            calc.process(
                single_match_schedule_df,
                {match_id: match_moves},
                {match_id: match_stats}
            )

            # --- Generate Report Components ---

            # 1. Summary (summary.md) - Placeholder function
            summary_content = generate_summary_md(match_info_row, match_stats, target_team_id)
            summary_md_path = match_output_dir / "summary.md"
            summary_md_path.write_text(summary_content, encoding='utf-8')
            summary_html = markdown2.markdown(summary_content, extras=["tables"])

            # 2. Statistics Tables (from calc)
            player_df = calc.player_table()  # Note: excludes handled if arg added later
            onoff_df = calc.on_off_table()
            lineup_df = calc.lineup_table()  # Uses lower default min_usage

            player_table_html = player_df.to_html(index=False, classes='stats-table',
                                                  border=0) if not player_df.empty else "<p>(No player data)</p>"
            on_off_table_html = onoff_df.to_html(index=False, classes='stats-table',
                                                 border=0) if not onoff_df.empty else "<p>(No On/Off data)</p>"
            lineup_table_html = lineup_df.head(10).to_html(index=False, classes='stats-table', border=0) if not lineup_df.empty else "<p>(No lineup data)</p>"

            # 3. AI Summary (Cached)
            llm_summary_text = None
            llm_summary_md_path = match_output_dir / "llm_summary.md"

            if llm_summary_md_path.exists():
                try:
                    llm_summary_text = llm_summary_md_path.read_text(encoding='utf-8').strip()
                    if llm_summary_text: # Ensure we read something
                        logger.info(f"Using cached LLM summary from: {llm_summary_md_path}")
                    else:
                        logger.warning(f"Cached LLM summary file is empty: {llm_summary_md_path}. Will attempt regeneration.")
                        llm_summary_text = None # Treat empty file as non-existent for regeneration
                except Exception as e:
                    logger.error(f"Error reading cached LLM summary {llm_summary_md_path}: {e}. Will attempt regeneration.")
                    llm_summary_text = None # Attempt regeneration on read error
            
            if llm_summary_text is None: # If cache doesn't exist, is empty, or failed to read
                logger.info(f"LLM summary cache not found or invalid. Generating...")
                llm_summary_text = generate_llm_summary(
                    match_info_row, match_stats, target_team_id, player_df
                )
                if llm_summary_text:
                    try:
                        llm_summary_md_path.write_text(llm_summary_text, encoding='utf-8')
                        logger.info(f"Saved newly generated LLM summary to: {llm_summary_md_path}")
                    except Exception as e:
                        logger.error(f"Error writing LLM summary cache {llm_summary_md_path}: {e}")

            # 4. Charts (PNGs) - Placeholder paths, functions to be added/called
            score_timeline_path_rel = plot_score_timeline(match_moves, match_stats, match_output_dir, target_team_id)
            pairwise_heatmap_path_rel = plot_pairwise_heatmap(calc.pairwise_minutes(), match_output_dir)
            on_net_chart_path_rel = plot_player_on_net(onoff_df, match_output_dir)
            lineup_chart_path_rel = plot_lineup_netrtg(lineup_df, match_output_dir)

            # --- Get Team Names for Report Title ---
            local_name = match_info_row.get("local_team", "Local Team")  # Correct key
            visitor_name = match_info_row.get("visitor_team", "Visitor Team")  # Correct key
            team_name = local_name if target_team_id == local_team_id else visitor_name
            opponent_name = visitor_name if target_team_id == local_team_id else local_name
            match_date = match_info_row.get("date_time", "Unknown Date")  # Correct key
            # Group name - ideally load from a mapping like in process_data.py if available
            group_name = f"Group {gid}"  # Placeholder
            score = match_info_row.get("score", "-")

            # --- Render HTML Report ---
            html_content = template.render(
                match_id=match_id,
                group_name=group_name,
                match_date=match_date,
                target_team_name=team_name, # Explicitly pass target team name
                local_name=local_name, # Pass original local name
                visitor_name=visitor_name, # Pass original visitor name
                summary_html=summary_html,
                player_table_html=player_table_html,
                on_off_table_html=on_off_table_html,
                lineup_table_html=lineup_table_html,
                # Pass relative paths for images
                score_timeline_path=score_timeline_path_rel,
                pairwise_heatmap_path=pairwise_heatmap_path_rel,
                on_net_chart_path=on_net_chart_path_rel,
                lineup_chart_path=lineup_chart_path_rel,
                season=args.season,  # Add season for link generation
                llm_summary=llm_summary_text,  # Add LLM summary
            )

            report_html_path = match_output_dir / "report.html"
            report_html_path.write_text(html_content, encoding='utf-8')
            logger.info(f"-> Successfully generated report: {report_html_path}")
            processed_match_count += 1

            # Store report link info
            report_links_data.append({
                'match_id': match_id,
                'target_team_name': team_name, # Keep track of the target team
                'opponent_name': opponent_name,
                'local_name': local_name, # Store original local team name
                'visitor_name': visitor_name, # Store original visitor team name
                'match_date': match_date,
                'group_name': group_name,
                'score': score, # Add score here
                'report_path': str(report_html_path.relative_to(output_dir))
            })

        logger.info(f"--- Finished Group: {gid} ---")

    logger.info(f"Report generation complete.")
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
                    return pd.to_datetime(report.get('match_date', '1900-01-01'), format='%d-%m-%Y %H:%M', errors='coerce')
                except ValueError:
                     # Fallback for unexpected formats or unparseable dates
                    return pd.Timestamp('1900-01-01')

            sorted_reports = sorted(report_links_data, key=sort_key)

            index_html_content = index_template.render(
                reports=sorted_reports,
                target_team_id=target_team_id # Pass team ID for title
            )
            index_html_path = output_dir / "index.html"
            index_html_path.write_text(index_html_content, encoding='utf-8')
            logger.info(f"Successfully generated index file: {index_html_path}")
        except Exception as e:
            logger.error(f"Failed to generate index.html: {e}")
            import traceback
            traceback.print_exc() # Print stack trace for debugging Jinja errors
    else:
        logger.info(f"No reports were generated for Team {target_team_id}, skipping index.html creation.")


if __name__ == "__main__":
    # Add json import at the top if not already there
    import json 
    # Need pandas for date sorting fallback
    import pandas as pd 
    # Need traceback for debugging
    import traceback 
    main()
