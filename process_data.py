"""
U13 ASFE season analysis – refactored module
-------------------------------------------
• Python 3.10+
• Pandas ≥ 2.0
• No external dependencies beyond std-lib + pandas

Usage (CLI example)
-------------------
$ python u13_analysis_refactor.py --team 69630 --groups 17182 18299 --data-dir data

This script prints basic player & lineup tables equivalent to the original script
but generated through a cleaner, testable architecture.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Any

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import pi

################################################################################
# Configuration & logging                                                       #
################################################################################

PERIOD_LENGTH_SEC: int = 600  # 10‑minute quarters in Catalan U13 competition

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("u13_analysis")

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
    if len(parts) > 1: # Only modify if there's more than one part
        return " ".join(parts[:-1]) # Join all parts except the last one
    return full_name # Return original if it's just one word or empty


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
    df = pd.read_csv(path, encoding='utf-8')
    if "match_id" not in df.columns:
        logger.error("Could not find 'match_id' column. Columns found: %s", df.columns.tolist())
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
        logger.warning("Could not decode team stats JSON for %s/%s: %s", team_id, season, exc)
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
# Core calculator                                                               #
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
        self.team_pts: int = 0
        self.team_fouls: int = 0
        # --- On/Off tracking ---
        self.on_secs: Dict[str, float] = defaultdict(float)   # seconds with player ON court
        self.on_pts_f: Dict[str, int] = defaultdict(int)      # points for while ON
        self.on_pts_a: Dict[str, int] = defaultdict(int)      # points against while ON
        self.team_pts_f: int = 0                              # total points for (team)
        self.team_pts_a: int = 0                              # total points against
        # --- Evolution & lineup tracking ---
        self.game_log: List[dict] = []  # one row per player per game
        self.lineup_stats: Dict[tuple, Dict[str, Any]] = defaultdict(lambda: {'secs': 0.0,
                                                                              'pts_for': 0,
                                                                              'pts_against': 0})
        self.plus_minus: Dict[str, int] = defaultdict(int)  # +/- por jugadora
        self.def_rating: Dict[str, float] = defaultdict(float)  # Defensive rating por jugadora

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self, schedule: pd.DataFrame, all_moves: Dict[str, Sequence[dict]],
        all_stats: Dict[str, dict] # Add match stats data
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
                continue # Skip rows without match ID

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

        logger.info("Processing complete. Total matches in schedule: %d", len(schedule))
        logger.info("Successfully processed stats for %d matches.", processed_matches)
        if skipped_matches_no_moves > 0:
            logger.warning("Skipped %d matches due to missing moves JSON.", skipped_matches_no_moves)
        if skipped_matches_no_stats > 0:
            logger.warning("Skipped %d matches due to missing stats JSON.", skipped_matches_no_stats)
        if skipped_matches_id_mapping > 0:
            logger.warning("Skipped %d matches due to failed team ID mapping (check stats JSON).", skipped_matches_id_mapping)
            
        # Log final lineup stats before generating table
        # logger.debug("Final Lineup Stats before table generation:") # REMOVED
        # for lu, stats in self.lineup_stats.items(): # REMOVED
            # logger.debug(f"  Lineup: {lu} -> {stats}") # REMOVED

    # ------------------------------------------------------------------
    # Private helpers – one game
    # ------------------------------------------------------------------

    def _process_single_game(
        self, match_id: str, row_series: pd.Series, events: Sequence[dict],
        stats_data: dict # Add stats data
    ) -> bool: # Return True if processed, False if skipped due to mapping
        """Core loop; translates original long logic into smaller steps."""

        target_team_schedule_id = self.team_id # ID from the schedule CSV (e.g., '69630')
        match_stats_teams = stats_data.get("teams", [])
        internal_id = None # Internal ID used in moves/stats JSON (e.g., 319033)
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
                match_id, target_team_schedule_id
            )
            return False # Indicate mapping failure
        else:
            logger.debug("Mapped schedule team %s (%s) to internal ID %s for match %s",
                         target_team_schedule_id, target_team_name, internal_id, match_id)

        # --- The rest of the processing logic remains largely the same ---
        # ---- per‑game accumulators ------------------------------------------------
        per_game_minutes: Dict[str, float] = defaultdict(float)
        per_game_points: Dict[str, int] = defaultdict(int)
        per_game_plusminus: Dict[str, int] = defaultdict(int)
        per_game_on_pts_a: Dict[str, int] = defaultdict(int) # Added for per-game DRtg
        last_lineup_tuple: tuple | None = None
        current_lineup_tuple: tuple | None = None # Track lineup for event attribution
        # ... (Initialize player_state, on_court, current_period_start_sec etc.)
        player_state: Dict[str, Dict] = {}
        on_court: Set[str] = set()
        current_period_start_sec = 0.0
        last_event_abs_seconds = 0.0
        game_end_sec = 0.0
        last_processed_period = 0

        for event in events:
            event_type = event.get("move", "")
            period = event.get("period", 0)
            minute = event.get("minute", 0)
            second = event.get("second", 0)
            actor_id = event.get("actorId")
            actor_name = event.get("actorName", "")
            actor_num = event.get("actorShirtNumber")
            event_team_id = event.get("idTeam")
            is_target_team_event = (event_team_id == internal_id)

            # Calculate absolute time in seconds for the event
            if period > last_processed_period:
                # Estimate period start time (assuming 10 min periods)
                # This might be inaccurate if overtime exists or periods have different lengths
                # A better approach would be to get actual period start/end times if available
                current_period_start_sec = last_event_abs_seconds # Or base on previous period end
                last_processed_period = period
                # Use the utility function which assumes 'minute' is time remaining in period
                event_abs_seconds = get_absolute_seconds(period, minute, second)

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
                    lu_stats['secs'] = lu_stats.get('secs', 0) + delta
                    # self.lineup_stats[lineup_tuple]['secs'] += delta # Original, potentially unsafe
                    current_lineup_tuple = lineup_tuple # Store the active lineup tuple for subsequent delta=0 events
                else:
                    current_lineup_tuple = None # Reset if time elapsed but not 5 players
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
                    for p in on_court:
                        self.on_pts_a[p] += pts
                        per_game_on_pts_a[p] += pts
                
                # --- Original Lineup point accumulation ---
                # Use current_lineup_tuple which is based on the state *before* this event
                if current_lineup_tuple is not None: 
                    if is_target_team_event: # Target team scored
                        self.lineup_stats[current_lineup_tuple]['pts_for'] = self.lineup_stats[current_lineup_tuple].get('pts_for', 0) + pts
                    elif not is_target_team_event: # Opponent scored
                         self.lineup_stats[current_lineup_tuple]['pts_against'] = self.lineup_stats[current_lineup_tuple].get('pts_against', 0) + pts
                    # Log the state *after* potential update for this specific lineup
                    # logger.debug(f"  LINEUP_PTS_UPDATE: {current_lineup_tuple} -> ...") # REMOVED
                    # pts_for for the lineup is implicitly handled ... <- Comment inaccurate
                # --- END Lineup point accumulation ---
                    
            # --- Check for non-scoring events relevant to lineup stats ---
            # ONLY when a 5-player lineup was active in the preceding interval and it's a target team event
            # elif is_target_team_event and current_lineup_tuple is not None: # Use elif to avoid double counting pts=0 events
            #      stats_dict = self.lineup_stats[current_lineup_tuple] 
            #      # Log when attempting to update for non-scoring events
            #      if event_type in ["Tir Fallat", "Tir lliure Fallat", "Pèrdua"]:
            #         logger.debug(f"LINEUP_UPDATE [Match:{match_id}, Lineup:{current_lineup_tuple}]: Event=\"{event_type}\"")
            #      # ... removed FGA/FTA/TOV increment logic ...

            # --------------------------------------------------------------------
            last_event_abs_seconds = event_abs_seconds
            game_end_sec = max(game_end_sec, event_abs_seconds) # Keep track of game end time

            # Process only if it's a target team event, has an actor name,
            # and the actor name is NOT the team name itself (i.e., ignore team actions like Timeouts)
            if is_target_team_event and actor_name and actor_name != target_team_name:
                # Ensure player is in our main tracker
                if actor_name not in self.players:
                    self.players[actor_name] = PlayerAggregate(number=actor_num)
                elif self.players[actor_name].number is None and actor_num is not None:
                     self.players[actor_name].number = actor_num # Fill in number if missing

                pa = self.players[actor_name]
                st = player_state.setdefault(
                    actor_name,
                    {"status": "out", "since": 0.0, "number": actor_num}
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
                elif event_type == "Cistella de 1": # Correct event for FT made
                    pa.t1 += 1
                    pa.pts += 1
                    per_game_points[actor_name] += 1
                elif event_type.startswith("Personal"): # Check if it starts with Personal
                    pa.fouls += 1
                # Add other event types if needed (rebounds, assists, etc.)

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
                             self._credit_pairwise_secs(actor_name, other_player, duration)
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
            self.lineup_stats[lineup_tuple]['secs'] += remaining
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
        if player_state: # Only increment if the target team played and had events
            for player_name in player_state:
                self.players[player_name].gp += 1

        # ---- save per‑player game row -------------------------------------------
        game_idx = len(self.game_log) // len(player_state) + 1 if player_state else 1 # Avoid division by zero if no players
        for pname, secs in per_game_minutes.items():
            # Ensure player exists in player_state before adding to game_log?
            # Or just add anyone with minutes?
            # Current logic adds anyone with per_game_minutes > 0
            if pname in self.players: # Only log if player is tracked
                self.game_log.append({
                    'game_idx': game_idx,
                    'player': pname,
                    'mins': round(secs / 60, 1),
                    'pts': per_game_points.get(pname, 0),
                    '+/-': per_game_plusminus.get(pname, 0),
                    'drtg': round((per_game_on_pts_a.get(pname, 0) * 40 / (secs / 60)), 1) if secs >= 60 else None
                })

        return True # Indicate successful processing

    # ------------------------------------------------------------------
    # Evolution (rolling) table
    # ------------------------------------------------------------------
    def evolution_table(self, names_to_exclude: set[str] = set()) -> pd.DataFrame:
        # Filter game log *before* creating DataFrame
        filtered_log = [row for row in self.game_log if row['player'] not in names_to_exclude]
        df = pd.DataFrame(filtered_log)
        if df.empty:
            return df
        df.sort_values('game_idx', inplace=True)
        df['roll_pts'] = df.groupby('player')['pts'].transform(lambda s: s.rolling(3, 1).mean())
        df['roll_mins'] = df.groupby('player')['mins'].transform(lambda s: s.rolling(3, 1).mean())
        df['roll_pm'] = df.groupby('player')['+/-'].transform(lambda s: s.rolling(3, 1).mean())
        df['roll_drtg'] = df.groupby('player')['drtg'].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        return df

    # ------------------------------------------------------------------
    # Lineup summary table
    # ------------------------------------------------------------------
    def lineup_table(self, min_usage: float = 0.05, names_to_exclude: set[str] = set()) -> pd.DataFrame:
        """Return DataFrame of lineups filtered by usage with Net Rating.

        Avoids KeyError when no qualifying lineups exist.
        """
        # Calculate total seconds across all tracked lineups with positive time
        # Filter lineups *before* calculating total seconds for usage %
        valid_lineups = {lu: v for lu, v in self.lineup_stats.items()
                         if v.get('secs', 0) > 0 and not any(p in names_to_exclude for p in lu)}

        total_lineup_secs = sum(stats['secs'] for stats in valid_lineups.values())
        if total_lineup_secs == 0:
            logger.info("No lineup data with positive time found after exclusions.")
            return pd.DataFrame()

        records = []
        # Iterate through the pre-filtered valid lineups
        # for lu, v in self.lineup_stats.items():
        for lu, v in valid_lineups.items():
            # if v['secs'] <= 0: # Already filtered
            #     continue
            # if any(p in names_to_exclude for p in lu): # Filter here
            #     continue

            # Calculate usage % relative to total *valid* lineup seconds
            usage_percent = (v['secs'] / total_lineup_secs) * 100 if total_lineup_secs > 0 else 0

            # Filter based on the calculated percentage (min_usage is 0.0 to 1.0)
            if (usage_percent / 100.0) < min_usage:
                continue
                
            # Log values used for NetRtg calculation
            # logger.debug(f"NETRTG_CALC Lineup: {lu}") # REMOVED
            # logger.debug(f"  Using: pts_for=... ") # REMOVED
            
            # Original NetRtg calculation
            net = (v.get('pts_for', 0) - v.get('pts_against', 0)) * 40 / (v.get('secs', 0) / 60) if v.get('secs', 0) > 0 else 0
            
            # Remove eFG% and Pace calculations
            # efg_pct = ... 
            # pace = ...
            
            records.append({
                'lineup': '- '.join(map(shorten_name, lu)),
                'mins': round(v['secs'] / 60, 1),
                'usage_%': round(usage_percent, 1),
                'NetRtg': round(net, 1) 
                # Removed 'eFG%' and 'Pace' keys
            })

        df = pd.DataFrame(records)
        # Sort by NetRtg by default, but ensure columns exist
        if df.empty or 'NetRtg' not in df.columns:
            return df  # nothing to sort
        return df.sort_values('NetRtg', ascending=False).reset_index(drop=True)

    def _credit_minutes(self, player: str, secs: float) -> None:
        pa = self.players[player]
        pa.merge_secs(secs)
        # pa.gp += 1  # Move GP increment to end of game processing

    def _credit_pairwise_secs(self, p1: str, p2: str, secs: float) -> None:
        """Adds played time to the pairwise matrix (for p1<->p2)."""
        # Ensure order doesn't matter (store under alphabetically first name)
        # Although defaultdict handles this, explicit check might be clearer
        # if p1 > p2:
        #     p1, p2 = p2, p1 # Swap to ensure consistency
        self.pairwise_secs[p1][p2] += secs
        if p1 != p2: # Avoid double-counting self-time
             self.pairwise_secs[p2][p1] += secs

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def player_table(self, names_to_exclude: set[str] = set()) -> pd.DataFrame:
        data = {
            "Player": [],
            "#": [],
            "GP": [],
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
            # Add filtering condition here
            if name not in names_to_exclude:
                data["Player"].append(shorten_name(name))
                data["#"].append(aggr.number)
                data["GP"].append(aggr.gp)
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
        valid_players = {name: data for name, data in self.players.items() if name not in names_to_exclude}
        if not valid_players:
            return pd.DataFrame() # Return empty if no players left

        sorted_players_by_minutes = sorted(
            # self.players.items(),
            valid_players.items(),
            key=lambda item: item[1].minutes,
            reverse=True
        )
        sorted_names = [name for name, _ in sorted_players_by_minutes]

        # Create matrix using the sorted names of *valid* players
        matrix = pd.DataFrame(index=sorted_names, columns=sorted_names, dtype=int)
        for p1 in sorted_names:
            for p2 in sorted_names:
                # No need for inner check as names are already filtered
                # if p1 not in names_to_exclude and p2 not in names_to_exclude:
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
        valid_on_secs = {p: s for p, s in self.on_secs.items() if p not in names_to_exclude}
        total_secs = sum(valid_on_secs.values())
        if total_secs == 0:
            return pd.DataFrame()  # no data

        # Iterate through filtered players
        # for player, secs_on in self.on_secs.items():
        for player, secs_on in valid_on_secs.items():
            # if player not in names_to_exclude: # No longer needed here
            mins_on = secs_on / 60
            if mins_on < 1:
                continue  # skip very small samples

            mins_off = (total_secs - secs_on) / 60
            on_net = (self.on_pts_f[player] - self.on_pts_a[player]) * 40 / mins_on

            if mins_off >= 1:
                off_pts_f = self.team_pts_f - self.on_pts_f[player]
                off_pts_a = self.team_pts_a - self.on_pts_a[player]
                off_net = (off_pts_f - off_pts_a) * 40 / mins_off
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
        return (
            pd.DataFrame(rows)
            .sort_values("ON-OFF", ascending=False, na_position="last")
            .reset_index(drop=True)
        )


################################################################################
# CLI entrypoint                                                               #
################################################################################


def main() -> None:
    parser = argparse.ArgumentParser(
        description="U13 ASFE season analysis (refactor)"
    )
    parser.add_argument("--team", required=True, help="Team ID to analyse")
    parser.add_argument("--groups", nargs="+", required=True, help="Competition group IDs")
    parser.add_argument(
        "--data-dir", default="data", help="Root data folder (CSV & JSON)"
    )
    parser.add_argument("--match", default=None, help="Optional: Process only a single match ID")
    parser.add_argument("--season", default="2024", help="Season identifier (default: 2024)")
    parser.add_argument("--plot-dir", default="plots", help="Directory to save plot images")
    parser.add_argument("--exclude-players", nargs='+', default=[], help="List of player UUIDs to exclude from reports")
    parser.add_argument("--compare-player", default=None, help="Player name (as in data) to compare between the two groups using a radar chart")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    moves_dir = data_dir / "match_moves"
    stats_dir = data_dir / "match_stats"
    team_stats_dir = data_dir / "team_stats" # Directory for team stats

    # --- Try to load team info for headers ---
    team_name_for_header = args.team # Default to ID
    group_name_for_header = "Unknown Group" # Default group name
    names_to_exclude = set() # Initialize empty set for names

    team_stats_data = load_team_stats(args.team, args.season, team_stats_dir)
    if team_stats_data and "team" in team_stats_data:
        team_name_for_header = team_stats_data["team"].get("teamName", args.team)
        group_name_for_header = team_stats_data["team"].get("categoryName", "Unknown Group")
        # Build the set of names to exclude based on UUIDs
        if args.exclude_players:
            logger.info(f"Excluding player UUIDs: {args.exclude_players}")
            for player_info in team_stats_data.get("players", []):
                player_uuid = player_info.get("uuid")
                player_name = player_info.get("name")
                if player_uuid and player_name and player_uuid in args.exclude_players:
                    names_to_exclude.add(player_name)
            if names_to_exclude:
                logger.info(f"Mapped to excluding names: {names_to_exclude}")
            else:
                logger.warning("Could not find names for provided excluded UUIDs in team stats file.")

    # Use categoryName as group name if available, otherwise keep it generic
    # The categoryName applies to the team/season, not necessarily the specific group ID being processed
    # We will use the specific group ID (gid) in the loop header for clarity

    if args.match:
        # --- Process Single Match ---
        # (Header logic for single match might also use team_name_for_header)
        logger.info(f"Processing single match ID: {args.match} for Team: {team_name_for_header} ({args.team})")
        match_found_in_schedule = False
        schedule_df_single = None
        moves_single = None
        stats_single = None

        for gid in args.groups:
            csv_path = data_dir / f"results_{gid}.csv"
            if not csv_path.exists():
                logger.warning("Schedule file %s not found, skipping group.", csv_path)
                continue
            try:
                schedule_df_group = load_schedule(csv_path)
                # Check if the match ID exists in this group's schedule
                match_row = schedule_df_group[schedule_df_group['match_id'] == args.match]
                if not match_row.empty:
                    logger.info(f"Match {args.match} found in group {gid} schedule.")
                    schedule_df_single = match_row
                    moves_single = load_match_moves(args.match, moves_dir)
                    stats_single = load_match_stats(args.match, stats_dir) # Load stats for the single match
                    match_found_in_schedule = True
                    break # Found the match, no need to check other groups
            except FileNotFoundError:
                logger.error("Schedule file not found (should not happen after check): %s", csv_path)
            except ValueError as e:
                logger.error("Error loading schedule %s: %s", csv_path, e)
            except Exception as e:
                 logger.error("Unexpected error processing schedule %s: %s", csv_path, e)

        if not match_found_in_schedule:
            logger.error(f"Match ID {args.match} not found in any specified group's schedule.")
            return

        if not moves_single:
            logger.error(f"Match moves data not found or failed to load for match {args.match}.")
            # Optionally continue if only stats are needed, but core calculation depends on moves
            return
        if not stats_single:
            logger.error(f"Match stats data not found or failed to load for match {args.match}. Required for team ID mapping.")
            return

        calc = StatsCalculator(args.team)
        # Pass the single match data as dictionaries keyed by the match ID
        calc.process(schedule_df_single, {args.match: moves_single}, {args.match: stats_single})

        # Display results for the single match
        print_results(calc, names_to_exclude)

    else:
        # --- Process Multiple Groups Separately ---
        group_results = {}
        for gid in args.groups:
            # Use loaded team name, but specific group ID (gid) for clarity
            print(f"\n{'='*15} Processing Group ID: {gid} | Team: {team_name_for_header} ({args.team}) {'='*15}")
            csv_path = data_dir / f"results_{gid}.csv"

            # Data specific to this group
            current_group_schedule = None
            current_group_moves: Dict[str, List[dict]] = {}
            current_group_stats: Dict[str, dict] = {}

            try:
                schedule_df = load_schedule(csv_path)
                current_group_schedule = schedule_df # Keep this group's schedule separate

                # Load moves and stats only for matches in this group's schedule
                match_ids_in_group = schedule_df["match_id"].dropna().astype(str).unique()
                logger.info(f"Loading JSON data for {len(match_ids_in_group)} matches in group {gid}...")
                for mid in match_ids_in_group:
                    loaded_moves = load_match_moves(mid, moves_dir)
                    if loaded_moves:
                        current_group_moves[mid] = loaded_moves
                    loaded_stats = load_match_stats(mid, stats_dir)
                    if loaded_stats:
                        current_group_stats[mid] = loaded_stats

            except FileNotFoundError:
                logger.error("Schedule file not found: %s. Skipping group %s.", csv_path, gid)
                continue
            except ValueError as e:
                logger.error("Error loading schedule %s: %s. Skipping group %s.", csv_path, e, gid)
                continue
            except Exception as e:
                 logger.error("Unexpected error processing schedule %s: %s. Skipping group %s.", csv_path, e, gid)
                 continue

            if current_group_schedule is None or current_group_schedule.empty:
                 logger.warning(f"No schedule data loaded for group {gid}. Skipping processing.")
                 continue

            # Instantiate and process *for this group only*
            logger.info(f"Calculating stats for group {gid}...")
            calc = StatsCalculator(args.team)
            calc.process(current_group_schedule, current_group_moves, current_group_stats)
            
            # --- Store results for potential comparison ---
            # Store calculated dataframes along with calc object if needed
            group_results[gid] = {
                'calc': calc,
                'evo_table': calc.evolution_table(names_to_exclude=names_to_exclude),
                'pairwise_df': calc.pairwise_minutes(names_to_exclude=names_to_exclude),
                'onoff_df': calc.on_off_table(names_to_exclude=names_to_exclude),
                'lineup_df': calc.lineup_table(names_to_exclude=names_to_exclude),
                'player_df': calc.player_table(names_to_exclude=names_to_exclude)
            }

            # Display results *for this group*
            print_results(calc, names_to_exclude)

            # --- Generate and Save Plots --- 
            plot_output_dir = Path(args.plot_dir)
            plot_output_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists
            
            # Existing evolution plots
            evo_table = calc.evolution_table(names_to_exclude=names_to_exclude)
            plot_evolution(evo_table, gid, plot_output_dir, names_to_exclude=names_to_exclude)
            
            # New Plots
            pairwise_df = calc.pairwise_minutes(names_to_exclude=names_to_exclude)
            plot_pairwise_heatmap(pairwise_df, gid, plot_output_dir)
            
            plot_minutes_stacked(evo_table, gid, plot_output_dir) # Uses evo_table
            
            onoff_df = calc.on_off_table(names_to_exclude=names_to_exclude)
            plot_player_on_net(onoff_df, gid, plot_output_dir)
            
            lineup_df = calc.lineup_table(names_to_exclude=names_to_exclude)
            plot_lineup_netrtg(lineup_df, gid, plot_output_dir)
            # --- End Plots --- 
            
            print(f"\n{'='*15} Finished Group: {gid} {'='*15}") # Mark end of group output

        # --- END GROUP LOOP ---
        
        # --- Generate Comparison Radar Chart (if applicable) ---
        if args.compare_player and len(args.groups) == 2:
            group_ids = args.groups
            player_name = args.compare_player # Use the name provided directly
            # Find the full name if excluded names were generated based on UUID
            # This requires access to the original team_stats_data name mapping if needed
            # Simple approach assumes compare_player is the full name from JSON
            if player_name in names_to_exclude:
                 logger.warning(f"Cannot generate comparison chart for excluded player: {player_name}")
            else:
                 logger.info(f"Attempting to generate comparison radar chart for player: {player_name} between groups {group_ids[0]} and {group_ids[1]}")
                 plot_output_dir = Path(args.plot_dir)
                 # Pass the whole results dict for flexibility
                 plot_player_comparison_radar(group_results, player_name, plot_output_dir)
        elif args.compare_player:
             logger.warning("Radar chart comparison requires exactly two group IDs to be provided.")

def print_results(calc: 'StatsCalculator', names_to_exclude: set[str]):
    """Helper function to print the standard output tables."""
    # Filter players before checking if data exists
    filtered_players = {name: data for name, data in calc.players.items() if name not in names_to_exclude}
    if not filtered_players:
        print("\n(No data found for this team OR all players excluded in the processed match(es))")
        return

    print("\n==== Player Aggregates ====")
    player_df = calc.player_table(names_to_exclude=names_to_exclude)
    if not player_df.empty:
        print(player_df.to_string(index=False))
    else:
        print("(No player data to display after exclusions)")

    print("\n==== Pairwise minutes (first 10×10) ====")
    pair_df = calc.pairwise_minutes(names_to_exclude=names_to_exclude)
    if not pair_df.empty:
        # Limit to max 10x10 for display
        max_dim = min(10, pair_df.shape[0], pair_df.shape[1])
        print(pair_df.iloc[:max_dim, :max_dim].to_string(float_format='{:d}'.format))
    else:
        print("(No pairwise data to display after exclusions)")

    print("\n==== On/Off Net Rating ====")
    onoff_df = calc.on_off_table(names_to_exclude=names_to_exclude)
    if not onoff_df.empty:
        print(onoff_df.to_string(index=False))
    else:
        print("(Insufficient data for On/Off calculation or all players excluded)")

    print("\n==== Evolution (rolling 3‑games) ====") # Removed 'sample' from header
    evo = calc.evolution_table(names_to_exclude=names_to_exclude)
    if not evo.empty:
        # print(evo.head(10).to_string(index=False))
        print(evo.to_string(index=False, float_format='{:.1f}'.format)) # Format floats to 1 decimal place
    else:
        print("(No evolution data after exclusions)")

    print("\n==== Lineups (usage ≥5 %) ====")
    lu_df = calc.lineup_table(names_to_exclude=names_to_exclude)
    if not lu_df.empty:
        print(lu_df.head(10).to_string(index=False))
    else:
        print("(No lineup data meeting criteria after exclusions)")


def plot_evolution(evo_df: pd.DataFrame, group_id: str, plot_dir: Path, names_to_exclude: set[str] = set()):
    """Generates and saves faceted line plots for player evolution."""
    # Filter DataFrame before plotting
    filtered_evo_df = evo_df[~evo_df['player'].isin(names_to_exclude)]

    # if evo_df.empty:
    if filtered_evo_df.empty:
        logger.info("Evolution DataFrame is empty after exclusions, skipping plots for group %s.", group_id)
        return

    # Ensure plot directory exists
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Melt DataFrame for easier plotting with Seaborn's hue/style
    df_melt_mins = filtered_evo_df.melt(id_vars=['game_idx', 'player'],
                               value_vars=['mins', 'roll_mins'],
                               var_name='Metric_Type',
                               value_name='Minutes')
    df_melt_pts = filtered_evo_df.melt(id_vars=['game_idx', 'player'],
                              value_vars=['pts', 'roll_pts'],
                              var_name='Metric_Type',
                              value_name='Points')

    # --- Plot Minutes ---
    try:
        logger.info("Generating Minutes evolution plot for group %s...", group_id)
        sns.set_theme(style="ticks")
        g_mins = sns.relplot(
            data=df_melt_mins,
            x="game_idx", y="Minutes",
            hue="player",  # Color lines by player
            style="Metric_Type", # Different style for actual vs rolling
            col="player", col_wrap=4, # Facet by player, 4 columns wide
            kind="line",
            height=3, aspect=1.5,
            legend=False # Avoid giant legend, colors identify player in title
        )
        g_mins.set_titles("{col_name}")
        g_mins.set_axis_labels("Game Index", "Minutes")
        g_mins.figure.suptitle(f'Minutes Evolution - Group {group_id}', y=1.03)
        plot_path_mins = plot_dir / f"evolution_minutes_group_{group_id}.png"
        g_mins.savefig(plot_path_mins, dpi=150)
        plt.close(g_mins.figure) # Close figure to free memory
        logger.info("-> Saved Minutes plot: %s", plot_path_mins)
    except Exception as e:
        logger.error("Failed to generate/save minutes plot for group %s: %s", group_id, e)
        if 'g_mins' in locals() and hasattr(g_mins, 'figure'):
             plt.close(g_mins.figure) # Attempt to close if figure exists

    # --- Plot Points ---
    try:
        logger.info("Generating Points evolution plot for group %s...", group_id)
        sns.set_theme(style="ticks")
        g_pts = sns.relplot(
            data=df_melt_pts,
            x="game_idx", y="Points",
            hue="player",
            style="Metric_Type",
            col="player", col_wrap=4,
            kind="line",
            height=3, aspect=1.5,
            legend=False
        )
        g_pts.set_titles("{col_name}")
        g_pts.set_axis_labels("Game Index", "Points")
        g_pts.figure.suptitle(f'Points Evolution - Group {group_id}', y=1.03)
        plot_path_pts = plot_dir / f"evolution_points_group_{group_id}.png"
        g_pts.savefig(plot_path_pts, dpi=150)
        plt.close(g_pts.figure) # Close figure
        logger.info("-> Saved Points plot: %s", plot_path_pts)
    except Exception as e:
        logger.error("Failed to generate/save points plot for group %s: %s", group_id, e)
        if 'g_pts' in locals() and hasattr(g_pts, 'figure'):
             plt.close(g_pts.figure)

    # --- Plot DRtg ---
    try:
        if 'roll_drtg' in filtered_evo_df.columns:
            logger.info("Generating DRtg evolution plot for group %s...", group_id)
            df_melt_drtg = filtered_evo_df.melt(id_vars=['game_idx', 'player'],
                                      value_vars=['drtg', 'roll_drtg'],
                                      var_name='Metric_Type',
                                      value_name='DRtg')
            sns.set_theme(style="ticks")
            g_drtg = sns.relplot(
                data=df_melt_drtg,
                x="game_idx", y="DRtg",
                hue="player",
                style="Metric_Type",
                col="player", col_wrap=4,
                kind="line",
                height=3, aspect=1.5,
                legend=False
            )
            g_drtg.set_titles("{col_name}")
            g_drtg.set_axis_labels("Game Index", "Defensive Rating (DRtg)")
            for ax in g_drtg.axes.flat:
                ax.invert_yaxis()  # Lower DRtg is better
            g_drtg.figure.suptitle(f'Defensive Rating Evolution - Group {group_id}', y=1.03)
            plot_path_drtg = plot_dir / f"evolution_drtg_group_{group_id}.png"
            g_drtg.savefig(plot_path_drtg, dpi=150)
            plt.close(g_drtg.figure)
            logger.info("-> Saved DRtg plot: %s", plot_path_drtg)
    except Exception as e:
        logger.error("Failed to generate/save DRtg plot for group %s: %s", group_id, e)
        if 'g_drtg' in locals() and hasattr(g_drtg, 'figure'):
            plt.close(g_drtg.figure)


# --- Plotting Function: Pairwise Heatmap ---
def plot_pairwise_heatmap(pairwise_df: pd.DataFrame, group_id: str, plot_dir: Path):
    """Generates and saves a heatmap for pairwise minutes."""
    if pairwise_df.empty:
        logger.info("Pairwise DataFrame is empty, skipping heatmap for group %s.", group_id)
        return
    if pairwise_df.shape[0] < 2 or pairwise_df.shape[1] < 2:
         logger.info("Insufficient players for pairwise heatmap for group %s.", group_id)
         return

    plot_dir.mkdir(parents=True, exist_ok=True)
    filename = plot_dir / f"heatmap_pairwise_minutes_group_{group_id}.png"

    try:
        plt.figure(figsize=(10, 8)) # Adjust size as needed
        sns.heatmap(pairwise_df, annot=True, fmt="d", cmap="viridis", linewidths=.5)
        plt.title(f'Pairwise Minutes Played Together - Group {group_id}')
        plt.xlabel("Player") # X-axis label might be redundant if player names are clear
        plt.ylabel("Player") # Y-axis label might be redundant
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        plt.savefig(filename, dpi=150)
        plt.close()
        logger.info("-> Saved Pairwise Heatmap: %s", filename)
    except Exception as e:
        logger.error("Failed to generate/save pairwise heatmap for group %s: %s", group_id, e)
        plt.close() # Attempt to close figure on error

# --- Plotting Function: Stacked Minutes Bar Chart ---
def plot_minutes_stacked(evo_df: pd.DataFrame, group_id: str, plot_dir: Path):
    """Generates and saves a stacked bar chart of minutes per player per game."""
    if evo_df.empty:
        logger.info("Evolution DataFrame is empty, skipping stacked minutes plot for group %s.", group_id)
        return

    # Pivot the table for stacked bar chart: games as index, players as columns, minutes as values
    try:
        minutes_pivot = evo_df.pivot_table(index='game_idx', columns='player', values='mins', aggfunc='sum').fillna(0)
    except Exception as e:
        logger.error("Failed to pivot evolution data for stacked minutes plot for group %s: %s", group_id, e)
        return

    if minutes_pivot.empty:
         logger.info("Pivoted minutes data is empty, skipping stacked minutes plot for group %s.", group_id)
         return

    plot_dir.mkdir(parents=True, exist_ok=True)
    filename = plot_dir / f"stacked_minutes_per_game_group_{group_id}.png"

    try:
        # Apply name shortening to columns for better display
        minutes_pivot.columns = minutes_pivot.columns.map(shorten_name)
        # Sort columns alphabetically for consistent legend order
        minutes_pivot = minutes_pivot.reindex(sorted(minutes_pivot.columns), axis=1)

        ax = minutes_pivot.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='tab20') # Adjust size and colormap
        plt.title(f'Minutes Played per Game - Group {group_id}')
        plt.xlabel("Game Index")
        plt.ylabel("Minutes Played")
        plt.xticks(rotation=0) # Keep game index horizontal
        # Place legend outside the plot
        plt.legend(title='Player', bbox_to_anchor=(1.04, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        plt.savefig(filename, dpi=150)
        plt.close()
        logger.info("-> Saved Stacked Minutes Plot: %s", filename)
    except Exception as e:
        logger.error("Failed to generate/save stacked minutes plot for group %s: %s", group_id, e)
        plt.close()

# --- Plotting Function: Player On_Net Bar Chart ---
def plot_player_on_net(onoff_df: pd.DataFrame, group_id: str, plot_dir: Path):
    """Generates and saves a bar chart for Player On_Net rating."""
    if onoff_df.empty or 'On_Net' not in onoff_df.columns:
        logger.info("On/Off DataFrame is empty or missing 'On_Net', skipping plot for group %s.", group_id)
        return

    # Ensure 'On_Net' is numeric, coercing errors
    onoff_df['On_Net'] = pd.to_numeric(onoff_df['On_Net'], errors='coerce')
    onoff_df = onoff_df.dropna(subset=['On_Net']) # Drop rows where conversion failed

    if onoff_df.empty:
        logger.info("No valid numeric 'On_Net' data found, skipping plot for group %s.", group_id)
        return

    # Sort by On_Net for plotting
    onoff_df_sorted = onoff_df.sort_values('On_Net', ascending=False)

    plot_dir.mkdir(parents=True, exist_ok=True)
    filename = plot_dir / f"barchart_player_on_net_group_{group_id}.png"

    try:
        plt.figure(figsize=(10, 6))
        barplot = sns.barplot(x='Player', y='On_Net', hue='Player', data=onoff_df_sorted, palette='coolwarm', legend=False)
        # Add value labels on bars
        for index, row in onoff_df_sorted.iterrows():
             barplot.text(index, row.On_Net + (1 if row.On_Net >= 0 else -1), # Adjust position based on value
                          f'{row.On_Net:.1f}', color='black', ha="center", va='bottom' if row.On_Net >=0 else 'top')

        plt.title(f'Player On-Court Net Rating (per 40 min) - Group {group_id}')
        plt.xlabel("Player")
        plt.ylabel("On_Net Rating")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        logger.info("-> Saved Player On_Net Bar Chart: %s", filename)
    except Exception as e:
        logger.error("Failed to generate/save player On_Net plot for group %s: %s", group_id, e)
        plt.close()

# --- Plotting Function: Lineup NetRtg Bar Chart ---
def plot_lineup_netrtg(lineup_df: pd.DataFrame, group_id: str, plot_dir: Path):
    """Generates and saves a bar chart for Lineup NetRtg."""
    if lineup_df.empty or 'NetRtg' not in lineup_df.columns:
        logger.info("Lineup DataFrame is empty or missing 'NetRtg', skipping plot for group %s.", group_id)
        return

    # Ensure 'NetRtg' is numeric
    lineup_df['NetRtg'] = pd.to_numeric(lineup_df['NetRtg'], errors='coerce')
    lineup_df = lineup_df.dropna(subset=['NetRtg'])

    if lineup_df.empty:
        logger.info("No valid numeric 'NetRtg' data found for lineups, skipping plot for group %s.", group_id)
        return

    # Sort by NetRtg and take top N (e.g., top 15) for readability
    top_n = 15
    lineup_df_sorted = lineup_df.sort_values('NetRtg', ascending=False).head(top_n)

    plot_dir.mkdir(parents=True, exist_ok=True)
    filename = plot_dir / f"barchart_lineup_netrtg_group_{group_id}.png"

    try:
        plt.figure(figsize=(12, 7)) # Wider figure for lineup names
        barplot = sns.barplot(x='lineup', y='NetRtg', hue='lineup', data=lineup_df_sorted, palette='viridis', legend=False)
        # Add value labels
        for index, row in lineup_df_sorted.iterrows():
             barplot.text(index, row.NetRtg + (1 if row.NetRtg >= 0 else -1),
                          f'{row.NetRtg:.1f}', color='black', ha="center", va='bottom' if row.NetRtg >=0 else 'top')
        plt.title(f'Top {top_n} Lineup Net Rating (per 40 min) - Group {group_id}')
        plt.xlabel("Lineup (Shortened Names)")
        plt.ylabel("Net Rating")
        plt.xticks(rotation=75, ha='right') # Rotate more for long lineup names
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(filename, dpi=150)
        plt.close()
        logger.info("-> Saved Lineup NetRtg Bar Chart: %s", filename)
    except Exception as e:
        logger.error("Failed to generate/save lineup NetRtg plot for group %s: %s", group_id, e)
        plt.close()

# --- Plotting Function: Player Comparison Radar Chart ---
def plot_player_comparison_radar(group_results: dict, player_name: str, plot_dir: Path):
    """Generates a radar chart comparing a player's stats between two groups."""
    group_ids = list(group_results.keys())
    if len(group_ids) != 2:
        logger.error("Radar plot function called with incorrect number of groups.")
        return
        
    g1_id, g2_id = group_ids[0], group_ids[1]
    g1_data = group_results[g1_id]
    g2_data = group_results[g2_id]

    # Extract player aggregate stats (assuming stored in player_df)
    try:
        player1_stats = g1_data['player_df'][g1_data['player_df']['Player'] == shorten_name(player_name)].iloc[0]
        player2_stats = g2_data['player_df'][g2_data['player_df']['Player'] == shorten_name(player_name)].iloc[0]
    except IndexError:
        logger.error(f"Could not find player '{player_name}' (shortened: {shorten_name(player_name)}) in the results for both groups. Check spelling and data.")
        # Let's log the available players for debugging
        logger.debug(f"Group {g1_id} players: {g1_data['player_df']['Player'].tolist()}")
        logger.debug(f"Group {g2_id} players: {g2_data['player_df']['Player'].tolist()}")
        return
    except KeyError as e:
        logger.error(f"Missing key '{e}' in stored group data for radar plot.")
        return
        
    # --- Select & Normalize Stats ---
    # Define categories (stats) to plot - use per-game stats where sensible
    # Note: Needs GP > 0 for division
    gp1 = player1_stats.get('GP', 0)
    gp2 = player2_stats.get('GP', 0)
    
    stats_to_plot = {
        'Mins/GP': (player1_stats.get('Mins', 0) / gp1 if gp1 > 0 else 0, player2_stats.get('Mins', 0) / gp2 if gp2 > 0 else 0),
        'Pts/GP': (player1_stats.get('PTS', 0) / gp1 if gp1 > 0 else 0, player2_stats.get('PTS', 0) / gp2 if gp2 > 0 else 0),
        # Add Plus/Minus per GP from evolution table or calculate aggregate
        # For simplicity, using aggregate +/- here if available, else 0
        # Need to calculate aggregate +/- and add to player_df first!
        # Let's omit +/- for now until it's added to player_df
        # '+/-/GP': (0, 0), 
        'T3/GP': (player1_stats.get('T3', 0) / gp1 if gp1 > 0 else 0, player2_stats.get('T3', 0) / gp2 if gp2 > 0 else 0),
        'T2/GP': (player1_stats.get('T2', 0) / gp1 if gp1 > 0 else 0, player2_stats.get('T2', 0) / gp2 if gp2 > 0 else 0),
        'T1/GP': (player1_stats.get('T1', 0) / gp1 if gp1 > 0 else 0, player2_stats.get('T1', 0) / gp2 if gp2 > 0 else 0),
        'Fouls/GP': (player1_stats.get('Fouls', 0) / gp1 if gp1 > 0 else 0, player2_stats.get('Fouls', 0) / gp2 if gp2 > 0 else 0),
    }
    
    categories = list(stats_to_plot.keys())
    n_categories = len(categories)
    
    # Extract values for each group
    values1 = [stats_to_plot[cat][0] for cat in categories]
    values2 = [stats_to_plot[cat][1] for cat in categories]
    
    # Compute max values for normalization (across both players)
    max_values = [max(stats_to_plot[cat][0], stats_to_plot[cat][1]) for cat in categories]
    # Avoid division by zero if max is 0 for a category
    max_values = [v if v > 0 else 1 for v in max_values]
    
    # Normalize values (0 to 1 scale)
    norm_values1 = [v / max_v for v, max_v in zip(values1, max_values)]
    norm_values2 = [v / max_v for v, max_v in zip(values2, max_values)]

    # --- Create Radar Chart --- 
    angles = [n / float(n_categories) * 2 * pi for n in range(n_categories)]
    angles += angles[:1] # Close the plot
    
    norm_values1 += norm_values1[:1]
    norm_values2 += norm_values2[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Set ticks and labels
    plt.xticks(angles[:-1], categories)
    ax.set_yticks(np.arange(0, 1.1, 0.2)) # Ticks from 0 to 1
    ax.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0, 1.1, 0.2)])
    ax.set_ylim(0, 1.1) # Ensure scale goes slightly beyond 1
    
    # Plot data
    ax.plot(angles, norm_values1, linewidth=2, linestyle='solid', label=f'Group {g1_id}', color='skyblue')
    ax.fill(angles, norm_values1, 'skyblue', alpha=0.4)
    ax.plot(angles, norm_values2, linewidth=2, linestyle='solid', label=f'Group {g2_id}', color='lightcoral')
    ax.fill(angles, norm_values2, 'lightcoral', alpha=0.4)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title(f'Player Comparison: {player_name}\n(Normalized Stats per GP)', size=16, y=1.1)
    
    # Save plot
    plot_dir.mkdir(parents=True, exist_ok=True)
    filename = plot_dir / f"radar_compare_{player_name.replace(' ', '_')}_groups_{g1_id}_{g2_id}.png"
    try:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"-> Saved Player Comparison Radar Chart: {filename}")
    except Exception as e:
        logger.error(f"Failed to save radar chart for player {player_name}: {e}")
        plt.close(fig)


if __name__ == "__main__":
    main()
