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
from typing import Dict, List, Sequence

import pandas as pd

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
                # Reset game clock time for the new period? Assuming minute resets to 0? 
                # The data shows minute continues across periods (e.g. minute 15 is in period 2)
                # We need to be careful how absolute time is calculated. 
                # Simplest: assume 10 min (600s) per period before the current one.
                current_period_start_sec = (period - 1) * 600.0 

            # event_abs_seconds = current_period_start_sec + (minute * 60.0) + second
            # Recalculate absolute seconds based on period * duration - remaining time
            # Assumes period duration is 10 mins (600s)
            # Time remaining in period = (10 - minute) * 60 - second
            # Time elapsed in game = (period-1)*600 + (600 - time_remaining) 
            # Simplified: elapsed = (period-1)*600 + minute*60 + second (This seems correct based on data)
            event_abs_seconds = ((period - 1) * 600.0) + (minute * 60.0) + second
            last_event_abs_seconds = event_abs_seconds
            game_end_sec = max(game_end_sec, event_abs_seconds) # Keep track of game end time

            if is_target_team_event and actor_name:
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
                elif event_type in ["Cistella de 2", "Esmaixada"]:
                    pa.t2 += 1
                    pa.pts += 2
                elif event_type == "Tir lliure convertit": # Free Throw Made
                    pa.t1 += 1
                    pa.pts += 1
                elif event_type == "Falta Personal":
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
                        # Credit pairwise time for player leaving
                        on_court.remove(actor_name)
                        for other_player in on_court:
                             self._credit_pairwise_secs(actor_name, other_player, duration)
                        st["status"] = "out"
                        st["since"] = event_abs_seconds

        # --- End of Game Processing --- 
        # Credit remaining time for players still on court
        for p, st in player_state.items():
            if st["status"] == "in":
                duration = game_end_sec - st["since"]
                self._credit_minutes(p, duration)
                # Credit remaining pairwise time
                temp_on_court = on_court.copy()
                temp_on_court.remove(p)
                for other_player in temp_on_court:
                     self._credit_pairwise_secs(p, other_player, duration)

        # Increment GP for players who participated in this game
        if player_state: # Only increment if the target team played and had events
            for player_name in player_state: 
                self.players[player_name].gp += 1
        
        return True # Indicate successful processing

    # ------------------------------------------------------------------

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

    def player_table(self) -> pd.DataFrame:
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
        }
        for name, aggr in sorted(
            self.players.items(), key=lambda it: it[1].pts, reverse=True
        ):
            data["Player"].append(shorten_name(name))
            data["#"].append(aggr.number)
            data["GP"].append(aggr.gp)
            data["Mins"].append(aggr.minutes)
            data["PTS"].append(aggr.pts)
            data["T3"].append(aggr.t3)
            data["T2"].append(aggr.t2)
            data["T1"].append(aggr.t1)
            data["Fouls"].append(aggr.fouls)
        return pd.DataFrame(data)

    def pairwise_minutes(self) -> pd.DataFrame:
        # Get players sorted by total minutes played (descending)
        sorted_players_by_minutes = sorted(
            self.players.items(), 
            key=lambda item: item[1].minutes, 
            reverse=True
        )
        sorted_names = [name for name, _ in sorted_players_by_minutes]
        
        # Create matrix using the sorted names
        matrix = pd.DataFrame(index=sorted_names, columns=sorted_names, dtype=int)
        for p1 in sorted_names:
            for p2 in sorted_names:
                matrix.loc[p1, p2] = int(round(self.pairwise_secs[p1].get(p2, 0) / 60))
        
        # Rename index and columns to shortened names
        matrix.index = matrix.index.map(shorten_name)
        matrix.columns = matrix.columns.map(shorten_name)
        
        # Ensure all values are integers before returning
        return matrix.astype(int)


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
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    moves_dir = data_dir / "match_moves"
    stats_dir = data_dir / "match_stats" # Directory for aggregated stats

    if args.match:
        # --- Process Single Match --- 
        logger.info(f"Processing single match ID: {args.match} for Team ID: {args.team}")
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
        print_results(calc)

    else:
        # --- Process Multiple Groups (original logic, but adapted) ---
        all_schedule_frames: list[pd.DataFrame] = []
        all_moves_data: Dict[str, List[dict]] = {}
        all_stats_data: Dict[str, dict] = {} # Store loaded match stats

        for gid in args.groups:
            print(f"\n{'='*15} Loading Group: {gid} {'='*15}")
            csv_path = data_dir / f"results_{gid}.csv"
            try:
                schedule_df = load_schedule(csv_path)
                all_schedule_frames.append(schedule_df)
                # Pre-load moves and stats for all matches in this schedule
                for mid in schedule_df["match_id"].dropna().astype(str).unique():
                    if mid not in all_moves_data:
                         loaded_moves = load_match_moves(mid, moves_dir)
                         if loaded_moves:
                             all_moves_data.setdefault(mid, loaded_moves)
                    if mid not in all_stats_data:
                        loaded_stats = load_match_stats(mid, stats_dir) # Load stats
                        if loaded_stats:
                            all_stats_data.setdefault(mid, loaded_stats)

            except FileNotFoundError:
                logger.error("Schedule file not found: %s", csv_path)
                continue # Skip to next group if schedule is missing
            except ValueError as e:
                logger.error("Error loading schedule %s: %s", csv_path, e)
                continue
            except Exception as e:
                 logger.error("Unexpected error processing schedule %s: %s", csv_path, e)
                 continue
        
        if not all_schedule_frames:
            logger.error("No schedule data loaded successfully. Exiting.")
            return

        # Combine all loaded schedule data
        combined_schedule_df = pd.concat(all_schedule_frames, ignore_index=True)

        print(f"\n{'='*15} Processing All Loaded Data | Team: {args.team} {'='*15}")
        calc = StatsCalculator(args.team)
        calc.process(combined_schedule_df, all_moves_data, all_stats_data) # Pass stats data

        # Display results for the combined data
        print_results(calc)

def print_results(calc: 'StatsCalculator'):
    """Helper function to print the standard output tables."""
    if not calc.players:
        print("\n(No data found for this team in the processed match(es))")
        return

    print("\n==== Player Aggregates ====")
    player_df = calc.player_table()
    if not player_df.empty:
        print(player_df.to_string(index=False))
    else:
        print("(No data found for this team in the processed match(es))")

    print("\n==== Pairwise minutes (first 10×10) ====")
    pair_df = calc.pairwise_minutes()
    if not pair_df.empty:
        # Limit to max 10x10 for display
        max_dim = min(10, pair_df.shape[0], pair_df.shape[1])
        print(pair_df.iloc[:max_dim, :max_dim].to_string(float_format='{:d}'.format))
    else:
        print("(No data found for this team in the processed match(es))")


if __name__ == "__main__":
    main()
