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
from typing import Dict, List, Sequence, Set

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
            # --- On/Off tracking -------------------------------------------------
            delta = event_abs_seconds - last_event_abs_seconds
            if delta > 0:
                for p in on_court:
                    self.on_secs[p] += delta

            pts = self.points_map.get(event_type, 0)
            if pts:
                if is_target_team_event:
                    self.team_pts_f += pts
                    for p in on_court:
                        self.on_pts_f[p] += pts
                else:
                    self.team_pts_a += pts
                    for p in on_court:
                        self.on_pts_a[p] += pts
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
                elif event_type in ["Cistella de 2", "Esmaixada"]:
                    pa.t2 += 1
                    pa.pts += 2
                elif event_type == "Cistella de 1": # Correct event for FT made
                    pa.t1 += 1
                    pa.pts += 1
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
        # Credit remaining time for players still on court
        for p, st in player_state.items():
            if st["status"] == "in":
                duration = game_end_sec - st["since"]
                self._credit_minutes(p, duration)
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


    # ------------------------------------------------------------------
    # On/Off Net Rating table
    # ------------------------------------------------------------------
    def on_off_table(self) -> pd.DataFrame:
        rows = []
        total_secs = sum(self.on_secs.values())
        if total_secs == 0:
            return pd.DataFrame()  # no data

        for player, secs_on in self.on_secs.items():
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
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    moves_dir = data_dir / "match_moves"
    stats_dir = data_dir / "match_stats" 
    team_stats_dir = data_dir / "team_stats" # Directory for team stats

    # --- Try to load team info for headers --- 
    team_name_for_header = args.team # Default to ID
    group_name_for_header = "Unknown Group" # Default group name
    team_stats_data = load_team_stats(args.team, args.season, team_stats_dir)
    if team_stats_data and "team" in team_stats_data:
        team_name_for_header = team_stats_data["team"].get("teamName", args.team)
        group_name_for_header = team_stats_data["team"].get("categoryName", "Unknown Group")
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
        print_results(calc)

    else:
        # --- Process Multiple Groups Separately ---
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

            # Display results *for this group*
            print_results(calc)
            
            print(f"\n{'='*15} Finished Group: {gid} {'='*15}") # Mark end of group output

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

    print("\n==== On/Off Net Rating ====")
    onoff_df = calc.on_off_table()
    if not onoff_df.empty:
        print(onoff_df.to_string(index=False))
    else:
        print("(Insufficient data for On/Off calculation)")


if __name__ == "__main__":
    main()
