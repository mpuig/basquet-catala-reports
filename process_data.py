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
        self, schedule: pd.DataFrame, all_moves: Dict[str, Sequence[dict]]
    ) -> None:
        """Iterate schedule rows & their events to populate aggregates."""
        # Iterate using iterrows() to reliably access columns by name
        for index, row_series in schedule.iterrows():
            # Access columns directly from the row Series using lowercase names
            match_id = str(row_series.get("match_id", ""))
            local_id = str(row_series.get('local_team_id', ""))
            visitor_id = str(row_series.get('visitor_team_id', ""))
            
            # --- Debug row --- 
            # print(f"DEBUG: Checking row - Match: {match_id}, Local: {local_id}, Visitor: {visitor_id}, Target: {self.team_id}")
            # --- End Debug row ---

            # Skip if Match ID is missing (e.g., NaN row)
            if not match_id:
                continue
                
            events = all_moves.get(match_id, [])
            if events:
                # Pass the row Series directly to the processing function
                self._process_single_game(match_id, row_series, events)

    # ------------------------------------------------------------------
    # Private helpers – one game
    # ------------------------------------------------------------------

    def _process_single_game(
        self, match_id: str, row_series: pd.Series, events: Sequence[dict]
    ) -> None:
        """Core loop; translates original long logic into smaller steps."""
        # Get team name corresponding to the target team ID from the schedule row Series
        target_team_name = None
        local_id = str(row_series.get('local_team_id', ''))
        visitor_id = str(row_series.get('visitor_team_id', ''))
        
        if local_id == self.team_id:
            target_team_name = row_series.get('local_team')
        elif visitor_id == self.team_id:
            target_team_name = row_series.get('visitor_team')

        if not target_team_name:
            logger.debug("Target team %s not involved in match %s", self.team_id, match_id)
            return # Skip game if our team isn't in it

        # Detect internal idTeam by matching the *name* from a teamAction event
        internal_id = next((
            ev["idTeam"]
            for ev in events
            if ev.get("teamAction") and ev.get("actorName") == target_team_name
        ), None)
        
        if internal_id is None:
            logger.warning(
                "Skipping match %s for team '%s' – cannot map to internal idTeam", 
                match_id, target_team_name
            )
            return
        else:
            logger.info("Processing match %s for team '%s' (Internal ID: %s)", match_id, target_team_name, internal_id)

        on_court: set[str] = set()
        player_state: Dict[str, dict] = {}
        last_abs_sec = 0

        def flush_pairwise(duration: float) -> None:
            """Credit *duration* seconds to every pair currently on court."""
            if duration <= 0 or len(on_court) < 2:
                return
            for p1 in on_court:
                for p2 in on_court:
                    self.pairwise_secs[p1][p2] += duration

        # chronological loop
        for ev in sorted(events, key=lambda e: e.get("timestamp", "")):
            if ev.get("idTeam") != internal_id:
                continue

            # time
            period, mm, ss = ev.get("period"), ev.get("min"), ev.get("sec")
            if not all(isinstance(x, int) for x in (period, mm, ss)):
                continue
            abs_sec = get_absolute_seconds(period, mm, ss)
            flush_pairwise(abs_sec - last_abs_sec)
            last_abs_sec = abs_sec

            actor = ev.get("actorName")
            if not actor or ev.get("teamAction"):
                continue

            # ---- Store Player Number ----
            if actor not in self.players or self.players[actor].number == "??":
                num = ev.get('actorShirtNumber')
                if num is not None:
                    self.players[actor].number = str(num) # Store as string
            # ---- End Store Player Number ----

            # starters inference
            if actor not in player_state and len(on_court) < 5:
                player_state[actor] = {"status": "in", "since": (period - 1) * PERIOD_LENGTH_SEC}
                on_court.add(actor)

            move = ev.get("move", "")

            # substitutions
            if move == "Entra al camp":
                player_state[actor] = {"status": "in", "since": abs_sec}
                on_court.add(actor)
                continue
            if move == "Surt del camp" and player_state.get(actor, {}).get("status") == "in":
                played = abs_sec - player_state[actor]["since"]
                self._credit_minutes(actor, played)
                player_state[actor]["status"] = "out"
                on_court.discard(actor)
                continue

            # boxscore
            pts = self.points_map.get(move, 0)
            if pts:
                self.team_pts += pts
                pa = self.players[actor]
                pa.pts += pts
                if pts == 3:
                    pa.t3 += 1
                elif pts == 2:
                    pa.t2 += 1
                else:
                    pa.t1 += 1

            if any(k in move for k in self.foul_keywords):
                self.team_fouls += 1
                self.players[actor].fouls += 1

        # end of game
        game_end_sec = PERIOD_LENGTH_SEC * 4
        flush_pairwise(game_end_sec - last_abs_sec)
        for p, st in player_state.items():
            if st["status"] == "in":
                self._credit_minutes(p, game_end_sec - st["since"])

    # ------------------------------------------------------------------

    def _credit_minutes(self, player: str, secs: float) -> None:
        pa = self.players[player]
        pa.merge_secs(secs)
        pa.gp += 1  # crude GP increment

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
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    moves_dir = data_dir / "match_moves"

    for gid in args.groups:
        print(f"\n{'='*15} Processing Group: {gid} | Team: {args.team} {'='*15}")
        csv_path = data_dir / f"results_{gid}.csv"
        try:
            schedule_df = load_schedule(csv_path)
        except FileNotFoundError:
            logger.error("Schedule file not found for group %s at %s. Skipping group.", gid, csv_path)
            continue
        except ValueError as e:
            logger.error("Error loading schedule for group %s: %s. Skipping group.", gid, e)
            continue
            
        # Load moves only for the current group's matches
        current_group_moves: Dict[str, List[dict]] = {}
        match_ids_in_group = schedule_df["match_id"].dropna().astype(str).unique()
        logger.info("Found %d unique match IDs in schedule for group %s", len(match_ids_in_group), gid)
        for mid in match_ids_in_group:
            moves_data = load_match_moves(mid, moves_dir)
            if moves_data: # Only add if moves were successfully loaded
                 current_group_moves[mid] = moves_data
            # No need for setdefault as we start with an empty dict each time

        if not current_group_moves:
             logger.warning("No valid match moves data loaded for group %s. Cannot calculate stats.", gid)
             continue
             
        # Instantiate calculator for *this group*
        calc = StatsCalculator(args.team)
        # Process only this group's data
        calc.process(schedule_df, current_group_moves)

        print("\n---- Player aggregate ----")
        player_table = calc.player_table()
        if not player_table.empty:
            print(player_table.to_string(index=False))
        else:
            print("(No data found for this team in this group)")

        print("\n---- Pairwise minutes (first 10×10) ----")
        pair_df = calc.pairwise_minutes()
        if not pair_df.empty:
            # Limit to max 10x10 for display
            max_dim = min(10, pair_df.shape[0], pair_df.shape[1])
            print(pair_df.iloc[:max_dim, :max_dim].to_string(float_format='{:d}'.format))
        else:
             print("(No data found for this team in this group)")
        
        print(f"{'='*15} Finished Group: {gid} {'='*15}")


if __name__ == "__main__":
    main()
