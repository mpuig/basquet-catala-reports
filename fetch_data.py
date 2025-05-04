"""
Data‐collector for Catalan Basketball Federation website
-------------------------------------------------------

Features
--------
* Download schedule (per group) and save to CSV
* Download play‑by‑play, aggregate stats, team stats and player stats JSONs
* Minimal external deps: requests, beautifulsoup4, pandas

CLI examples
------------
1) Fetch only schedules
   $ python fetch_data.py --groups 17182 18299 --mode schedule

2) Fetch everything for a season
   $ python fetch_data.py --groups 17182 18299 --season 2024 --mode all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup

###############################################################################
# logging                                                                     #
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("fetch_data")

###############################################################################
# constants                                                                   #
###############################################################################

BASE_URL = "https://www.basquetcatala.cat/competicions/resultats/{group_id}/0"
MATCH_MOVES_URL = (
    "https://msstats.optimalwayconsulting.com/v1/fcbq/"
    "getJsonWithMatchMoves/{match_id}?currentSeason=true"
)
MATCH_STATS_URL = (
    "https://msstats.optimalwayconsulting.com/v1/fcbq/"
    "getJsonWithMatchStats/{match_id}?currentSeason=true"
)
TEAM_STATS_URL = (
    "https://msstats.optimalwayconsulting.com/v1/fcbq/"
    "team-stats/team/{team_id}/season/{season_id}"
)
PLAYER_STATS_URL = (
    "https://msstats.optimalwayconsulting.com/v1/fcbq/"
    "player-stats/federated/{player_uuid}/team/{team_id}"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; basquetcatala-bot/1.0; +https://github.com)"
}

###############################################################################
# dataclass                                                                   #
###############################################################################


@dataclass
class MatchInfo:
    jornada: str
    date_time: str
    local_team: str
    local_team_id: str
    visitor_team: str
    visitor_team_id: str
    score: str
    match_id: str

    @staticmethod
    def header() -> List[str]:
        return list(asdict(MatchInfo("", "", "", "", "", "", "", "")).keys())

    def to_list(self) -> List[str]:
        return list(asdict(self).values())


###############################################################################
# network helpers                                                             #
###############################################################################


def get_soup(url: str) -> BeautifulSoup:
    log.info("Fetching %s", url)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.content, "html.parser")


def save_json(url: str, path: Path) -> None:
    if path.exists():
        log.debug("File exists, skipping %s", path.name)
        return
    log.info("→ %s", url)
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    os.makedirs(path.parent, exist_ok=True)
    path.write_text(resp.text, encoding="utf-8")


###############################################################################
# parsing schedule                                                            #
###############################################################################


def parse_group_schedule(group_id: str) -> List[MatchInfo]:
    soup = get_soup(BASE_URL.format(group_id=group_id))
    matches: List[MatchInfo] = []

    jornada_headers = soup.find_all(
        "div", class_=re.compile(r"bg-2 pd-5 ff-1 fs-16 c-5")
    )
    if not jornada_headers:
        log.warning("No jornada headers found for group %s", group_id)
        return matches

    for header in jornada_headers:
        jornada_txt = header.get_text(strip=True)
        if jornada_txt == "Equips del grup":
            continue
        jornada_match = re.search(r"Jornada\s*(\d+)", jornada_txt)
        jornada = jornada_match.group(1) if jornada_match else jornada_txt

        container = header.find_next("div", class_="container m-bottom")
        rows_div = container.find("div", class_="rowsJornada") if container else None
        if not rows_div:
            log.warning("No rowsJornada div for jornada %s", jornada)
            continue

        for row in rows_div.find_all("div", recursive=False):
            # detect 'Descansa'
            if (
                "rowJornada" in row.get("class", [])
                and "col-md-12" in row.get("class", [])
                and "Descansa" in row.get_text()
            ):
                team_link = row.find("a", class_="teamNameLink")
                team = team_link.get_text(strip=True) if team_link else row.get_text(strip=True)
                team_id = team_link["href"].strip("/").split("/")[-1] if team_link else "N/A"
                matches.append(
                    MatchInfo(jornada, "", team, team_id, "Descansa", "N/A", "", "N/A")
                )
                continue

            fila = row if row.get("id") != "fila" else row.find("div", class_="rowJornada")
            if not fila or "rowJornada" not in fila.get("class", []):
                continue

            main_col = fila.find("div", class_=re.compile(r"col-md-10"))
            if not main_col:
                continue

            info_row = main_col.find("div", class_=re.compile(r"rowJornada col-md-12"))
            score_row = main_col.find("div", class_=re.compile(r"rowJornada fs-38"))
            if not info_row or not score_row:
                continue

            cols = info_row.find_all("div", class_=re.compile(r"col-md-4"))
            if len(cols) != 3:
                continue

            # local
            local_link = cols[0].find("a", class_="teamNameLink")
            local_team = local_link.get_text(strip=True) if local_link else cols[0].get_text(strip=True)
            local_team_id = local_link["href"].strip("/").split("/")[-1] if local_link else "N/A"

            # date/time
            date_div = cols[1].find("div", id="time2")
            date_time = date_div.get_text(strip=True) if date_div else cols[1].get_text(strip=True)

            # visitor
            visitor_link = cols[2].find("a", class_="teamNameLink")
            visitor_team = visitor_link.get_text(strip=True) if visitor_link else cols[2].get_text(strip=True)
            visitor_team_id = visitor_link["href"].strip("/").split("/")[-1] if visitor_link else "N/A"

            # score and match id
            sc_cols = score_row.find_all("div", class_=re.compile(r"col-md-4"))
            score = ""
            match_id = "N/A"
            if len(sc_cols) == 3:
                score = f"{sc_cols[0].get_text(strip=True)}-{sc_cols[2].get_text(strip=True)}"
                stats_img = sc_cols[1].find("img", title="Estadística")
                if stats_img and stats_img.find_parent("a"):
                    match_id = stats_img.find_parent("a")["href"].strip("/").split("/")[-1]

            matches.append(
                MatchInfo(
                    jornada,
                    date_time,
                    local_team,
                    local_team_id,
                    visitor_team,
                    visitor_team_id,
                    score,
                    match_id,
                )
            )

    log.info("Parsed %d matches for group %s", len(matches), group_id)
    return matches


###############################################################################
# write helpers                                                                #
###############################################################################


def write_schedule_csv(matches: List[MatchInfo], group_id: str, out_dir: Path) -> None:
    if not matches:
        log.warning("No matches – skipping CSV for group %s", group_id)
        return
    os.makedirs(out_dir, exist_ok=True)
    csv_path = out_dir / f"results_{group_id}.csv"
    df = pd.DataFrame([m.to_list() for m in matches], columns=MatchInfo.header())
    df.to_csv(csv_path, index=False, encoding="utf-8")
    log.info("→ CSV saved %s", csv_path)


###############################################################################
# JSON download orchestration                                                  #
###############################################################################


def download_json_assets(matches: List[MatchInfo], season_id: str, out_root: Path) -> None:
    moves_dir = out_root / "match_moves"
    stats_dir = out_root / "match_stats"
    team_dir = out_root / "team_stats"
    player_dir = out_root / "player_stats"

    # moves & match stats
    for m in matches:
        if m.match_id != "N/A":
            try:
                save_json(
                    MATCH_MOVES_URL.format(match_id=m.match_id),
                    moves_dir / f"{m.match_id}.json",
                )
                save_json(
                    MATCH_STATS_URL.format(match_id=m.match_id),
                    stats_dir / f"{m.match_id}.json",
                )
            except requests.exceptions.RequestException as e:
                log.warning("Failed to download JSON for match %s: %s", m.match_id, e)

    # team stats
    team_ids = set()
    for m in matches:
        if m.local_team_id != "N/A":
            team_ids.add(m.local_team_id)
        if m.visitor_team_id != "N/A":
            team_ids.add(m.visitor_team_id)

    for tid in team_ids:
        try:
            save_json(
                TEAM_STATS_URL.format(team_id=tid, season_id=season_id),
                team_dir / f"team_{tid}_season_{season_id}.json",
            )
        except requests.exceptions.RequestException as e:
            log.warning("Failed to download team stats for team %s, season %s: %s", tid, season_id, e)

    # player stats (requires team stats)
    player_uuid_team_map = {}
    for team_file in team_dir.glob(f"team_*_season_{season_id}.json"):
        try:
            team_data = json.loads(team_file.read_text(encoding="utf-8"))
            current_team_id = team_file.stem.split("_")[1]
            for player in team_data.get("players", []):
                if player_uuid := player.get("uuid"):
                    player_uuid_team_map[player_uuid] = current_team_id
        except (FileNotFoundError, json.JSONDecodeError, IndexError) as e:
            log.warning("Could not process team file %s: %s", team_file.name, e)

    # --- Debug Player Map ---
    log.info(f"Found {len(player_uuid_team_map)} player UUIDs to fetch stats for.")
    if not player_uuid_team_map:
        log.warning("Player UUID map is empty. No player stats will be downloaded.")
    # --- End Debug Player Map ---

    for uuid, team_id in player_uuid_team_map.items():
        try:
            save_json(
                PLAYER_STATS_URL.format(player_uuid=uuid, team_id=team_id),
                player_dir / f"player_{uuid}_team_{team_id}.json",  # Store with team id too
            )
        except requests.exceptions.RequestException as e:
            log.warning("Failed to download player stats for player %s (team %s): %s", uuid, team_id, e)


###############################################################################
# cli                                                                         #
###############################################################################


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch data from basquetcatala.cat")
    parser.add_argument("--groups", nargs="+", required=True, help="Competition group IDs")
    parser.add_argument("--mode", choices=["schedule", "all"], default="schedule")
    parser.add_argument("--season", default="2024", help="Season id for team/player stats")
    parser.add_argument("--output", default="data", help="Output directory")
    args = parser.parse_args()

    out_root = Path(args.output)
    for gid in args.groups:
        log.info("=== Group %s ===", gid)
        matches = parse_group_schedule(gid)
        write_schedule_csv(matches, gid, out_root)

        if args.mode == "all":
            download_json_assets(matches, args.season, out_root)

    log.info("Done.")


if __name__ == "__main__":
    main()
