"""
Unified Reports Entrypoint
-------------------------

This script combines the functionality of process_data.py (group-level stats/plots) and generate_match_reports.py (individual match HTML reports).

Usage:
    $ python reports.py --team TEAM_ID --groups GROUP_ID [GROUP_ID ...] [--data-dir ...] [--output-dir ...] [--season ...]

"""
import argparse
from pathlib import Path

import pandas as pd

from report_tools.data_loaders import load_schedule, load_match_moves, load_match_stats
from report_tools.logger import logger
from report_tools.plotting import (
    plot_pairwise_heatmap,
    plot_player_on_net,
    plot_lineup_netrtg,
    plot_score_timeline,
)
from report_tools.stats_calculator import StatsCalculator


# --- Individual match report helpers (from generate_match_reports.py) ---

def parse_args():
    parser = argparse.ArgumentParser(description="Unified group and match report generator.")
    parser.add_argument("--team", required=True, help="Team ID to analyse")
    parser.add_argument("--groups", nargs="+", required=True, help="Competition group IDs")
    parser.add_argument("--data-dir", default="data", help="Root data folder (CSV & JSON)")
    parser.add_argument("--output-dir", default="reports", help="Directory to save generated reports")
    parser.add_argument("--season", default="2024", help="Season identifier (default: 2024)")
    return parser.parse_args()


def generate_group_report(team_id: str, group_id: str, data_dir: Path, season: str, plot_dir: Path):
    """
    Run group-level stats/plots as in process_data.py for the given team and group.
    """
    # Load schedule
    csv_path = data_dir / f"results_{group_id}.csv"
    if not csv_path.exists():
        logger.warning(f"Schedule file {csv_path} not found, skipping group.")
        return None, None
    schedule_df = load_schedule(csv_path)
    # Load moves and stats for all matches in group
    moves_dir = data_dir / "match_moves"
    stats_dir = data_dir / "match_stats"
    moves = {}
    stats = {}
    for match_id in schedule_df["match_id"]:
        moves[match_id] = load_match_moves(match_id, moves_dir)
        stats[match_id] = load_match_stats(match_id, stats_dir)
    # Calculate stats
    calc = StatsCalculator(team_id)
    calc.process(schedule_df, moves, stats)

    # Generate plots (as in process_data.py)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Generate the requested plots
    # 1. Score timeline for each match
    for match_id in schedule_df["match_id"]:
        match_moves = moves.get(match_id)
        match_stats = stats.get(match_id)
        if match_moves and match_stats:
            plot_score_timeline(match_moves, match_stats, plot_dir, team_id)

    # 2. Pairwise heatmap
    pairwise_df = calc.pairwise_df
    if pairwise_df is not None:
        plot_pairwise_heatmap(pairwise_df, plot_dir)

    # 3. Player On/Off court net rating
    onoff_df = calc.onoff_df
    if onoff_df is not None:
        plot_player_on_net(onoff_df, plot_dir)

    # 4. Lineup net rating
    lineup_df = calc.lineup_df
    if lineup_df is not None:
        plot_lineup_netrtg(lineup_df, plot_dir)

    return schedule_df, calc


def generate_match_reports(team_id: str, group_id: str, schedule_df: pd.DataFrame, data_dir: Path, output_dir: Path,
                           season: str):
    """
    For each match in the group, generate the individual match HTML report as in generate_match_reports.py.
    """
    from report_tools.templates import REPORT_HTML_TEMPLATE
    from jinja2 import Environment, select_autoescape
    env = Environment(loader=None, autoescape=select_autoescape(["html", "xml"]))
    template = env.from_string(REPORT_HTML_TEMPLATE)
    moves_dir = data_dir / "match_moves"
    stats_dir = data_dir / "match_stats"
    reports = []
    for _, match_info_row in schedule_df.iterrows():
        match_id = str(match_info_row.get("match_id", ""))
        if not match_id:
            continue
        # --- Call the _generate_single_report logic from generate_match_reports.py ---
        # TODO: Consider deduplicating this logic into a shared function/module
        from generate_match_reports import _generate_single_report
        report_data = _generate_single_report(
            match_info_row=match_info_row,
            target_team_id=team_id,
            gid=group_id,
            moves_dir=moves_dir,
            stats_dir=stats_dir,
            output_dir=output_dir,
            template=template,
            season=season,
        )
        if report_data:
            reports.append(report_data)
    # Optionally generate index.html for the group
    from generate_match_reports import generate_index_page
    if reports:
        generate_index_page(reports, output_dir, team_id)
    return reports


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    team_id = args.team
    season = args.season
    for group_id in args.groups:
        logger.info(f"Processing group {group_id} for team {team_id}")
        schedule_df, calc = generate_group_report(team_id, group_id, data_dir, season, plot_dir)
        if schedule_df is None:
            continue
        logger.info(f"Generating individual match reports for group {group_id}")
        generate_match_reports(team_id, group_id, schedule_df, data_dir, output_dir, season)


if __name__ == "__main__":
    main()
