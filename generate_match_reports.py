"""
Generate Individual Match HTML Reports
--------------------------------------

This script generates detailed HTML reports for individual basketball matches
based on data collected via `fetch_data.py`.

For each match involving the target team:
1. Calculates per-match stats (aggregates, on/off, lineups)
2. Generates a markdown summary with team stat comparison
3. (Optional) Creates an LLM-generated narrative summary (cached)
4. Produces visual plots (PNG): score timeline, heatmaps, charts
5. Combines components into `report.html`
6. Generates a top-level `index.html` listing all reports

Dependencies:
- Python 3.10+
- pandas >= 2.0
- matplotlib, seaborn, numpy
- jinja2, markdown2
- (Optional) litellm with API key (e.g., OPENAI_API_KEY)

CLI Usage Examples:
- Basic:
    $ python generate_match_reports.py --team 69630 --groups 17182 18299
- Custom paths:
    $ python generate_match_reports.py --team 69630 --groups 17182 --data-dir ../data --output-dir ../output/reports
- Change season tag:
    $ python generate_match_reports.py --team 69630 --groups 17182 --season 2023
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any, Optional

import pandas as pd
from jinja2 import Environment, select_autoescape, Template

from report_tools.data_loaders import (
    load_schedule,
    load_match_moves,
    load_team_stats,
    load_match_stats,
)
from report_tools.llm import generate_llm_summary
from report_tools.logger import logger
from report_tools.plotting import (
    plot_score_timeline,
    plot_pairwise_heatmap,
    plot_player_on_net,
    plot_lineup_netrtg,
)
from report_tools.stats_calculator import StatsCalculator
from report_tools.templates import REPORT_HTML_TEMPLATE, INDEX_HTML_TEMPLATE


def _match_team_stats(
    stats: dict, target_team_id: str
) -> tuple[Optional[dict], Optional[dict]]:
    teams = stats.get("teams", [])
    if len(teams) != 2:
        return None, None
    if str(teams[0].get("teamIdExtern")) == target_team_id:
        return teams[0], teams[1]
    elif str(teams[1].get("teamIdExtern")) == target_team_id:
        return teams[1], teams[0]
    return None, None


def generate_summary_md(
    match_info: pd.Series, match_stats: dict, target_team_id: str
) -> str:
    """Generates a markdown summary of the match."""

    local_name = match_info.get("local_team", "Local")
    visitor_name = match_info.get("visitor_team", "Visitor")
    local_id = str(match_info.get("local_team_id", ""))
    match_id = match_info.get("match_id", "Unknown")

    is_target_local = target_team_id == local_id
    target_team_name = local_name if is_target_local else visitor_name
    opponent_team_name = visitor_name if is_target_local else local_name

    summary_lines = []

    target_stats, opponent_stats = _match_team_stats(match_stats, target_team_id)
    if target_stats and opponent_stats:
        ts_data = target_stats.get("data", {}) if target_stats else {}
        os_data = opponent_stats.get("data", {}) if opponent_stats else {}
        summary_lines.append("## Team Stat Comparison")
        summary_lines.append(
            f"| Stat         | {target_team_name} | {opponent_team_name} |"
        )
        summary_lines.append(
            "|:-------------|-------------------:|-------------------:|"
        )
        rows = [
            ("Points", ts_data.get("score", "?"), os_data.get("score", "?")),
            (
                "T2 Made/Att",
                f"{ts_data.get('shotsOfTwoSuccessful', '?')}/{ts_data.get('shotsOfTwoAttempted', '?')}",
                f"{os_data.get('shotsOfTwoSuccessful', '?')}/{os_data.get('shotsOfTwoAttempted', '?')}",
            ),
            (
                "T3 Made/Att",
                f"{ts_data.get('shotsOfThreeSuccessful', '?')}/{ts_data.get('shotsOfThreeAttempted', '?')}",
                f"{os_data.get('shotsOfThreeSuccessful', '?')}/{os_data.get('shotsOfThreeAttempted', '?')}",
            ),
            (
                "T1 Made/Att",
                f"{ts_data.get('shotsOfOneSuccessful', '?')}/{ts_data.get('shotsOfOneAttempted', '?')}",
                f"{os_data.get('shotsOfOneSuccessful', '?')}/{os_data.get('shotsOfOneAttempted', '?')}",
            ),
            ("Fouls", ts_data.get("faults", "?"), os_data.get("faults", "?")),
        ]
        for label, tval, oval in rows:
            summary_lines.append(f"| {label:<13} | {tval} | {oval} |")
        summary_lines.append("")
    else:
        logger.warning(f"Could not properly match teams in match_stats for {match_id}")

    return "\n".join(summary_lines)


@dataclass
class Scores:
    avg_ppg: float
    avg_t1: float
    avg_t2: float
    avg_t3: float
    avg_fouls: float
    score: int
    t1: int
    t2: int
    t3: int
    faults: int


def build_scores_from_dict(team_info, stats_match) -> Scores:
    stats = stats_match.get("data", {})

    return Scores(
        avg_ppg=round(team_info.get("totalScoreAvgByMatch"), 1),
        avg_t1=round(team_info.get("shotsOfOneSuccessfulAvgByMatch"), 1),
        avg_t2=round(team_info.get("shotsOfTwoSuccessfulAvgByMatch"), 1),
        avg_t3=round(team_info.get("shotsOfThreeSuccessfulAvgByMatch"), 1),
        avg_fouls=round(team_info.get("totalFoulsAvgByMatch"), 1),
        score=stats.get('score', '?'),
        t1=stats.get('shotsOfOneSuccessful', '?'),
        t2=stats.get('shotsOfTwoSuccessful', '?'),
        t3=stats.get('shotsOfThreeSuccessful', '?'),
        faults=stats.get('faults', '?'),
    )


def _generate_single_report(
    match_info_row: pd.Series,
    target_team_id: str,
    gid: str,
    moves_dir: Path,
    stats_dir: Path,
    output_dir: Path,
    template: Template,
    season: str,
) -> Optional[dict]:
    """Loads data, calculates stats, generates components, and renders HTML for ONE match."""
    match_id = str(match_info_row.get("match_id", ""))
    logger.info(f"Processing Match ID: {match_id} (Group: {gid})")

    match_output_dir = output_dir / f"match_{match_id}"
    match_output_dir.mkdir(parents=True, exist_ok=True)

    match_moves = load_match_moves(match_id, moves_dir)
    match_stats = load_match_stats(match_id, stats_dir)

    if not match_moves:
        logger.warning(
            f"Missing moves JSON for match {match_id}. Cannot generate full report."
        )
        return None
    if not match_stats:
        logger.warning(
            f"Missing stats JSON for match {match_id}. Cannot generate full report."
        )
        return None

    # --- Determine Opponent and Load Seasonal Stats ---
    local_team_id = str(int(match_info_row.get("local_team_id", "")))
    visitor_team_id = str(int(match_info_row.get("visitor_team_id", "")))
    local_name = str(match_info_row.get("local_team", ""))
    visitor_name = str(match_info_row.get("visitor_team", ""))
    match_date = match_info_row.get("date_time", "Unknown Date")
    score = match_info_row.get("score", "-")

    team_comparison_html = build_team_comparison_html(local_name, local_team_id, match_stats, moves_dir, season,
                                                      visitor_name,
                                                      visitor_team_id)
    # 2. Generate Team Player Aggregates HTMLs
    local_calc = calculate_stats(local_team_id, match_id, match_info_row, match_moves, match_stats)

    local_table_html = local_calc.player_table().to_html(
        index=False, classes="stats-table", border=0
    )

    visitor_calc = calculate_stats(visitor_team_id, match_id, match_info_row, match_moves, match_stats)
    visitor_table_html = visitor_calc.player_table().to_html(
        index=False, classes="stats-table", border=0
    )

    # 5. Other Tables (On/Off, Lineups)
    target_calc = calculate_stats(target_team_id, match_id, match_info_row, match_moves, match_stats)
    target_player_df = target_calc.player_table()

    onoff_df = target_calc.on_off_table()
    lineup_df = target_calc.lineup_table()
    on_off_table_html = onoff_df.to_html(index=False, classes="stats-table", border=0)
    lineup_table_html = lineup_df.head(5).to_html(index=False, classes="stats-table", border=0)

    # 6. LLM Summary
    llm_summary_text = build_llm_summary_text(match_id, match_info_row, match_moves, match_output_dir, match_stats,
                                              target_player_df, target_team_id)

    # 7. Charts
    score_timeline_path_rel = plot_score_timeline(
        match_moves if match_moves else [],
        match_stats,
        match_output_dir,
        target_team_id,
    )
    pairwise_heatmap_path_rel = plot_pairwise_heatmap(
        local_calc.pairwise_minutes(), match_output_dir
    )
    on_net_chart_path_rel = plot_player_on_net(onoff_df, match_output_dir)
    lineup_chart_path_rel = plot_lineup_netrtg(lineup_df, match_output_dir)

    # Rendering
    is_target_local = target_team_id == local_team_id
    target_team_name = local_name if is_target_local else visitor_name
    group_name = f"Group {gid}"

    try:
        html_content = template.render(
            match_id=match_id,
            group_name=group_name,
            match_date=match_date,
            target_team_name=target_team_name,
            local_name=local_name,
            visitor_name=visitor_name,
            team_comparison_html=team_comparison_html,
            local_table_html=local_table_html,
            visitor_table_html=visitor_table_html,
            on_off_table_html=on_off_table_html,
            lineup_table_html=lineup_table_html,
            score_timeline_path=score_timeline_path_rel,
            pairwise_heatmap_path=pairwise_heatmap_path_rel,
            on_net_chart_path=on_net_chart_path_rel,
            lineup_chart_path=lineup_chart_path_rel,
            season=season,
            llm_summary=llm_summary_text,
        )
        report_html_path = match_output_dir / "report.html"
        report_html_path.write_text(html_content, encoding="utf-8")
        logger.info(f"-> Successfully generated report: {report_html_path}")
    except Exception as e:
        logger.error(f"Error rendering or writing HTML for match {match_id}: {e}")
        return None

    return {
        "match_id": match_id,
        "local_name": local_name,
        "visitor_name": visitor_name,
        "match_date": match_date,
        "group_name": group_name,
        "score": score,
        "report_path": str(report_html_path.relative_to(output_dir)),
    }


def build_llm_summary_text(match_id, match_info_row, match_moves, match_output_dir, match_stats, target_player_df,
                           target_team_id):
    llm_summary_text = None
    llm_summary_md_path = match_output_dir / "llm_summary.md"
    if llm_summary_md_path.exists():
        try:
            llm_summary_text = llm_summary_md_path.read_text(encoding="utf-8").strip()
            if not llm_summary_text:
                llm_summary_text = None
        except Exception:
            llm_summary_text = None
    if llm_summary_text is None and match_moves:
        logger.info("LLM summary cache not found or invalid. Generating...")
        llm_summary_text = generate_llm_summary(
            match_info_row,
            match_stats,
            target_team_id,
            target_player_df,
        )
        if llm_summary_text:
            try:
                llm_summary_md_path.write_text(llm_summary_text, encoding="utf-8")
            except Exception as e:
                logger.error(
                    f"Error writing LLM summary cache {llm_summary_md_path}: {e}"
                )
    elif not match_moves:
        logger.warning(
            f"Skipping LLM summary generation for {match_id} due to missing moves data."
        )
    return llm_summary_text


def calculate_stats(team_id, match_id, match_info_row, match_moves, match_stats):
    calc = StatsCalculator(team_id)
    single_match_schedule_df = pd.DataFrame([match_info_row])
    calc.process(
        single_match_schedule_df,
        {match_id: match_moves} if match_moves else {},
        {match_id: match_stats},
    )
    return calc


def build_team_comparison_html(local_name, local_team_id, match_stats, moves_dir, season, visitor_name,
                               visitor_team_id):
    team1_stats = match_stats["teams"][0]
    team2_stats = match_stats["teams"][1]
    local_stats_match = None
    visitor_stats_match = None
    if str(team1_stats.get("teamIdExtern")) == local_team_id:
        local_stats_match = team1_stats
        visitor_stats_match = team2_stats
    elif str(team2_stats.get("teamIdExtern")) == local_team_id:
        local_stats_match = team2_stats
        visitor_stats_match = team1_stats
    team_stats_dir = moves_dir.parent / "team_stats"
    local_stats_data = load_team_stats(local_team_id, season, team_stats_dir)
    local = build_scores_from_dict(local_stats_data["team"], local_stats_match)
    visitor_stats_data = load_team_stats(visitor_team_id, season, team_stats_dir)
    visitor = build_scores_from_dict(visitor_stats_data["team"], visitor_stats_match)
    rows = [
        f"<tr><th>Stat</th><th>{local_name}</th><th>{visitor_name}</th></tr>",
        f"<tr><td>Points</td><td>{local.score} (avg. {local.avg_ppg})</td><td>{visitor.score} (avg. {visitor.avg_ppg})</td></tr>",
        f"<tr><td>T2</td><td>{local.t2} (avg. {local.avg_t2})</td><td>{visitor.t2} (avg. {visitor.avg_t2})</td></tr>",
        f"<tr><td>T3</td><td>{local.t3} (avg. {local.avg_t3})</td><td>{visitor.t3} (avg. {visitor.avg_t3})</td></tr>",
        f"<tr><td>T1</td><td>{local.t1} (avg. {local.avg_t1})</td><td>{visitor.t1} (avg. {visitor.avg_t1})</td></tr>",
        f"<tr><td>Fouls</td><td>{local.faults} (avg. {local.avg_fouls})</td><td>{visitor.faults} (avg. {visitor.avg_fouls})</td></tr>",
    ]
    return f"<h2>Team Stat Comparison</h2><table class='comparison-table'>{''.join(rows)}</table>"


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def load_and_validate_schedule(group_id: str, data_dir: Path) -> Optional[pd.DataFrame]:
    csv_path = data_dir / f"results_{group_id}.csv"
    if not csv_path.exists():
        logger.warning(
            f"Schedule file not found: {csv_path}. Skipping group {group_id}."
        )
        return None
    try:
        return load_schedule(csv_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error(
            f"Error loading schedule {csv_path}: {e}. Skipping group {group_id}."
        )
        return None


def _normalize_team_id(raw_id: Any) -> str:
    try:
        return str(int(float(raw_id)))
    except (ValueError, TypeError):
        return str(raw_id)


def process_group(
    group_id: str,
    schedule_df: pd.DataFrame,
    args: argparse.Namespace,
    moves_dir: Path,
    stats_dir: Path,
    output_dir: Path,
    env: Environment,
    template: Template,
    target_team_id: str,
) -> List[dict]:
    reports = []
    for _, match_info_row in schedule_df.iterrows():
        match_id = str(match_info_row.get("match_id", ""))
        if not match_id:
            continue

        local_team_id = _normalize_team_id(match_info_row.get("local_team_id", ""))
        visitor_team_id = _normalize_team_id(match_info_row.get("visitor_team_id", ""))

        if target_team_id not in (local_team_id, visitor_team_id):
            continue

        report_data = _generate_single_report(
            match_info_row=match_info_row,
            target_team_id=target_team_id,
            gid=group_id,
            moves_dir=moves_dir,
            stats_dir=stats_dir,
            output_dir=output_dir,
            template=template,
            season=args.season,
        )

        if report_data:
            reports.append(report_data)

    return reports


def _index_sort_key(report: dict) -> pd.Timestamp:
    return pd.to_datetime(
        report.get("match_date", "1900-01-01"),
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )


def generate_index_page(
    report_links_data: List[dict], output_dir: Path, target_team_id: str
) -> None:
    logger.info("Generating index.html...")
    try:
        html_template = Environment(
            autoescape=select_autoescape(["html", "xml"])
        ).from_string(INDEX_HTML_TEMPLATE)
        sorted_reports = sorted(report_links_data, key=_index_sort_key)
        html_content = html_template.render(
            reports=sorted_reports, target_team_id=target_team_id
        )
    except Exception as e:
        logger.error(f"Failed to render index HTML: {e}")
        return

    index_html = output_dir / "index.html"
    try:
        index_html.write_text(html_content, encoding="utf-8", errors="replace")
        logger.info(f"Successfully generated index file: {index_html}")
    except Exception as e:
        logger.error(f"Failed to write index.html to disk: {e}")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    moves_dir = data_dir / "match_moves"
    stats_dir = data_dir / "match_stats"
    target_team_id = args.team

    env = Environment(loader=None, autoescape=select_autoescape(["html", "xml"]))
    template = env.from_string(REPORT_HTML_TEMPLATE)

    logger.info(f"Starting report generation for Team ID: {target_team_id}")
    logger.info(f"Data Source: {data_dir.resolve()}")
    logger.info(f"Output Directory: {output_dir.resolve()}")

    total_processed = 0
    total_skipped = 0
    all_report_links = []

    for gid in args.groups:
        logger.info(f"--- Processing Group: {gid} ---")
        schedule_df = load_and_validate_schedule(gid, data_dir)
        if schedule_df is None:
            continue
        group_reports = process_group(
            gid,
            schedule_df,
            args,
            moves_dir,
            stats_dir,
            output_dir,
            env,
            template,
            target_team_id,
        )
        total_processed += len(group_reports)
        total_skipped += len(schedule_df) - len(group_reports)
        all_report_links.extend(group_reports)

    logger.info("Report generation complete.")
    logger.info(f"Successfully processed {total_processed} matches.")
    if total_skipped > 0:
        logger.warning(f"Skipped {total_skipped} matches due to missing data.")

    if all_report_links:
        generate_index_page(all_report_links, output_dir, target_team_id)
    else:
        logger.info(
            f"No reports were generated for Team {target_team_id}, skipping index.html creation."
        )


if __name__ == "__main__":
    main()
