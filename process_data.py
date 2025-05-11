"""
Process Basketball Statistics Data
----------------------------------

This script analyzes basketball match data downloaded by `fetch_data.py`.
It calculates various statistics for a specified team across one or more competition groups,
prints summary tables to the console, and generates plots.

Features:
---------
*   Calculates aggregate player stats (GP, Mins, PTS, T3/T2/T1, Fouls, +/-).
*   Calculates On/Off court Net Rating per player.
*   Calculates pairwise minutes played between teammates.
*   Calculates lineup statistics (Mins, Usage %, NetRtg).
*   Generates evolution tables/plots showing rolling averages for key stats.
*   Generates plots: pairwise heatmap, stacked points, On/Off bars, lineup bars, player comparison radar.
*   Supports processing multiple groups or a single match.
*   Allows excluding specific players from analysis and plots.
*   Optionally generates individual HTML reports for each match (integrating logic from generate_match_reports.py).

Dependencies:
-------------
*   Python 3.10+
*   pandas >= 2.0
*   matplotlib
*   seaborn
*   numpy
*   (Optional, for match reports) jinja2, markdown2, litellm

CLI Examples:
-------------
1.  Analyze two groups for a team, saving plots:
    $ python process_data.py --team 69630 --groups 17182 18299 --data-dir data --plot-dir plots

2.  Analyze only group 17182, excluding player UUID 'uuid1':
    $ python process_data.py --team 69630 --groups 17182 --data-dir data --exclude-players uuid1

3.  Analyze only a single match:
    $ python process_data.py --team 69630 --match 66eee759189aa402d83001d7 --data-dir data

4.  Analyze two groups and generate a comparison radar chart for 'Player A':
    $ python process_data.py --team 69630 --groups 17182 18299 --compare-player "Player A"

5.  (If merged) Analyze groups and generate individual match reports:
    $ python process_data.py --team 69630 --groups 17182 18299 --generate-match-reports --reports-dir reports

"""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from report_tools.data_loaders import (
    load_schedule,
    load_match_moves,
    load_team_stats,
    load_match_stats,
)
from report_tools.logger import logger
from report_tools.plotting import (
    plot_pairwise_heatmap,
    plot_player_on_net,
    plot_lineup_netrtg,
    plot_score_timeline,
    plot_evolution,
    plot_stacked_points_vs_opponent,
    plot_player_comparison_radar,
)
from report_tools.stats_calculator import StatsCalculator


def main() -> None:
    parser = argparse.ArgumentParser(description="U13 ASFE season analysis (refactor)")
    parser.add_argument("--team", required=True, help="Team ID to analyse")
    parser.add_argument(
        "--groups", nargs="+", required=True, help="Competition group IDs"
    )
    parser.add_argument(
        "--data-dir", default="data", help="Root data folder (CSV & JSON)"
    )
    parser.add_argument(
        "--match", default=None, help="Optional: Process only a single match ID"
    )
    parser.add_argument(
        "--season", default="2024", help="Season identifier (default: 2024)"
    )
    parser.add_argument(
        "--plot-dir", default="plots", help="Directory to save plot images"
    )
    parser.add_argument(
        "--exclude-players",
        nargs="+",
        default=[],
        help="List of player UUIDs to exclude from reports",
    )
    parser.add_argument(
        "--compare-player",
        default=None,
        help="Player name (as in data) to compare between the two groups using a radar chart",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    moves_dir = data_dir / "match_moves"
    stats_dir = data_dir / "match_stats"
    team_stats_dir = data_dir / "team_stats"  # Directory for team stats

    # Mapping from Group ID to descriptive name
    GROUP_ID_TO_NAME = {
        "17182": "Infantil 1r Any - Fase 1",
        "18299": "Infantil 1r Any - Fase 2",
    }

    # Try to load team info for headers
    team_name_for_header = args.team  # Default to ID
    names_to_exclude = set()  # Initialize empty set for names

    team_stats_data = load_team_stats(args.team, args.season, team_stats_dir)
    if team_stats_data and "team" in team_stats_data:
        team_name_for_header = team_stats_data["team"].get("teamName", args.team)
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
                logger.warning(
                    "Could not find names for provided excluded UUIDs in team stats file."
                )

    if args.match:
        # Process Single Match
        logger.info(
            f"Processing single match ID: {args.match} for Team: {team_name_for_header} ({args.team})"
        )
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
                match_row = schedule_df_group[
                    schedule_df_group["match_id"] == args.match
                    ]
                if not match_row.empty:
                    logger.info(f"Match {args.match} found in group {gid} schedule.")
                    schedule_df_single = match_row
                    moves_single = load_match_moves(args.match, moves_dir)
                    stats_single = load_match_stats(
                        args.match, stats_dir
                    )  # Load stats for the single match
                    match_found_in_schedule = True
                    break  # Found the match, no need to check other groups
            except FileNotFoundError:
                logger.error(
                    "Schedule file not found (should not happen after check): %s",
                    csv_path,
                )
            except ValueError as e:
                logger.error("Error loading schedule %s: %s", csv_path, e)
            except Exception as e:
                logger.error("Unexpected error processing schedule %s: %s", csv_path, e)

        if not match_found_in_schedule:
            logger.error(
                f"Match ID {args.match} not found in any specified group's schedule."
            )
            return

        if not moves_single:
            logger.error(
                f"Match moves data not found or failed to load for match {args.match}."
            )
            return
        if not stats_single:
            logger.error(
                f"Match stats data not found or failed to load for match {args.match}. Required for team ID mapping."
            )
            return

        calc = StatsCalculator(args.team)
        # Pass the single match data as dictionaries keyed by the match ID
        calc.process(
            schedule_df_single, {args.match: moves_single}, {args.match: stats_single}
        )

        # Display results for the single match
        print_results(calc, names_to_exclude)

        # Generate plots for single match
        plot_output_dir = Path(args.plot_dir) / args.match
        plot_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate score timeline plot
        plot_score_timeline(moves_single, stats_single, plot_output_dir, args.team)

        # Generate pairwise heatmap
        pairwise_df = calc.pairwise_minutes(names_to_exclude=names_to_exclude)
        plot_pairwise_heatmap(pairwise_df, plot_output_dir)

        # Generate On/Off net rating plot
        onoff_df = calc.on_off_table(names_to_exclude=names_to_exclude)
        plot_player_on_net(onoff_df, plot_output_dir)

        # Generate lineup net rating plot
        lineup_df = calc.lineup_table(names_to_exclude=names_to_exclude)
        plot_lineup_netrtg(lineup_df, plot_output_dir)

    else:
        # Process Multiple Groups Separately
        group_results = {}
        for gid in args.groups:
            # Use descriptive group name if available, otherwise use ID
            group_name = GROUP_ID_TO_NAME.get(gid, f"Group {gid}")
            print(
                f"\n{'=' * 15} Processing Group: {group_name} ({gid}) | Team: {team_name_for_header} ({args.team}) {'=' * 15}"
            )
            csv_path = data_dir / f"results_{gid}.csv"

            # Data specific to this group
            current_group_schedule = None
            current_group_moves: Dict[str, List[dict]] = {}
            current_group_stats: Dict[str, dict] = {}

            try:
                schedule_df = load_schedule(csv_path)
                current_group_schedule = (
                    schedule_df  # Keep this group's schedule separate
                )

                # Load moves and stats only for matches in this group's schedule
                match_ids_in_group = (
                    schedule_df["match_id"].dropna().astype(str).unique()
                )
                logger.info(
                    f"Loading JSON data for {len(match_ids_in_group)} matches in group {gid}..."
                )
                for mid in match_ids_in_group:
                    loaded_moves = load_match_moves(mid, moves_dir)
                    if loaded_moves:
                        current_group_moves[mid] = loaded_moves
                    loaded_stats = load_match_stats(mid, stats_dir)
                    if loaded_stats:
                        current_group_stats[mid] = loaded_stats

            except FileNotFoundError:
                logger.error(
                    "Schedule file not found: %s. Skipping group %s.", csv_path, gid
                )
                continue
            except ValueError as e:
                logger.error(
                    "Error loading schedule %s: %s. Skipping group %s.",
                    csv_path,
                    e,
                    gid,
                )
                continue
            except Exception as e:
                logger.error(
                    "Unexpected error processing schedule %s: %s. Skipping group %s.",
                    csv_path,
                    e,
                    gid,
                )
                continue

            if current_group_schedule is None or current_group_schedule.empty:
                logger.warning(
                    f"No schedule data loaded for group {gid}. Skipping processing."
                )
                continue

            # Instantiate and process *for this group only*
            logger.info(f"Calculating stats for group {gid}...")
            calc = StatsCalculator(args.team)
            calc.process(
                current_group_schedule, current_group_moves, current_group_stats
            )

            # Store results for potential comparison
            group_results[gid] = {
                "calc": calc,
                "evo_table": calc.evolution_table(names_to_exclude=names_to_exclude),
                "pairwise_df": calc.pairwise_minutes(names_to_exclude=names_to_exclude),
                "onoff_df": calc.on_off_table(names_to_exclude=names_to_exclude),
                "lineup_df": calc.lineup_table(names_to_exclude=names_to_exclude),
                "player_df": calc.player_table(names_to_exclude=names_to_exclude),
                "group_name": group_name,  # Store descriptive name
            }

            # Display results *for this group*
            print_results(calc, names_to_exclude)

            # Generate and Save Plots
            plot_output_dir = Path(args.plot_dir) / gid
            plot_output_dir.mkdir(parents=True, exist_ok=True)

            evo_table = calc.evolution_table(names_to_exclude=names_to_exclude)
            plot_evolution(
                evo_table,
                group_name,
                gid,
                plot_output_dir,
                names_to_exclude=names_to_exclude,
            )

            pairwise_df = calc.pairwise_minutes(names_to_exclude=names_to_exclude)
            plot_pairwise_heatmap(pairwise_df, plot_output_dir)

            # Prepare game summary data for the new plot
            game_summary_df = pd.DataFrame(calc.game_summaries)
            plot_stacked_points_vs_opponent(
                evo_table,
                game_summary_df,
                group_name,
                gid,
                plot_output_dir,
                names_to_exclude,
            )

            onoff_df = calc.on_off_table(names_to_exclude=names_to_exclude)
            plot_player_on_net(onoff_df, plot_output_dir)

            lineup_df = calc.lineup_table(names_to_exclude=names_to_exclude)
            plot_lineup_netrtg(lineup_df, plot_output_dir)

            print(
                f"\n{'=' * 15} Finished Group: {group_name} ({gid}) {'=' * 15}"
            )  # Mark end of group output

        # Generate Comparison Radar Chart (if applicable)
        if args.compare_player and len(args.groups) == 2:
            group_ids = args.groups
            # Get descriptive names from stored results
            g1_name = group_results[group_ids[0]].get(
                "group_name", f"Group {group_ids[0]}"
            )
            g2_name = group_results[group_ids[1]].get(
                "group_name", f"Group {group_ids[1]}"
            )

            player_name = args.compare_player  # Use the name provided directly
            if player_name in names_to_exclude:
                logger.warning(
                    f"Cannot generate comparison chart for excluded player: {player_name}"
                )
            else:
                logger.info(
                    f"Attempting to generate comparison radar chart for player: {player_name} between groups {g1_name} and {g2_name}"
                )
                plot_output_dir = Path(args.plot_dir)
                # Pass the whole results dict, group names, and group ids
                plot_player_comparison_radar(
                    group_results,
                    player_name,
                    [g1_name, g2_name],
                    group_ids,
                    plot_output_dir,
                )
        elif args.compare_player:
            logger.warning(
                "Radar chart comparison requires exactly two group IDs to be provided."
            )


def print_results(calc: "StatsCalculator", names_to_exclude: set[str]):
    """Helper function to print the standard output tables."""
    # Filter players before checking if data exists
    filtered_players = {
        name: data
        for name, data in calc.players.items()
        if name not in names_to_exclude
    }
    if not filtered_players:
        print(
            "\n(No data found for this team OR all players excluded in the processed match(es))"
        )
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
        print(pair_df.iloc[:max_dim, :max_dim].to_string(float_format="{:d}".format))
    else:
        print("(No pairwise data to display after exclusions)")

    print("\n==== On/Off Net Rating ====")
    onoff_df = calc.on_off_table(names_to_exclude=names_to_exclude)
    if not onoff_df.empty:
        print(onoff_df.to_string(index=False))
    else:
        print("(Insufficient data for On/Off calculation or all players excluded)")

    print("\n==== Evolution (rolling 3‑games) ====")  # Removed 'sample' from header
    evo = calc.evolution_table(names_to_exclude=names_to_exclude)
    if not evo.empty:
        print(
            evo.to_string(index=False, float_format="{:.1f}".format)
        )  # Format floats to 1 decimal place
    else:
        print("(No evolution data after exclusions)")

    print("\n==== Lineups (usage ≥5 %) ====")
    lu_df = calc.lineup_table(names_to_exclude=names_to_exclude)
    if not lu_df.empty:
        print(lu_df.head(10).to_string(index=False))
    else:
        print("(No lineup data meeting criteria after exclusions)")


if __name__ == "__main__":
    main()
