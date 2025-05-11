from math import pi
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from report_tools.logger import logger
from report_tools.utils import get_absolute_seconds, shorten_name

PLOT_DPI = 100
SCORE_TIMELINE_COLOR_LOCAL = "#003366"
SCORE_TIMELINE_COLOR_VISITOR = "#7FFFD4"
PAIRWISE_CMAP = "Reds"
ON_NET_PALETTE = "coolwarm"
LINEUP_PALETTE = "viridis"


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


def plot_evolution(
    evo_df: pd.DataFrame,
    group_name: str,
    gid: str,
    plot_dir: Path,
    names_to_exclude: set[str] = set(),
):
    """Generates and saves faceted line plots for player evolution."""
    # Filter DataFrame before plotting
    filtered_evo_df = evo_df[~evo_df["player"].isin(names_to_exclude)]

    if filtered_evo_df.empty:
        logger.info(
            "Evolution DataFrame is empty after exclusions, skipping plots for group %s.",
            group_name,
        )
        return

    # Ensure plot directory exists
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Melt DataFrame for easier plotting with Seaborn's hue/style
    df_melt_mins = filtered_evo_df.melt(
        id_vars=["game_idx", "player"],
        value_vars=["mins", "roll_mins"],
        var_name="Metric_Type",
        value_name="Minutes",
    )
    df_melt_pts = filtered_evo_df.melt(
        id_vars=["game_idx", "player"],
        value_vars=["pts", "roll_pts"],
        var_name="Metric_Type",
        value_name="Points",
    )

    # Apply shortening to the player column for plot labels/titles
    df_melt_mins["player"] = df_melt_mins["player"].apply(shorten_name)
    df_melt_pts["player"] = df_melt_pts["player"].apply(shorten_name)

    sorted_players = sorted(df_melt_mins["player"].unique())

    # --- Plot Minutes ---
    try:
        logger.info("Generating Minutes evolution plot for group %s...", group_name)
        sns.set_theme(style="ticks")
        g_mins = sns.relplot(
            data=df_melt_mins,
            x="game_idx",
            y="Minutes",
            hue="player",  # Color lines by player
            style="Metric_Type",  # Different style for actual vs rolling
            col="player",
            col_order=sorted_players,
            col_wrap=4,  # Facet by player, 4 columns wide
            kind="line",
            height=3,
            aspect=1.5,
            legend=False,  # Avoid giant legend, colors identify player in title
        )
        g_mins.set_titles("{col_name}")
        g_mins.set_axis_labels("Game Index", "Minutes")
        g_mins.figure.suptitle(f"Minutes Evolution - Group {group_name}", y=1.03)
        plot_path_mins = plot_dir / f"evolution_minutes_group_{gid}.png"  # Use gid
        mins_desc = "Shows minutes played per game (line) and 3-game rolling average (dashed) for each player."
        g_mins.figure.text(
            0.5, 0.01, mins_desc, ha="center", va="bottom", fontsize=10, wrap=True
        )
        g_mins.figure.subplots_adjust(
            bottom=0.15
        )  # Adjust bottom margin for description
        g_mins.savefig(plot_path_mins, dpi=150)
        plt.close(g_mins.figure)
    except Exception as e:
        logger.error(
            "Failed to generate/save minutes plot for group %s: %s", group_name, e
        )
        if "g_mins" in locals() and hasattr(g_mins, "figure"):
            plt.close(g_mins.figure)  # Attempt to close if figure exists

    # --- Plot Points ---
    try:
        logger.info("Generating Points evolution plot for group %s...", group_name)
        sns.set_theme(style="ticks")
        g_pts = sns.relplot(
            data=df_melt_pts,
            x="game_idx",
            y="Points",
            hue="player",
            style="Metric_Type",
            col="player",
            col_order=sorted_players,
            col_wrap=4,
            kind="line",
            height=3,
            aspect=1.5,
            legend=False,
        )
        g_pts.set_titles("{col_name}")
        g_pts.set_axis_labels("Game Index", "Points")
        g_pts.figure.suptitle(f"Points Evolution - Group {group_name}", y=1.03)
        plot_path_pts = plot_dir / f"evolution_points_group_{gid}.png"  # Use gid
        pts_desc = "Shows points scored per game (line) and 3-game rolling average (dashed) for each player."
        g_pts.figure.text(
            0.5, 0.01, pts_desc, ha="center", va="bottom", fontsize=10, wrap=True
        )
        g_pts.figure.subplots_adjust(bottom=0.15)  # Adjust bottom margin
        g_pts.savefig(plot_path_pts, dpi=150)
        plt.close(g_pts.figure)
    except Exception as e:
        logger.error(
            "Failed to generate/save points plot for group %s: %s", group_name, e
        )
        if "g_pts" in locals() and hasattr(g_pts, "figure"):
            plt.close(g_pts.figure)

    # --- Plot DRtg ---
    try:
        if "roll_drtg" in filtered_evo_df.columns:
            logger.info("Generating DRtg evolution plot for group %s...", group_name)
            df_melt_drtg = filtered_evo_df.melt(
                id_vars=["game_idx", "player"],
                value_vars=["drtg", "roll_drtg"],
                var_name="Metric_Type",
                value_name="DRtg",
            )
            # Apply shortening here too
            df_melt_drtg["player"] = df_melt_drtg["player"].apply(shorten_name)
            sns.set_theme(style="ticks")
            g_drtg = sns.relplot(
                data=df_melt_drtg,
                x="game_idx",
                y="DRtg",
                hue="player",
                style="Metric_Type",
                col="player",
                col_order=sorted_players,
                col_wrap=4,
                kind="line",
                height=3,
                aspect=1.5,
                legend=False,
            )
            g_drtg.set_titles("{col_name}")
            g_drtg.set_axis_labels("Game Index", "Defensive Rating (DRtg)")
            for ax in g_drtg.axes.flat:
                ax.invert_yaxis()  # Lower DRtg is better
            g_drtg.figure.suptitle(
                f"Defensive Rating Evolution - Group {group_name}", y=1.03
            )
            plot_path_drtg = plot_dir / f"evolution_drtg_group_{gid}.png"  # Use gid
            drtg_desc = "Shows Defensive Rating (pts allowed per 40 min while on court) per game (line) and rolling average (dashed). Lower is better."
            g_drtg.figure.text(
                0.5, 0.01, drtg_desc, ha="center", va="bottom", fontsize=10, wrap=True
            )
            g_drtg.figure.subplots_adjust(bottom=0.15)  # Adjust bottom margin
            g_drtg.savefig(plot_path_drtg, dpi=150)
            plt.close(g_drtg.figure)
    except Exception as e:
        logger.error(
            "Failed to generate/save DRtg plot for group %s: %s", group_name, e
        )
        if "g_drtg" in locals() and hasattr(g_drtg, "figure"):
            plt.close(g_drtg.figure)


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


# --- Plotting Function: Stacked Points vs Opponent Points Bar Chart ---
def plot_stacked_points_vs_opponent(
    evo_df: pd.DataFrame,
    game_summary_df: pd.DataFrame,
    group_name: str,
    gid: str,
    plot_dir: Path,
    names_to_exclude: set[str] = set(),
):
    """Generates and saves a grouped bar chart: stacked points per player and opponent points per game."""
    # Filter player evolution data first
    filtered_evo_df = evo_df[~evo_df["player"].isin(names_to_exclude)]

    if filtered_evo_df.empty:
        logger.info(
            "Evolution DataFrame empty after exclusions, skipping stacked points plot for group %s.",
            group_name,
        )
        return

    if game_summary_df.empty:
        logger.info(
            "Game summary DataFrame is empty, skipping stacked points plot for group %s.",
            group_name,
        )
        return

    # Pivot the table for stacked bar chart: games as index, players as columns, POINTS as values
    try:
        points_pivot = filtered_evo_df.pivot_table(
            index="game_idx", columns="player", values="pts", aggfunc="sum"
        ).fillna(0)
        # Apply name shortening to columns for better display in legend
        points_pivot.columns = points_pivot.columns.map(shorten_name)
        # Sort columns alphabetically for consistent legend order
        points_pivot = points_pivot.reindex(sorted(points_pivot.columns), axis=1)

    except Exception as e:
        logger.error(
            "Failed to pivot evolution data for stacked points plot for group %s: %s",
            group_name,
            e,
        )
        return

    # Prepare opponent points data
    opponent_points = game_summary_df.set_index("game_idx")["opponent_pts"]

    # Align data - crucial step! Use intersection of indices
    common_indices = points_pivot.index.intersection(opponent_points.index)
    if common_indices.empty:
        logger.warning(
            "No common game indices between player points and opponent points data for group %s.",
            group_name,
        )
        return

    points_pivot = points_pivot.loc[common_indices]
    opponent_points = opponent_points.loc[common_indices]

    # Use the index from the pivoted minutes table directly
    game_indices = points_pivot.index

    plot_dir.mkdir(parents=True, exist_ok=True)
    filename = plot_dir / f"bars_stacked_points_vs_opponent_group_{gid}.png"

    try:
        n_games = len(game_indices)
        x = np.arange(n_games)  # the label locations
        width = 0.35  # Width for grouped bars

        fig, ax = plt.subplots(figsize=(max(14, n_games * 0.8), 7))  # Dynamic width

        # --- Plot Stacked Player Points Bar --- (at x - width/2)
        bottom = np.zeros(n_games)
        # Use a consistent color map
        colors = plt.cm.tab20(np.linspace(0, 1, len(points_pivot.columns)))
        for i, player in enumerate(points_pivot.columns):
            ax.bar(
                x - width / 2,
                points_pivot[player],
                width,
                label=player,
                bottom=bottom,
                color=colors[i],
            )
            bottom += points_pivot[player].values

        # --- Plot Opponent Points Bar --- (at x + width/2)
        ax.bar(
            x + width / 2, opponent_points, width, label="Opponent Pts", color="#FF5733"
        )  # A distinct color

        # Add some text for labels, title and axes ticks
        ax.set_ylabel("Points")
        ax.set_title(f"Player Points vs Opponent Points per Game - Group {group_name}")
        ax.set_xticks(x)
        ax.set_xticklabels(game_indices)
        ax.tick_params(axis="x", rotation=0)

        # --- Move Legend ---
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Player / Metric")

        summary_desc = "Grouped bar chart per game. Left bar (stacked) shows points scored by each player. Right bar shows total points scored by the opponent."
        plt.figtext(
            0.5, 0.01, summary_desc, ha="center", va="bottom", fontsize=10, wrap=True
        )

        # Adjust layout to prevent legend/description overlap
        plt.tight_layout(
            rect=[0, 0.03, 0.85, 1]
        )  # Leave space on right for legend and bottom for text

        plt.savefig(filename, dpi=150)
        plt.close(fig)
        logger.info(f"-> Saved Stacked Points vs Opponent Chart: {filename}")

    except Exception as e:
        logger.error(
            "Failed to generate/save stacked points plot for group %s: %s",
            group_name,
            e,
        )
        if "fig" in locals():
            plt.close(fig)


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

    top_n = 5
    # Make a copy before modifying for plotting
    lineup_df_plot = lineup_df.sort_values("NetRtg", ascending=False).head(top_n).copy()

    # Replace hyphens with newlines *specifically for plot labels*
    lineup_df_plot["lineup"] = lineup_df_plot["lineup"].str.replace(
        " - ", "\n", regex=False
    )

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


# --- Plotting Function: Player Comparison Radar Chart ---
def plot_player_comparison_radar(
    group_results: dict,
    player_name: str,
    group_names: list,
    group_ids: list,
    plot_dir: Path,
):
    """Generates a radar chart comparing a player's stats between two groups."""
    if len(group_names) != 2:
        logger.error("Radar plot function called with incorrect number of groups.")
        return

    # Use descriptive names passed in
    g1_name, g2_name = group_names[0], group_names[1]
    # Find the original group IDs from the keys of group_results based on the names
    g1_id = next(
        (
            gid
            for gid, data in group_results.items()
            if data.get("group_name") == g1_name
        ),
        None,
    )
    g2_id = next(
        (
            gid
            for gid, data in group_results.items()
            if data.get("group_name") == g2_name
        ),
        None,
    )

    if not g1_id or not g2_id:
        logger.error("Could not map provided group names back to group IDs in results.")
        return

    g1_data = group_results[g1_id]
    g2_data = group_results[g2_id]

    # Extract player aggregate stats (assuming stored in player_df)
    try:
        player1_stats = g1_data["player_df"][
            g1_data["player_df"]["Player"] == shorten_name(player_name)
        ].iloc[0]
        player2_stats = g2_data["player_df"][
            g2_data["player_df"]["Player"] == shorten_name(player_name)
        ].iloc[0]
    except IndexError:
        logger.error(
            f"Could not find player '{player_name}' (shortened: {shorten_name(player_name)}) in the results for both groups. Check spelling and data."
        )
        # REMOVED Debug note comment
        logger.debug(
            f"Group {g1_name} players: {g1_data['player_df']['Player'].tolist()}"
        )
        logger.debug(
            f"Group {g2_name} players: {g2_data['player_df']['Player'].tolist()}"
        )
        return
    except KeyError as e:
        logger.error(f"Missing key '{e}' in stored group data for radar plot.")
        return

    # --- Select & Normalize Stats ---
    # Define categories (stats) to plot - use per-game stats where sensible
    # Note: Needs GP > 0 for division
    gp1 = player1_stats.get("GP", 0)
    gp2 = player2_stats.get("GP", 0)

    stats_to_plot = {
        "Mins/GP": (
            player1_stats.get("Mins", 0) / gp1 if gp1 > 0 else 0,
            player2_stats.get("Mins", 0) / gp2 if gp2 > 0 else 0,
        ),
        "Pts/GP": (
            player1_stats.get("PTS", 0) / gp1 if gp1 > 0 else 0,
            player2_stats.get("PTS", 0) / gp2 if gp2 > 0 else 0,
        ),
        # Add Plus/Minus per GP from evolution table or calculate aggregate
        # For simplicity, using aggregate +/- here if available, else 0
        # Need to calculate aggregate +/- and add to player_df first!
        # Let's omit +/- for now until it's added to player_df
        # '+/-/GP': (0, 0),
        "T3/GP": (
            player1_stats.get("T3", 0) / gp1 if gp1 > 0 else 0,
            player2_stats.get("T3", 0) / gp2 if gp2 > 0 else 0,
        ),
        "T2/GP": (
            player1_stats.get("T2", 0) / gp1 if gp1 > 0 else 0,
            player2_stats.get("T2", 0) / gp2 if gp2 > 0 else 0,
        ),
        "T1/GP": (
            player1_stats.get("T1", 0) / gp1 if gp1 > 0 else 0,
            player2_stats.get("T1", 0) / gp2 if gp2 > 0 else 0,
        ),
        "Fouls/GP": (
            player1_stats.get("Fouls", 0) / gp1 if gp1 > 0 else 0,
            player2_stats.get("Fouls", 0) / gp2 if gp2 > 0 else 0,
        ),
    }

    categories = list(stats_to_plot.keys())
    n_categories = len(categories)

    # Extract values for each group
    values1 = [stats_to_plot[cat][0] for cat in categories]
    values2 = [stats_to_plot[cat][1] for cat in categories]

    # Compute max values for normalization (across both players)
    max_values = [
        max(stats_to_plot[cat][0], stats_to_plot[cat][1]) for cat in categories
    ]
    # Avoid division by zero if max is 0 for a category
    max_values = [v if v > 0 else 1 for v in max_values]

    # Normalize values (0 to 1 scale)
    norm_values1 = [v / max_v for v, max_v in zip(values1, max_values)]
    norm_values2 = [v / max_v for v, max_v in zip(values2, max_values)]

    # --- Create Radar Chart ---
    angles = [n / float(n_categories) * 2 * pi for n in range(n_categories)]
    angles += angles[:1]  # Close the plot

    norm_values1 += norm_values1[:1]
    norm_values2 += norm_values2[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Set ticks and labels
    plt.xticks(angles[:-1], categories)
    ax.set_yticks(np.arange(0, 1.1, 0.2))  # Ticks from 0 to 1
    ax.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0, 1.1, 0.2)])
    ax.set_ylim(0, 1.1)  # Ensure scale goes slightly beyond 1

    # Plot data
    ax.plot(
        angles,
        norm_values1,
        linewidth=2,
        linestyle="solid",
        label=g1_name,  # Use group name for label
        color="skyblue",
    )
    ax.fill(angles, norm_values1, "skyblue", alpha=0.4)
    ax.plot(
        angles,
        norm_values2,
        linewidth=2,
        linestyle="solid",
        label=g2_name,  # Use group name for label
        color="lightcoral",
    )
    ax.fill(angles, norm_values2, "lightcoral", alpha=0.4)

    # Add legend and title
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.title(
        f"Player Comparison: {player_name}\n(Normalized Stats per GP)", size=16, y=1.1
    )

    # Save plot
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe_g1_name = "".join(c if c.isalnum() else "_" for c in g1_name)
    safe_g2_name = "".join(c if c.isalnum() else "_" for c in g2_name)
    filename = (
        plot_dir
        / f"radar_compare_{player_name.replace(' ', '_')}_groups_{safe_g1_name}_vs_{safe_g2_name}.png"
    )
    try:
        radar_desc = "Compares selected player stats (normalized per GP) between the two phases. Values closer to the edge are better relative to the max observed for that stat."
        plt.figtext(
            0.5, 0.01, radar_desc, ha="center", va="bottom", fontsize=10, wrap=True
        )
        plt.subplots_adjust(bottom=0.1)  # Adjust bottom slightly for text
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"-> Saved Player Comparison Radar Chart: {filename}")
    except Exception as e:
        logger.error(f"Failed to save radar chart for player {player_name}: {e}")
        plt.close(fig)
