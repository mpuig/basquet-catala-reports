from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from report_tools.logger import logger
from report_tools.utils import get_absolute_seconds

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
