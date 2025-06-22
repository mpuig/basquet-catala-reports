"""Module for building and managing group reports."""

from pathlib import Path
from typing import List

import pandas as pd

from report_tools.data_loaders import (
    load_schedule,
    load_match_moves,
    load_match_stats,
    load_match,
    load_team_with_players,
)
from report_tools.models.groups import Group
from report_tools.models.teams import Team
from report_tools.models.matches import Match


def get_group_teams(
    df: pd.DataFrame, data_dir: Path, season: str = "2024"
) -> List[Team]:
    """Extract unique teams from the schedule DataFrame with stats."""
    team_ids = set()
    for _, row in df.iterrows():
        local_team = str(row["local_team"]).strip()
        visitor_team = str(row["visitor_team"]).strip()
        if local_team and local_team.lower() != "descansa":
            team_ids.add(int(row["local_team_id"]))
        if visitor_team and visitor_team.lower() != "descansa":
            team_ids.add(int(row["visitor_team_id"]))

    teams = []
    for team_id in sorted(team_ids):
        # Try to load team data
        team = load_team_with_players(team_id, season, data_dir)
        if team:
            teams.append(team)
        else:
            # Fallback to basic team info from schedule
            team_rows = df[
                (df["local_team_id"] == team_id) | (df["visitor_team_id"] == team_id)
            ]
            if not team_rows.empty:
                row = team_rows.iloc[0]
                if int(row["local_team_id"]) == team_id:
                    team_name = str(row["local_team"]).strip()
                else:
                    team_name = str(row["visitor_team"]).strip()
                teams.append(
                    Team(id=team_id, name=team_name, short_name=team_name[:3].upper())
                )

    return teams


def get_group_matches(
    group_name: str, df: pd.DataFrame, teams: List[Team], data_dir: Path
) -> List[Match]:
    """Build Match objects from schedule and detailed stats data."""
    result: List[Match] = []
    for _, match_info_row in df.iterrows():
        match_id = str(match_info_row.get("id", ""))
        if not match_id or match_id == "nan":
            continue

        # Try to load match data first
        match = load_match(match_id, data_dir)
        if match:
            # Set legacy compatibility fields from schedule
            match.match_date = match_info_row.get("date_time", "Unknown Date")
            match.group_name = group_name

            # Ensure final_score is set if not already
            if not match.final_score:
                match.final_score = match_info_row.get("score", "-")

            result.append(match)
        else:
            # Fallback to legacy match loading
            local_team_id = int(match_info_row["local_team_id"])
            local_team = next(
                (team for team in teams if team.id == local_team_id), None
            )

            visitor_team_id = int(match_info_row["visitor_team_id"])
            visitor_team = next(
                (team for team in teams if team.id == visitor_team_id), None
            )

            match_date = match_info_row.get("date_time", "Unknown Date")
            score = match_info_row.get("score", "-")
            moves = load_match_moves(match_id, data_dir)
            stats = load_match_stats(match_id, data_dir)

            # Create basic match structure for new Match model
            from report_tools.models.matches import PlayerStatsData, EventTime

            basic_match = Match(
                # Required fields for new model
                id_match_intern=int(match_id) if match_id.isdigit() else 0,
                id_match_extern=int(match_id) if match_id.isdigit() else 0,
                time=match_date,
                local_id=local_team_id,
                visit_id=visitor_team_id,
                data=PlayerStatsData(),
                event_time=EventTime(),
                # Legacy compatibility fields
                id=match_id,
                match_date=match_date,
                group_name=group_name,
                local=local_team,
                visitor=visitor_team,
                final_score=score,
                moves=moves,
                stats=stats,
            )
            result.append(basic_match)

    return result


def build_groups(
    group_ids: List[int], data_dir: Path, season: str = "2024"
) -> List[Group]:
    """Build Group objects with detailed statistics.

    Args:
        group_ids: List of group IDs to process
        data_dir: Directory containing the data files
        season: Season identifier for loading team stats

    Returns:
        List of Group objects with team and match data
    """
    GROUP_ID_TO_NAME = {
        17182: "Infantil 1r Any - Fase 1",
        18299: "Infantil 1r Any - Fase 2",
    }
    result: List[Group] = []
    for group_id in group_ids:
        group_id = int(group_id)
        schedule_df = load_schedule(group_id, data_dir)
        if schedule_df is None or schedule_df.empty:
            continue

        # Load teams with statistics
        teams = get_group_teams(schedule_df, data_dir, season)
        group_name = GROUP_ID_TO_NAME.get(group_id, f"Group {group_id}")

        # Load matches with statistics
        matches = get_group_matches(group_name, schedule_df, teams, data_dir)

        result.append(
            Group(
                id=group_id,
                name=group_name,
                schedule=schedule_df,
                teams=teams,
                matches=matches,
            )
        )

    return result
