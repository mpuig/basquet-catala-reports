import os
from typing import Optional

import litellm
import pandas as pd

from report_tools.logger import logger

LLM_MODEL = "gpt-4.1"
LLM_TEMPERATURE = 0.5
LLM_MAX_TOKENS = 250


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


def generate_llm_summary(
    match_info: pd.Series,
    match_stats: dict,
    target_team_id: str,
    player_stats_df: pd.DataFrame,
) -> Optional[str]:
    """Generates a narrative match summary using an LLM via litellm."""
    logger.info("Attempting to generate LLM summary...")

    # --- Prepare data for prompt ---
    local_name = match_info.get("local_team", "Local")
    visitor_name = match_info.get("visitor_team", "Visitor")
    local_id = str(match_info.get("local_team_id", ""))
    is_target_local = target_team_id == local_id
    target_team_name = local_name if is_target_local else visitor_name
    opponent_team_name = visitor_name if is_target_local else local_name
    score = match_info.get("score", "-")
    match_date_time = match_info.get("date_time", "Unknown")

    target_stats, opponent_stats = _match_team_stats(match_stats, target_team_id)
    if target_stats and opponent_stats:
        ts_data = target_stats.get("data", {})
        os_data = opponent_stats.get("data", {})
        team_stats_summary = "\n".join(
            [
                f"- Points: {ts_data.get('score', '?')} vs {os_data.get('score', '?')}",
                f"- T2: {ts_data.get('shotsOfTwoSuccessful', '?')}/{ts_data.get('shotsOfTwoAttempted', '?')} vs {os_data.get('shotsOfTwoSuccessful', '?')}/{os_data.get('shotsOfTwoAttempted', '?')}",
                f"- T3: {ts_data.get('shotsOfThreeSuccessful', '?')}/{ts_data.get('shotsOfThreeAttempted', '?')} vs {os_data.get('shotsOfThreeSuccessful', '?')}/{os_data.get('shotsOfThreeAttempted', '?')}",
                f"- T1: {ts_data.get('shotsOfOneSuccessful', '?')}/{ts_data.get('shotsOfOneAttempted', '?')} vs {os_data.get('shotsOfOneSuccessful', '?')}/{os_data.get('shotsOfOneAttempted', '?')}",
                f"- Fouls: {ts_data.get('faults', '?')} vs {os_data.get('faults', '?')}",
            ]
        )
    else:
        team_stats_summary = "(Team stats comparison not available)"

    # Top player stats (e.g., top 3 scorers for target team)
    top_players_summary = "(No player stats available)"
    if not player_stats_df.empty:
        top_scorers = player_stats_df.nlargest(3, "PTS")[["Player", "PTS"]]
        top_players_lines = [
            f"- {row['Player']}: {row['PTS']} PTS" for _, row in top_scorers.iterrows()
        ]
        top_players_summary = "\n".join(top_players_lines)

    # Optional momentum data – inject if available
    momentum_section = ""
    if momentum_info := match_stats.get("momentum"):
        momentum_section = f"""
    Momentum Insights:
    - Lead changes: {momentum_info.get('lead_changes', '?')}
    - Ties: {momentum_info.get('ties', '?')}
    - Largest lead held: {momentum_info.get('max_lead', '?')} points
    """

    # Optional per-quarter scores
    quarters_section = ""
    if "quarters" in match_stats:
        quarters = match_stats["quarters"]
        quarters_str = " | ".join(
            f"Q{i + 1}: {q.get('home', '?')}-{q.get('away', '?')}"
            for i, q in enumerate(quarters)
        )
        quarters_section = f"\nScoring by Quarter:\n{quarters_str}\n"

    # Final Prompt
    prompt = f"""
    Generate a concise and informative 2-paragraph summary of a youth basketball match, focusing on the team "{target_team_name}".

    Match Context:
    - Date: {match_date_time}
    - Home: {local_name}
    - Opponent: {visitor_name}
    - Final Score: {score} ({local_name} vs {visitor_name})

    Team Comparison ({target_team_name} vs {opponent_team_name}):
    {team_stats_summary}

    Top Scorers ({target_team_name}):
    {top_players_summary}
    {quarters_section}
    {momentum_section}
    Instructions:
    - Use a neutral and analytical tone, like a coach's or media summary.
    - Emphasize "{target_team_name}" performance: scoring rhythm, defense, substitutions.
    - Mention key momentum phases (e.g., big runs, fouls, FT%, lead changes).
    - Highlight top scorer contributions in-game context.
    - Keep output to 2 short but informative paragraphs (~150 words total).
    """

    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY environment variable not set. Skipping LLM summary."
        )
        return None

    try:
        response = litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        summary_text = response.choices[0].message.content.strip()
        logger.info("LLM summary generated successfully.")
        return summary_text
    except Exception as e:
        logger.error(f"Failed to generate LLM summary using litellm: {e}")
        return None
