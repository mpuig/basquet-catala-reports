from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class PlayerEvolutiveStats(BaseModel):
    """Single match evolution entry for a player."""

    # Basic stats
    total_score: int = Field(alias="totalScore")
    total_score_percent: float = Field(alias="totalScorePercent")
    total_score_avg_by_match: float = Field(alias="totalScoreAvgByMatch")
    avg_team_percent: float = Field(alias="avgTeamPercent")
    time_total_score: float = Field(alias="timeTotalScore")

    # Free throw stats
    sum_shots_of_one_successful: int = Field(alias="sumShotsOfOneSuccessful")
    sum_shots_of_one_attempted: int = Field(alias="sumShotsOfOneAttempted")
    sum_shots_of_one_failed: int = Field(alias="sumShotsOfOneFailed")
    sum_shots_of_one_successful_percent: float = Field(
        alias="sumShotsOfOneSuccessfulPercent"
    )
    sum_shots_of_one_successful_avg_by_match: float = Field(
        alias="sumShotsOfOneSuccessfulAvgByMatch"
    )
    time_sum_shots_of_one_successful: float = Field(alias="timeSumShotsOfOneSuccessful")

    # Two-point shot stats
    sum_shots_of_two_successful: int = Field(alias="sumShotsOfTwoSuccessful")
    sum_shots_of_two_attempted: int = Field(alias="sumShotsOfTwoAttempted")
    sum_shots_of_two_failed: int = Field(alias="sumShotsOfTwoFailed")
    sum_shots_of_two_successful_percent: float = Field(
        alias="sumShotsOfTwoSuccessfulPercent"
    )
    sum_shots_of_two_successful_avg_by_match: float = Field(
        alias="sumShotsOfTwoSuccessfulAvgByMatch"
    )
    time_sum_shots_of_two_successful: float = Field(alias="timeSumShotsOfTwoSuccessful")

    # Three-point shot stats
    sum_shots_of_three_successful: int = Field(alias="sumShotsOfThreeSuccessful")
    sum_shots_of_three_attempted: int = Field(alias="sumShotsOfThreeAttempted")
    sum_shots_of_three_failed: int = Field(alias="sumShotsOfThreeFailed")
    sum_shots_of_three_successful_percent: float = Field(
        alias="sumShotsOfThreeSuccessfulPercent"
    )
    sum_shots_of_three_successful_avg_by_match: float = Field(
        alias="sumShotsOfThreeSuccessfulAvgByMatch"
    )
    time_sum_shots_of_three_successful: float = Field(
        alias="timeSumShotsOfThreeSuccessful"
    )

    # Field throw stats
    sum_field_throw_of_one_successful: int = Field(alias="sumFieldThrowOfOneSuccessful")
    sum_field_throw_of_one_attempted: int = Field(alias="sumFieldThrowOfOneAttempted")
    sum_field_throw_of_one_failed: int = Field(alias="sumFieldThrowOfOneFailed")
    sum_field_throw_of_one_successful_percent: float = Field(
        alias="sumFieldThrowOfOneSuccessfulPercent"
    )
    sum_field_throw_of_one_successful_avg_by_match: float = Field(
        alias="sumFieldThrowOfOneSuccessfulAvgByMatch"
    )
    time_sum_field_throw_of_one_successful: float = Field(
        alias="timeSumFieldThrowOfOneSuccessful"
    )

    # Foul stats
    sum_fouls_received: int = Field(alias="sumFoulsReceived")
    time_sum_fouls_received: float = Field(alias="timeSumFoulsReceived")
    sum_fouls_received_avg_by_match: float = Field(alias="sumFoulsReceivedAvgByMatch")
    sum_fouls: int = Field(alias="sumFouls")
    time_sum_fouls: float = Field(alias="timeSumFouls")
    sum_fouls_avg_by_match: float = Field(alias="sumFoulsAvgByMatch")

    # Other stats
    sum_assists: int = Field(alias="sumAssists")
    sum_assists_avg_by_match: float = Field(alias="sumAssistsAvgByMatch")
    sum_valoration: int = Field(alias="sumValoration")
    sum_valoration_avg_by_match: float = Field(alias="sumValorationAvgByMatch")
    sum_rebounds: int = Field(alias="sumRebounds")
    sum_rebounds_avg_by_match: float = Field(alias="sumReboundsAvgByMatch")
    sum_defensive_rebounds: int = Field(alias="sumDefensiveRebounds")
    sum_defensive_rebounds_avg_by_match: float = Field(
        alias="sumDefensiveReboundsAvgByMatch"
    )
    sum_offensive_rebounds: int = Field(alias="sumOffensiveRebounds")
    sum_offensive_rebounds_avg_by_match: float = Field(
        alias="sumOffensiveReboundsAvgByMatch"
    )

    # Playing time
    time_played: int = Field(alias="timePlayed")
    games_starter: int = Field(alias="gamesStarter")

    # Player info
    name: str = Field()
    club: str = Field()
    club_id: int = Field(alias="clubId")
    team_name: str = Field(alias="teamName")
    team_id: int = Field(alias="teamId")
    category: str = Field()
    matches_played: int = Field(alias="matchesPlayed")
    score_each_10_min: float = Field(alias="scoreEach10Min")
    dorsal: int = Field()

    # Match-specific fields (for evolution entries)
    opponent_team_name: Optional[str] = Field(default=None, alias="opponentTeamName")
    opponent_team_id: Optional[int] = Field(default=None, alias="opponentTeamId")
    match_call_uuid: Optional[str] = Field(default=None, alias="matchCallUuid")
    num_match_day: Optional[int] = Field(default=None, alias="numMatchDay")
    match_day: Optional[str] = Field(default=None, alias="matchDay")

    model_config = ConfigDict(populate_by_name=True)


class PlayerStats(BaseModel):
    """
    player statistics model.

    Contains both general stats and evolutive stats (match-by-match progression).
    The general_stats contains season totals and averages.
    The evolutive_stats contains match-by-match progression.
    """

    general_stats: PlayerEvolutiveStats = Field(alias="generalStats")
    evolutive_stats: List[PlayerEvolutiveStats] = Field(
        default_factory=list, alias="evolutiveStats"
    )

    model_config = ConfigDict(populate_by_name=True)


class Player(BaseModel):
    """
    Represents a basketball player with statistics.

    Fields:
        id: Player ID (int) - from team_id key in player_stats.json
        name: Player name (str) - from general_stats.name
        uuid: Player UUID (str) - from general_stats.uuid
        club: Club name (str)
        club_id: Club ID (int)
        team_name: Team name (str)
        team_id: Team ID (int)
        category: Category (str)
        dorsal: Jersey number (int)
        stats: player statistics (PlayerStats)
    """

    id: int = Field(description="Player ID")
    name: str = Field(description="Player name")
    uuid: Optional[str] = Field(default=None, description="Player UUID")
    club: Optional[str] = Field(default=None, description="Club name")
    club_id: Optional[int] = Field(default=None, description="Club ID")
    team_name: Optional[str] = Field(default=None, description="Team name")
    team_id: Optional[int] = Field(default=None, description="Team ID")
    category: Optional[str] = Field(default=None, description="Category")
    dorsal: Optional[int] = Field(default=None, description="Jersey number")
    stats: Optional[PlayerStats] = Field(default=None, description="Player statistics")

    model_config = ConfigDict(populate_by_name=True)
