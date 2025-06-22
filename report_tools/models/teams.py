from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from report_tools.models.players import Player


class TeamScore(BaseModel):
    """Team scoring statistics."""

    score_for: int = Field(alias="ScoreFor")
    score_against: int = Field(alias="ScoreAgainst")

    model_config = ConfigDict(populate_by_name=True)


class TeamResults(BaseModel):
    """Team win/loss record."""

    wins: int = Field(default=0)
    losses: int = Field(default=0)

    model_config = ConfigDict(populate_by_name=True)


class TeamStats(BaseModel):
    """
    team statistics model.

    Fields:
        team_score: Team scoring statistics (TeamScore)
        team_results: Win/loss record (TeamResults)
        sum_matches: Total matches played (int)
        sum_shots_of_*: Shot statistics (int)
        sum_field_throw_of_*: Field goal statistics (int)
        sum_fouls*: Foul statistics (int)
        total_*: Aggregate statistics (int/float)
        *_avg_by_match: Per-game averages (float)
        club: Club name (str)
        team_name: Team name (str)
        team_id: Team ID (int)
        category_name: Category name (str)
    """

    team_score: TeamScore = Field(alias="teamScore")
    team_results: TeamResults = Field(alias="teamResults")
    sum_matches: int = Field(alias="sumMatches")

    # Shot statistics
    sum_shots_of_one_attempted: int = Field(alias="sumShotsOfOneAttempted")
    sum_shots_of_two_attempted: int = Field(alias="sumShotsOfTwoAttempted")
    sum_shots_of_three_attempted: int = Field(alias="sumShotsOfThreeAttempted")
    sum_field_throw_of_one_attempted: int = Field(alias="sumFieldThrowOfOneAttempted")
    sum_shots_of_one_successful: int = Field(alias="sumShotsOfOneSuccessful")
    sum_shots_of_two_successful: int = Field(alias="sumShotsOfTwoSuccessful")
    sum_shots_of_three_successful: int = Field(alias="sumShotsOfThreeSuccessful")
    sum_field_throw_of_one_successful: int = Field(alias="sumFieldThrowOfOneSuccessful")
    sum_shots_of_one_failed: int = Field(alias="sumShotsOfOneFailed")
    sum_shots_of_two_failed: int = Field(alias="sumShotsOfTwoFailed")
    sum_shots_of_three_failed: int = Field(alias="sumShotsOfThreeFailed")

    # Foul statistics
    sum_fouls: int = Field(alias="sumFouls")
    sum_fouls_received: int = Field(alias="sumFoulsReceived")

    # Aggregate statistics
    total_score: int = Field(alias="totalScore")
    total_valoration: int = Field(alias="totalValoration")

    # Per-game averages
    total_score_avg_by_match: float = Field(alias="totalScoreAvgByMatch")
    total_fouls_avg_by_match: float = Field(alias="totalFoulsAvgByMatch")
    total_fouls_received_avg_by_match: float = Field(
        alias="totalFoulsReceivedAvgByMatch"
    )
    total_valoration_avg_by_match: float = Field(alias="totalValorationAvgByMatch")
    shots_of_one_successful_avg_by_match: float = Field(
        alias="shotsOfOneSuccessfulAvgByMatch"
    )
    shots_of_two_successful_avg_by_match: float = Field(
        alias="shotsOfTwoSuccessfulAvgByMatch"
    )
    shots_of_three_successful_avg_by_match: float = Field(
        alias="shotsOfThreeSuccessfulAvgByMatch"
    )
    field_throw_of_one_successful_avg_by_match: float = Field(
        alias="fieldThrowOfOneSuccessfulAvgByMatch"
    )

    # Team metadata
    club: str = Field()
    team_name: str = Field(alias="teamName")
    team_id: int = Field(alias="teamId")
    category_name: str = Field(alias="categoryName")

    model_config = ConfigDict(populate_by_name=True)


class Team(BaseModel):
    """
    Represents a basketball team with statistics.

    Fields:
        id: Team ID (int)
        name: Team name (str)
        short_name: Team short name (str)
        stats: Optional team statistics (TeamStats)
        players: Optional list of players (List[Player])
    """

    id: int = Field(description="Team ID")
    name: str = Field(description="Team name")
    short_name: str = Field(description="Team short name")
    stats: Optional[TeamStats] = Field(default=None, description="Team statistics")
    players: Optional[List["Player"]] = Field(default=None, description="Team players")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"example": {"id": 1, "name": "Team A", "short_name": "TA"}},
    )
