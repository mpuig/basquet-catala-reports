from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, ConfigDict

from report_tools.models.teams import Team


class MoveType(str, Enum):
    """Types of moves that can occur in a match."""

    # Regular game moves
    JUMP_BALL_WON = "Salt guanyat"
    JUMP_BALL_LOST = "Salt perdut"
    FREE_THROW_MADE = "Cistella de 1"
    TWO_POINT_MADE = "Cistella de 2"
    THREE_POINT_MADE = "Cistella de 3"
    FREE_THROW_MISSED = "Intent fallat de 1"
    TWO_POINT_MISSED = "Intent fallat de 2"
    THREE_POINT_MISSED = "Intent fallat de 3"

    # Fouls
    PERSONAL_FOUL_FIRST = "Personal, 1a falta"
    PERSONAL_FOUL_SECOND = "Personal, 2a falta"
    PERSONAL_FOUL_THIRD = "Personal, 3a falta"
    PERSONAL_FOUL_FOURTH = "Personal, 4a falta"
    PERSONAL_FOUL_FIFTH = "Personal, 5a falta"

    # Free throws after fouls
    PERSONAL_FOUL_ONE_FREE_THROW_FIRST = "Personal 1 tir lliure, 1a falta"
    PERSONAL_FOUL_ONE_FREE_THROW_SECOND = "Personal 1 tir lliure, 2a falta"
    PERSONAL_FOUL_ONE_FREE_THROW_THIRD = "Personal 1 tir lliure, 3a falta"
    PERSONAL_FOUL_ONE_FREE_THROW_FOURTH = "Personal 1 tir lliure, 4a falta"
    PERSONAL_FOUL_ONE_FREE_THROW_FIFTH = "Personal 1 tir lliure, 5a falta"

    PERSONAL_FOUL_TWO_FREE_THROWS_FIRST = "Personal 2 tirs lliures, 1a falta"
    PERSONAL_FOUL_TWO_FREE_THROWS_SECOND = "Personal 2 tirs lliures, 2a falta"
    PERSONAL_FOUL_TWO_FREE_THROWS_THIRD = "Personal 2 tirs lliures, 3a falta"
    PERSONAL_FOUL_TWO_FREE_THROWS_FOURTH = "Personal 2 tirs lliures, 4a falta"
    PERSONAL_FOUL_TWO_FREE_THROWS_FIFTH = "Personal 2 tirs lliures, 5a falta"

    # Player substitutions
    PLAYER_ENTERS = "Entra al camp"
    PLAYER_EXITS = "Surt del camp"

    # Special game events
    TIMEOUT = "Temps mort"
    PERIOD_END = "Final de període"


class MatchMove(BaseModel):
    """Represents a single move/event in a match."""

    id_team: int = Field(..., description="Team identifier", alias="idTeam")
    actor_name: str = Field(
        ..., description="Name of the player who performed the move", alias="actorName"
    )
    actor_id: int = Field(
        ..., description="ID of the player who performed the move", alias="actorId"
    )
    actor_shirt_number: str = Field(
        ..., description="Shirt number of the player", alias="actorShirtNumber"
    )
    id_move: int = Field(..., description="Move identifier", alias="idMove")
    move: MoveType = Field(..., description="Type of move")
    min: int = Field(..., description="Minute of the move")
    sec: int = Field(..., description="Second of the move")
    period: int = Field(..., description="Period of the move")
    score: str = Field(..., description="Current score when the move occurred")
    team_action: bool = Field(
        ..., description="Whether the move was performed by a team", alias="teamAction"
    )
    event_uuid: str = Field(
        ..., description="Unique identifier for the event", alias="eventUuid"
    )
    timestamp: Optional[datetime] = Field(None, description="Timestamp of the move")
    foul_number: Optional[int] = Field(
        None, description="Foul number if applicable", alias="foulNumber"
    )
    license_id: Optional[int] = Field(
        None, description="License ID of the player", alias="licenseId"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "idTeam": 1,
                "actorName": "John Doe",
                "actorId": 123,
                "actorShirtNumber": "10",
                "idMove": 1,
                "move": "Cistella de 2",
                "min": 5,
                "sec": 30,
                "period": 1,
                "score": "2-0",
                "teamAction": True,
                "eventUuid": "abc123",
                "timestamp": "20240314150000",
                "foulNumber": None,
                "licenseId": None,
            }
        },
    )

    def get_absolute_seconds(self) -> int:
        """Calculate the absolute time in seconds from the start of the match.

        For a 10-minute period (600 seconds), this calculates the time remaining
        in the period. For example, at 5:30 in period 1, there are 270 seconds
        remaining (600 - (5*60 + 30) = 270).
        """
        period_len = 600  # 10 minutes in seconds
        return (self.period - 1) * period_len + (
            period_len - (self.min * 60 + self.sec)
        )


class MatchStats(BaseModel):
    """Match statistics model."""

    id_match_intern: Optional[int] = Field(
        default=None,
        description="Internal match ID",
        alias="idMatchIntern",
    )
    id_match_extern: Optional[int] = Field(
        default=None,
        description="External match ID",
        alias="idMatchExtern",
    )
    local_id: Optional[int] = Field(
        default=None,
        description="Local team ID",
        alias="localId",
    )
    visit_id: Optional[int] = Field(
        default=None,
        description="Visitor team ID",
        alias="visitId",
    )
    period: Optional[int] = Field(
        default=None,
        description="Total number of periods",
    )
    period_duration: Optional[int] = Field(
        default=None,
        description="Duration of each period in minutes",
        alias="periodDuration",
    )
    sum_period: Optional[int] = Field(
        default=None,
        description="Total minutes played",
        alias="sumPeriod",
    )
    period_duration_list: Optional[List[int]] = Field(
        default=None,
        description="List of period durations in minutes",
        alias="periodDurationList",
    )
    score: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Score progression throughout the match",
    )
    moves: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of moves in the match",
    )
    shots_of_two_successful: Optional[int] = Field(
        default=None,
        description="Successful two-point shots",
        alias="shotsOfTwoSuccessful",
    )
    shots_of_two_attempted: Optional[int] = Field(
        default=None,
        description="Attempted two-point shots",
        alias="shotsOfTwoAttempted",
    )
    shots_of_three_successful: Optional[int] = Field(
        default=None,
        description="Successful three-point shots",
        alias="shotsOfThreeSuccessful",
    )
    shots_of_three_attempted: Optional[int] = Field(
        default=None,
        description="Attempted three-point shots",
        alias="shotsOfThreeAttempted",
    )
    shots_of_one_successful: Optional[int] = Field(
        default=None, description="Successful free throws", alias="shotsOfOneSuccessful"
    )
    shots_of_one_attempted: Optional[int] = Field(
        default=None, description="Attempted free throws", alias="shotsOfOneAttempted"
    )
    faults: Optional[int] = Field(
        default=None, description="Total match faults", alias="faults"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "idMatchIntern": 158215,
                "idMatchExtern": 928403,
                "localId": 319130,
                "visitId": 319131,
                "period": 4,
                "periodDuration": 10,
                "sumPeriod": 30,
                "periodDurationList": [10, 10, 10, 10],
                "score": [],
                "moves": [],
                "shotsOfTwoSuccessful": 15,
                "shotsOfTwoAttempted": 30,
                "shotsOfThreeSuccessful": 8,
                "shotsOfThreeAttempted": 20,
                "shotsOfOneSuccessful": 12,
                "shotsOfOneAttempted": 15,
                "faults": 18,
            }
        },
    )


@dataclass
class Match:
    """Represents a basketball match."""

    id: str
    match_date: str
    group_name: str
    local: Team
    visitor: Team
    score: str
    moves: List[MatchMove]
    stats: Optional[MatchStats] = None


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


@dataclass
class PlayerAggregate:
    """Holds aggregated statistics for a single player."""

    name: str
    number: str = "??"
    gp: int = 0
    secs_played: float = 0.0
    pts: int = 0
    t3: int = 0
    t2: int = 0
    t1: int = 0
    fouls: int = 0

    def merge_secs(self, secs: float) -> None:
        self.secs_played += secs

    @property
    def minutes(self) -> int:
        return round(self.secs_played / 60)
