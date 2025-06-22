from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

if TYPE_CHECKING:
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
    PERSONAL_FOUL_SIXTH = "Desqualificant de partit, 6foul"

    UNSPORTSMANLIKE_FOUL_FIRST = "Antiesportiva 2 tirs lliures, 1a falta"
    UNSPORTSMANLIKE_FOUL_SECOND = "Antiesportiva 2 tirs lliures, 2a falta"
    UNSPORTSMANLIKE_FOUL_THIRD = "Antiesportiva 2 tirs lliures, 3a falta"
    UNSPORTSMANLIKE_FOUL_FOURTH = "Antiesportiva 2 tirs lliures, 4a falta"
    UNSPORTSMANLIKE_FOUL_FIFTH = "Antiesportiva 2 tirs lliures, 5a falta"

    # Additional foul type for attack fouls
    OFFENSIVE_FOUL_FIRST = "Falta en atac, 1a falta"
    OFFENSIVE_FOUL_SECOND = "Falta en atac, 2a falta"
    OFFENSIVE_FOUL_THIRD = "Falta en atac, 3a falta"
    OFFENSIVE_FOUL_FOURTH = "Falta en atac, 4a falta"
    OFFENSIVE_FOUL_FIFTH = "Falta en atac, 5a falta"

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

    PERSONAL_FOUL_THREE_FREE_THROWS_FIRST = "Personal 3 tirs lliures, 1a falta"
    PERSONAL_FOUL_THREE_FREE_THROWS_SECOND = "Personal 3 tirs lliures, 2a falta"
    PERSONAL_FOUL_THREE_FREE_THROWS_THIRD = "Personal 3 tirs lliures, 3a falta"
    PERSONAL_FOUL_THREE_FREE_THROWS_FOURTH = "Personal 3 tirs lliures, 4a falta"
    PERSONAL_FOUL_THREE_FREE_THROWS_FIFTH = "Personal 3 tirs lliures, 5a falta"

    # Player substitutions
    PLAYER_ENTERS = "Entra al camp"
    PLAYER_EXITS = "Surt del camp"

    # Special game events
    TIMEOUT = "Temps mort"
    PERIOD_END = "Final de període"

    DUNK = "Esmaixada"
    THREE_POINTER = "Triple"

    # Turnover
    TURNOVER = "Pèrdua"

    # Coach technical foul (1 free throw)
    COACH_TECHNICAL_ONE_FREE_THROW = "Tècnica entrenador 1 tir lliure"

    # Bench technical foul (1 free throw)
    BENCH_TECHNICAL_ONE_FREE_THROW = "Tècnica banqueta 1 tir lliure"

    # Player technical foul (1 free throw, with foul number)
    TECHNICAL_FOUL_ONE_FREE_THROW_FIRST = "Falta tècnica 1 tir lliure, 1a falta"
    TECHNICAL_FOUL_ONE_FREE_THROW_SECOND = "Falta tècnica 1 tir lliure, 2a falta"
    TECHNICAL_FOUL_ONE_FREE_THROW_THIRD = "Falta tècnica 1 tir lliure, 3a falta"
    TECHNICAL_FOUL_ONE_FREE_THROW_FOURTH = "Falta tècnica 1 tir lliure, 4a falta"
    TECHNICAL_FOUL_ONE_FREE_THROW_FIFTH = "Falta tècnica 1 tir lliure, 5a falta"


FOUL_MOVES = {
    MoveType.PERSONAL_FOUL_ONE_FREE_THROW_FIRST,
    MoveType.PERSONAL_FOUL_ONE_FREE_THROW_SECOND,
    MoveType.PERSONAL_FOUL_ONE_FREE_THROW_THIRD,
    MoveType.PERSONAL_FOUL_ONE_FREE_THROW_FOURTH,
    MoveType.PERSONAL_FOUL_ONE_FREE_THROW_FIFTH,
    MoveType.PERSONAL_FOUL_FIRST,
    MoveType.PERSONAL_FOUL_SECOND,
    MoveType.PERSONAL_FOUL_THIRD,
    MoveType.PERSONAL_FOUL_FOURTH,
    MoveType.PERSONAL_FOUL_FIFTH,
    MoveType.PERSONAL_FOUL_SIXTH,
    MoveType.UNSPORTSMANLIKE_FOUL_FIRST,
    MoveType.UNSPORTSMANLIKE_FOUL_SECOND,
    MoveType.UNSPORTSMANLIKE_FOUL_THIRD,
    MoveType.UNSPORTSMANLIKE_FOUL_FOURTH,
    MoveType.UNSPORTSMANLIKE_FOUL_FIFTH,
    MoveType.PERSONAL_FOUL_THREE_FREE_THROWS_FIRST,
    MoveType.PERSONAL_FOUL_THREE_FREE_THROWS_SECOND,
    MoveType.PERSONAL_FOUL_THREE_FREE_THROWS_THIRD,
    MoveType.PERSONAL_FOUL_THREE_FREE_THROWS_FOURTH,
    MoveType.PERSONAL_FOUL_THREE_FREE_THROWS_FIFTH,
    MoveType.OFFENSIVE_FOUL_FIRST,
    MoveType.OFFENSIVE_FOUL_SECOND,
    MoveType.OFFENSIVE_FOUL_THIRD,
    MoveType.OFFENSIVE_FOUL_FOURTH,
    MoveType.OFFENSIVE_FOUL_FIFTH,
    MoveType.BENCH_TECHNICAL_ONE_FREE_THROW,
    MoveType.TECHNICAL_FOUL_ONE_FREE_THROW_FIRST,
    MoveType.TECHNICAL_FOUL_ONE_FREE_THROW_SECOND,
    MoveType.TECHNICAL_FOUL_ONE_FREE_THROW_THIRD,
    MoveType.TECHNICAL_FOUL_ONE_FREE_THROW_FOURTH,
    MoveType.TECHNICAL_FOUL_ONE_FREE_THROW_FIFTH,
}


class MatchMove(BaseModel):
    """
    Represents a single move/event in a basketball match.

    Fields:
        id_team: Team identifier (int)
        actor_name: Name of the player who performed the move (str)
        actor_id: ID of the player who performed the move (int)
        actor_shirt_number: Shirt number of the player (as integer)
        id_move: Move identifier (int)
        move: Type of move (MoveType)
        min: Minute of the move (int)
        sec: Second of the move (int)
        period: Period of the move (int)
        score: Current score when the move occurred (str)
        team_action: Whether the move was performed by a team (bool)
        event_uuid: Unique identifier for the event (str)
        timestamp: Timestamp of the move (datetime or None)
        foul_number: Foul number if applicable (int or None)
        license_id: License ID of the player (int or None)
    """

    id_team: int = Field(..., description="Team identifier", alias="idTeam")
    actor_name: str = Field(
        ..., description="Name of the player who performed the move", alias="actorName"
    )
    actor_id: int = Field(
        ..., description="ID of the player who performed the move", alias="actorId"
    )
    actor_shirt_number: int = Field(
        ...,
        description="Shirt number of the player (as integer)",
        alias="actorShirtNumber",
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

    @field_validator("actor_shirt_number", mode="before")
    @classmethod
    def cast_shirt_number(cls, v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        if isinstance(v, datetime) or v is None:
            return v
        try:
            return datetime.strptime(v, "%Y%m%d%H%M%S")
        except Exception:
            return None

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "idTeam": 1,
                "actorName": "John Doe",
                "actorId": 123,
                "actorShirtNumber": 10,
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

        In basketball, the clock counts DOWN from 10:00 to 0:00 each period.
        So self.min and self.sec represent TIME REMAINING in the current period.
        We need to convert this to elapsed time.
        """
        period_duration_seconds = 600  # Standard 10-minute period
        elapsed_in_prior_periods = (self.period - 1) * period_duration_seconds
        # Convert remaining time to elapsed time: elapsed = 10:00 - remaining
        time_remaining_in_period = self.min * 60 + self.sec
        elapsed_in_current_period = period_duration_seconds - time_remaining_in_period
        return elapsed_in_prior_periods + elapsed_in_current_period


class MatchStats(BaseModel):
    """
    Aggregated statistics for a basketball match.

    Fields:
        id_match_intern: Internal match ID (str or int)
        id_match_extern: External match ID (int or None)
        local_id: Local team ID (int)
        visit_id: Visitor team ID (int)
        period: Number of periods (int)
        period_duration: Duration of each period in minutes (int)
        sum_period: Total match duration in minutes (int)
        period_duration_list: Duration of each period (list of int)
        score: Match score history (list of dict)
        moves: Match moves history (list of dict)
        shots_of_two_successful: Successful two-point shots (int)
        shots_of_two_attempted: Attempted two-point shots (int)
        shots_of_three_successful: Successful three-point shots (int)
        shots_of_three_attempted: Attempted three-point shots (int)
        shots_of_one_successful: Successful free throws (int)
        shots_of_one_attempted: Attempted free throws (int)
        faults: Total team faults (int)
        team_points_for: Points scored by the team (int)
        team_points_against: Points scored against the team (int)
        local_stats: Local team stats (dict)
        visitor_stats: Visitor team stats (dict)
    """

    id_match_intern: Union[str, int] = Field(description="Internal match ID")
    id_match_extern: Optional[int] = Field(
        default=None, description="External match ID"
    )
    local_id: int = Field(description="Local team ID")
    visit_id: int = Field(description="Visitor team ID")
    period: int = Field(default=4, description="Number of periods")
    period_duration: int = Field(
        default=10, description="Duration of each period in minutes"
    )
    sum_period: int = Field(default=40, description="Total match duration in minutes")
    period_duration_list: List[int] = Field(
        default=[10, 10, 10, 10], description="Duration of each period"
    )
    score: List[Dict[str, Any]] = Field(
        default_factory=list, description="Match score history"
    )
    moves: List[Dict[str, Any]] = Field(
        default_factory=list, description="Match moves history"
    )
    shots_of_two_successful: int = Field(
        default=0, description="Successful two-point shots"
    )
    shots_of_two_attempted: int = Field(
        default=0, description="Attempted two-point shots"
    )
    shots_of_three_successful: int = Field(
        default=0, description="Successful three-point shots"
    )
    shots_of_three_attempted: int = Field(
        default=0, description="Attempted three-point shots"
    )
    shots_of_one_successful: int = Field(
        default=0, description="Successful free throws"
    )
    shots_of_one_attempted: int = Field(default=0, description="Attempted free throws")
    faults: int = Field(default=0, description="Total team faults")
    team_points_for: int = Field(default=0, description="Points scored by the team")
    team_points_against: int = Field(
        default=0, description="Points scored against the team"
    )
    local_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Local team stats"
    )
    visitor_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Visitor team stats"
    )

    @model_validator(mode="before")
    @classmethod
    def preprocess(cls, data):
        import re

        def camel_to_snake(name):
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

        if isinstance(data, dict):
            return {camel_to_snake(k): v for k, v in data.items()}
        return data

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "id_match_intern": None,
                "id_match_extern": None,
                "local_id": 1,
                "visit_id": 2,
                "period": 4,
                "period_duration": 10,
                "sum_period": 40,
                "period_duration_list": [10, 10, 10, 10],
                "score": [],
                "moves": [],
                "shots_of_two_successful": 20,
                "shots_of_two_attempted": 40,
                "shots_of_three_successful": 8,
                "shots_of_three_attempted": 20,
                "shots_of_one_successful": 12,
                "shots_of_one_attempted": 15,
                "faults": 18,
                "team_points_for": 80,
                "team_points_against": 75,
            }
        },
    )


class EventTime(BaseModel):
    """Event timing information."""

    minute: int = Field(default=0)
    second: int = Field(default=0)

    model_config = ConfigDict(populate_by_name=True)


class InOutEntry(BaseModel):
    """Player substitution entry."""

    type: str = Field(description="IN_TYPE or OUT_TYPE")
    minute_absolut: int = Field(alias="minuteAbsolut")
    point_diff: int = Field(alias="pointDiff")

    model_config = ConfigDict(populate_by_name=True)


class PlayerStatsData(BaseModel):
    """Player statistics within a match."""

    type: int = Field(default=0)
    score: int = Field(default=0)
    valoration: int = Field(default=0)
    shots_of_one_attempted: int = Field(default=0, alias="shotsOfOneAttempted")
    shots_of_two_attempted: int = Field(default=0, alias="shotsOfTwoAttempted")
    shots_of_three_attempted: int = Field(default=0, alias="shotsOfThreeAttempted")
    shots_of_one_successful: int = Field(default=0, alias="shotsOfOneSuccessful")
    shots_of_two_successful: int = Field(default=0, alias="shotsOfTwoSuccessful")
    shots_of_three_successful: int = Field(default=0, alias="shotsOfThreeSuccessful")

    model_config = ConfigDict(populate_by_name=True)


class MatchPlayer(BaseModel):
    """Player within a match with detailed stats."""

    actor_id: int = Field(alias="actorId")
    uuid: str = Field()
    player_ids_interns: List[int] = Field(alias="playerIdsInterns")
    team_id: int = Field(alias="teamId")
    name: str = Field()
    dorsal: str = Field()
    starting: bool = Field(default=False)
    captain: bool = Field(default=False)
    sum_period: int = Field(alias="sumPeriod")
    period: int = Field()
    period_duration: int = Field(alias="periodDuration")
    time_played: int = Field(alias="timePlayed")
    in_outs_list: List[InOutEntry] = Field(default_factory=list, alias="inOutsList")
    game_played: int = Field(alias="gamePlayed")
    in_out: int = Field(alias="inOut")
    match_has_starting_players: bool = Field(alias="matchHasStartingPlayers")
    team_score: int = Field(alias="teamScore")
    opp_score: int = Field(alias="oppScore")
    data: PlayerStatsData = Field()
    periods: List[PlayerStatsData] = Field(default_factory=list)
    event_time: EventTime = Field(alias="eventTime")

    model_config = ConfigDict(populate_by_name=True)


class MatchTeam(BaseModel):
    """Team within a match with detailed stats."""

    team_id_intern: int = Field(alias="teamIdIntern")
    team_id_extern: int = Field(alias="teamIdExtern")
    color_rgb: str = Field(alias="colorRgb")
    name: str = Field()
    short_name: str = Field(alias="shortName")
    fede: str = Field()
    players: List[MatchPlayer] = Field(default_factory=list)
    data: PlayerStatsData = Field()
    periods: List[PlayerStatsData] = Field(default_factory=list)
    event_time: EventTime = Field(alias="eventTime")

    model_config = ConfigDict(populate_by_name=True)


class ScoreEntry(BaseModel):
    """Score entry at a specific moment."""

    local: int = Field()
    visit: int = Field()
    minute_quarter: int = Field(alias="minuteQuarter")
    minute_absolute: int = Field(alias="minuteAbsolute")
    period: int = Field()

    model_config = ConfigDict(populate_by_name=True)


class Match(BaseModel):
    """
    Represents a complete basketball match with all statistics.

    Fields:
        id: Match identifier (str) - derived from idMatchIntern
        match_date: Date of the match (str) - for compatibility
        group_name: Name of the group/competition (str) - for compatibility
        id_match_intern: Internal match ID (int)
        id_match_extern: External match ID (int)
        time: Match time string (str)
        local_id: Local team ID (int)
        visit_id: Visitor team ID (int)
        period: Number of periods (int)
        period_duration: Duration of each period (int)
        sum_period: Total match duration (int)
        period_duration_list: List of period durations (List[int])
        last_minute_used: Last minute used (int)
        moves: List of match moves (List[MatchMove])
        recalculated: Whether match was recalculated (bool)
        score: Score progression (List[ScoreEntry])
        teams: Teams with detailed stats (List[MatchTeam])
        data: Match-level aggregated stats (PlayerStatsData)
        periods: Period-by-period stats (List[PlayerStatsData])
        event_time: Event timing (EventTime)

        # Legacy compatibility fields
        local: Local team (Team) - derived from teams[0]
        visitor: Visitor team (Team) - derived from teams[1]
        final_score: Final score string (str) - derived from last score entry
        stats: Optional match statistics (MatchStats or None) - for compatibility
    """

    # Core match data
    id_match_intern: int = Field(alias="idMatchIntern")
    id_match_extern: int = Field(alias="idMatchExtern")
    time: str = Field(default="")
    local_id: int = Field(alias="localId")
    visit_id: int = Field(alias="visitId")
    period: int = Field(default=4)
    period_duration: int = Field(default=10, alias="periodDuration")
    sum_period: int = Field(default=40, alias="sumPeriod")
    period_duration_list: List[int] = Field(
        default_factory=list, alias="periodDurationList"
    )
    last_minute_used: int = Field(default=1, alias="lastMinuteUsed")
    moves: List[MatchMove] = Field(default_factory=list)
    recalculated: bool = Field(default=False)
    score: List[ScoreEntry] = Field(default_factory=list)
    teams: List[MatchTeam] = Field(default_factory=list)
    data: PlayerStatsData = Field(default_factory=PlayerStatsData)
    periods: List[PlayerStatsData] = Field(default_factory=list)
    event_time: EventTime = Field(default_factory=EventTime, alias="eventTime")

    # Legacy compatibility fields
    id: str = Field(default="", description="Match identifier")
    match_date: str = Field(default="", description="Date of the match")
    group_name: str = Field(default="", description="Name of the group/competition")
    local: Optional["Team"] = Field(default=None, description="Local team")
    visitor: Optional["Team"] = Field(default=None, description="Visitor team")
    final_score: str = Field(default="", description="Final score")
    stats: Optional[MatchStats] = Field(
        default=None, description="Optional match statistics"
    )

    model_config = ConfigDict(
        populate_by_name=True,
    )

    def model_post_init(self, __context) -> None:
        """Set legacy compatibility fields after initialization."""
        from report_tools.models.teams import Team

        # Set id from internal match ID
        if not self.id:
            self.id = str(self.id_match_intern)

        # Set final score from last score entry
        if not self.final_score and self.score:
            last_score = self.score[-1]
            self.final_score = f"{last_score.local}-{last_score.visit}"

        # Create legacy Team objects from MatchTeam data
        if not self.local and len(self.teams) > 0:
            local_team = self.teams[0]
            self.local = Team(
                id=local_team.team_id_extern,
                name=local_team.name,
                short_name=local_team.short_name,
            )

        if not self.visitor and len(self.teams) > 1:
            visitor_team = self.teams[1]
            self.visitor = Team(
                id=visitor_team.team_id_extern,
                name=visitor_team.name,
                short_name=visitor_team.short_name,
            )


class Scores(BaseModel):
    """
    Aggregated scoring statistics for a team or player.

    Fields:
        avg_ppg: Average points per game (float)
        avg_t1: Average free throws made per game (float)
        avg_t2: Average two-point shots made per game (float)
        avg_t3: Average three-point shots made per game (float)
        avg_fouls: Average fouls per game (float)
        score: Total score (int)
        t1: Total free throws made (int)
        t2: Total two-point shots made (int)
        t3: Total three-point shots made (int)
        faults: Total fouls (int)
    """

    avg_ppg: float = Field(description="Average points per game")
    avg_t1: float = Field(description="Average free throws made per game")
    avg_t2: float = Field(description="Average two-point shots made per game")
    avg_t3: float = Field(description="Average three-point shots made per game")
    avg_fouls: float = Field(description="Average fouls per game")
    score: int = Field(description="Total score")
    t1: int = Field(description="Total free throws made")
    t2: int = Field(description="Total two-point shots made")
    t3: int = Field(description="Total three-point shots made")
    faults: int = Field(description="Total fouls")

    model_config = ConfigDict(
        populate_by_name=True,
    )


class PlayerAggregate(BaseModel):
    """
    Holds aggregated statistics for a single player across matches.

    Fields:
        name: Player name (str)
        number: Player shirt number (str)
        gp: Games played (int)
        secs_played: Total seconds played (float)
        pts: Total points scored (int)
        t3: Total three-point shots made (int)
        t2: Total two-point shots made (int)
        t1: Total free throws made (int)
        fouls: Total fouls committed (int)
    """

    name: str = Field(description="Player name")
    number: str = Field(default="??", description="Player shirt number")
    gp: int = Field(default=0, description="Games played")
    secs_played: float = Field(default=0.0, description="Total seconds played")
    pts: int = Field(default=0, description="Total points scored")
    t3: int = Field(default=0, description="Total three-point shots made")
    t2: int = Field(default=0, description="Total two-point shots made")
    t1: int = Field(default=0, description="Total free throws made")
    fouls: int = Field(default=0, description="Total fouls committed")

    model_config = ConfigDict(
        populate_by_name=True,
    )

    def merge_secs(self, secs: float) -> None:
        self.secs_played += secs

    @property
    def minutes(self) -> int:
        return round(self.secs_played / 60)
