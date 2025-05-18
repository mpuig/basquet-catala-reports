from dataclasses import dataclass

from pydantic import BaseModel, Field


class TeamStats(BaseModel):
    """Team statistics model."""

    score: int = Field(default=0, description="Total team score")
    shots_of_two_successful: int = Field(
        default=0,
        description="Successful two-point shots",
        alias="sumShotsOfTwoSuccessful",
    )
    shots_of_two_attempted: int = Field(
        default=0,
        description="Attempted two-point shots",
        alias="sumShotsOfTwoAttempted",
    )
    shots_of_three_successful: int = Field(
        default=0,
        description="Successful three-point shots",
        alias="sumShotsOfThreeSuccessful",
    )
    shots_of_three_attempted: int = Field(
        default=0,
        description="Attempted three-point shots",
        alias="sumShotsOfThreeAttempted",
    )
    shots_of_one_successful: int = Field(
        default=0, description="Successful free throws", alias="sumShotsOfOneSuccessful"
    )
    shots_of_one_attempted: int = Field(
        default=0, description="Attempted free throws", alias="sumShotsOfOneAttempted"
    )
    faults: int = Field(default=0, description="Total team faults", alias="sumFouls")

    class Config:
        """Pydantic model configuration."""

        allow_population_by_field_name = True
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "score": 80,
                "sumShotsOfTwoSuccessful": 20,
                "sumShotsOfTwoAttempted": 40,
                "sumShotsOfThreeSuccessful": 8,
                "sumShotsOfThreeAttempted": 20,
                "sumShotsOfOneSuccessful": 12,
                "sumShotsOfOneAttempted": 15,
                "sumFouls": 18,
            }
        }


@dataclass
class Team:
    """Represents a basketball team."""

    id: int
    name: str
