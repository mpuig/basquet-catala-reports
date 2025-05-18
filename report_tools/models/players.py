from pydantic import BaseModel, Field, ConfigDict


class PlayerStats(BaseModel):
    """Player statistics model."""

    shots_of_two_successful: int = Field(
        default=0,
        description="Successful two-point shots",
        alias="shotsOfTwoSuccessful",
    )
    shots_of_two_attempted: int = Field(
        default=0, description="Attempted two-point shots", alias="shotsOfTwoAttempted"
    )
    shots_of_three_successful: int = Field(
        default=0,
        description="Successful three-point shots",
        alias="shotsOfThreeSuccessful",
    )
    shots_of_three_attempted: int = Field(
        default=0,
        description="Attempted three-point shots",
        alias="shotsOfThreeAttempted",
    )
    shots_of_one_successful: int = Field(
        default=0, description="Successful free throws", alias="shotsOfOneSuccessful"
    )
    shots_of_one_attempted: int = Field(
        default=0, description="Attempted free throws", alias="shotsOfOneAttempted"
    )
    faults: int = Field(default=0, description="Total player faults", alias="faults")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "shotsOfTwoSuccessful": 1,
                "shotsOfTwoAttempted": 1,
                "shotsOfThreeSuccessful": 0,
                "shotsOfThreeAttempted": 0,
                "shotsOfOneSuccessful": 0,
                "shotsOfOneAttempted": 0,
                "faults": 11,
            }
        },
    )
