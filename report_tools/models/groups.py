from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

import pandas as pd

from report_tools.models.matches import Match
from report_tools.models.teams import Team


class Group(BaseModel):
    """
    Represents a competition group in a basketball league or tournament.

    Fields:
        id: Group identifier (int)
        name: Name of the group (str)
        schedule: Optional schedule DataFrame for the group (pandas.DataFrame or None)
        teams: Optional list of teams in the group (List[Team] or None)
        matches: Optional list of matches in the group (List[Match] or None)
    """

    id: int = Field(description="Group identifier")
    name: str = Field(description="Name of the group")
    schedule: Optional[pd.DataFrame] = Field(
        default=None, description="Schedule DataFrame for the group"
    )
    teams: Optional[List[Team]] = Field(
        default=None, description="List of teams in the group"
    )
    matches: Optional[List[Match]] = Field(
        default=None, description="List of matches in the group"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )
