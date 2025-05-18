"""Group model for basketball competition groups."""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from report_tools.models.matches import Match
from report_tools.models.teams import Team


@dataclass
class Group:
    """Represents a competition group."""

    id: int
    name: str
    schedule: Optional[pd.DataFrame] = None
    teams: Optional[List[Team]] = None
    matches: Optional[List[Match]] = None
