"""Models package for basketball reports."""

# Import models in dependency order to avoid circular imports
from .players import Player, PlayerStats
from .teams import Team, TeamStats
from .matches import Match, MatchMove, MatchStats, Scores, PlayerAggregate
from .groups import Group

# Rebuild models to resolve forward references after all imports
Player.model_rebuild()
Team.model_rebuild()
Match.model_rebuild()

__all__ = [
    "Player",
    "PlayerStats",
    "Match",
    "MatchMove",
    "MatchStats",
    "Scores",
    "PlayerAggregate",
    "Team",
    "TeamStats",
    "Group",
]
