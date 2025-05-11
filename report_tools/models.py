from dataclasses import dataclass


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
