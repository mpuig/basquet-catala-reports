# Duration of a quarter in the Catalan U13 competition (10 minutes)
PERIOD_LENGTH_SEC: int = 600


def get_absolute_seconds(
    period: int, minute: int, second: int, *, period_len: int = PERIOD_LENGTH_SEC
) -> int:
    """Convert a (period, minute, second) timestamp into absolute seconds since game start."""
    return (period - 1) * period_len + (period_len - (minute * 60 + second))


def shorten_name(full_name: str) -> str:
    """
    Removes the last part of a full name.

    Example:
        'FIRST MIDDLE LAST' -> 'FIRST MIDDLE'
    """
    parts = full_name.split()
    return " ".join(parts[:-1]) if len(parts) > 1 else full_name
