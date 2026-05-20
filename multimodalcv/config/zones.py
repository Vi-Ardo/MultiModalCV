"""Zone configuration helpers."""

from multimodalcv.core.models import Zone


class ZoneConfigError(ValueError):
    """Raised when a zone configuration value is invalid."""


def parse_zone_rect(value: str, *, name: str = "main") -> Zone:
    """Parse a rectangle zone from 'x1,y1,x2,y2' text."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ZoneConfigError("zone rectangle must use format x1,y1,x2,y2")

    try:
        x1, y1, x2, y2 = [float(part) for part in parts]
    except ValueError as error:
        raise ZoneConfigError("zone rectangle coordinates must be numbers") from error

    if x2 <= x1 or y2 <= y1:
        raise ZoneConfigError("zone rectangle must satisfy x2 > x1 and y2 > y1")

    return Zone(
        name=name,
        points=(
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ),
    )


def default_zone(*, name: str = "main") -> Zone:
    """Return the default MVP zone."""
    return parse_zone_rect("0,0,100,100", name=name)

