"""Domain models for detections, tracks, zones, and events."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ObjectClass(StrEnum):
    """Supported object classes for the initial MVP."""

    PERSON = "person"
    CAR = "car"


class EventType(StrEnum):
    """Event types produced by the rule engine."""

    ENTER_ZONE = "enter_zone"
    LEAVE_ZONE = "leave_zone"
    EMPTY_ZONE = "empty_zone"
    COUNT_IN_ZONE = "count_in_zone"
    COUNT_IN_FRAME = "count_in_frame"
    TRACK_OBJECT = "track_object"


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class Detection:
    """Single object detection on one video frame."""

    frame_index: int
    timestamp_sec: float
    object_class: ObjectClass
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class Track:
    """Tracked object state at one frame."""

    track_id: int
    frame_index: int
    timestamp_sec: float
    object_class: ObjectClass
    bbox: BoundingBox


@dataclass(frozen=True)
class Zone:
    """Area of interest represented as a polygon."""

    name: str
    points: tuple[tuple[float, float], ...]

    def contains_point(self, point: tuple[float, float]) -> bool:
        """Return whether a point is inside the polygon."""
        x, y = point
        inside = False
        point_count = len(self.points)

        if point_count < 3:
            return False

        previous_x, previous_y = self.points[-1]
        for current_x, current_y in self.points:
            crosses_y = (current_y > y) != (previous_y > y)
            if crosses_y:
                slope_x = (previous_x - current_x) * (y - current_y)
                slope_y = previous_y - current_y
                intersect_x = slope_x / slope_y + current_x
                if x < intersect_x:
                    inside = not inside
            previous_x, previous_y = current_x, current_y

        return inside

    def contains_bbox_center(self, bbox: BoundingBox) -> bool:
        return self.contains_point(bbox.center)


@dataclass(frozen=True)
class Event:
    """Structured event emitted by the rule engine."""

    event_type: EventType
    frame_index: int
    timestamp_sec: float
    zone_name: str | None = None
    track_id: int | None = None
    object_class: ObjectClass | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
