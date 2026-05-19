from multimodalcv.core.models import BoundingBox, Event, EventType, ObjectClass, Zone


def test_bounding_box_properties() -> None:
    bbox = BoundingBox(x1=10, y1=20, x2=30, y2=50)

    assert bbox.width == 20
    assert bbox.height == 30
    assert bbox.area == 600
    assert bbox.center == (20, 35)


def test_invalid_bounding_box_has_zero_area() -> None:
    bbox = BoundingBox(x1=30, y1=50, x2=10, y2=20)

    assert bbox.width == 0
    assert bbox.height == 0
    assert bbox.area == 0


def test_zone_contains_point_inside_polygon() -> None:
    zone = Zone(name="room", points=((0, 0), (100, 0), (100, 100), (0, 100)))

    assert zone.contains_point((50, 50))


def test_zone_rejects_point_outside_polygon() -> None:
    zone = Zone(name="room", points=((0, 0), (100, 0), (100, 100), (0, 100)))

    assert not zone.contains_point((150, 50))


def test_zone_contains_bbox_center() -> None:
    zone = Zone(name="room", points=((0, 0), (100, 0), (100, 100), (0, 100)))
    bbox = BoundingBox(x1=40, y1=40, x2=60, y2=60)

    assert zone.contains_bbox_center(bbox)


def test_event_can_store_metadata() -> None:
    event = Event(
        event_type=EventType.ENTER_ZONE,
        frame_index=10,
        timestamp_sec=1.5,
        zone_name="room",
        track_id=7,
        object_class=ObjectClass.PERSON,
        metadata={"confidence": 0.95},
    )

    assert event.metadata["confidence"] == 0.95

