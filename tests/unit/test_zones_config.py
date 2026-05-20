import pytest

from multimodalcv.config.zones import ZoneConfigError, default_zone, parse_zone_rect


def test_parse_zone_rect() -> None:
    zone = parse_zone_rect("10,20,110,220", name="entrance")

    assert zone.name == "entrance"
    assert zone.points == (
        (10.0, 20.0),
        (110.0, 20.0),
        (110.0, 220.0),
        (10.0, 220.0),
    )


def test_parse_zone_rect_accepts_spaces() -> None:
    zone = parse_zone_rect(" 10, 20, 110, 220 ")

    assert zone.points[0] == (10.0, 20.0)


def test_parse_zone_rect_rejects_wrong_part_count() -> None:
    with pytest.raises(ZoneConfigError, match="format"):
        parse_zone_rect("1,2,3")


def test_parse_zone_rect_rejects_non_numbers() -> None:
    with pytest.raises(ZoneConfigError, match="numbers"):
        parse_zone_rect("1,2,nope,4")


def test_parse_zone_rect_rejects_invalid_shape() -> None:
    with pytest.raises(ZoneConfigError, match="x2 > x1"):
        parse_zone_rect("10,20,5,30")


def test_default_zone() -> None:
    zone = default_zone()

    assert zone.name == "main"
    assert zone.points == (
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 100.0),
        (0.0, 100.0),
    )

