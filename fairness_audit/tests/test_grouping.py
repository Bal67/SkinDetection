import pytest

from fairness_audit.grouping import assign_skin_tone_group


@pytest.mark.parametrize(
    "scale,expected",
    [
        (1, "light"),
        (2, "light"),
        (3, "mid"),
        (4, "mid"),
        (5, "dark"),
        (6, "dark"),
        (-1, "unknown"),
    ],
)
def test_light_mid_dark_scheme(scale, expected):
    assert assign_skin_tone_group(scale, scheme="light_mid_dark") == expected


@pytest.mark.parametrize(
    "scale,expected",
    [
        (1, "light"),
        (2, "light"),
        (3, "light"),
        (4, "dark"),
        (5, "dark"),
        (6, "dark"),
        (-1, "unknown"),
    ],
)
def test_binary_scheme(scale, expected):
    assert assign_skin_tone_group(scale, scheme="binary") == expected


def test_default_scheme_is_light_mid_dark():
    assert assign_skin_tone_group(3) == "mid"


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        assign_skin_tone_group(3, scheme="not_a_real_scheme")


def test_out_of_range_scale_is_unknown():
    assert assign_skin_tone_group(0, scheme="binary") == "unknown"
    assert assign_skin_tone_group(7, scheme="binary") == "unknown"
