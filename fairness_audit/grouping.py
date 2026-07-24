"""Mapping raw Fitzpatrick scale values (1-6, or -1 for unknown) to
coarser skin-tone groups for fairness comparisons.
"""

from __future__ import annotations

UNKNOWN_FITZPATRICK = -1
UNKNOWN_GROUP = "unknown"

SCHEMES = ("light_mid_dark", "binary")


def assign_skin_tone_group(fitzpatrick_scale: int, scheme: str = "light_mid_dark") -> str:
    """Map a single Fitzpatrick scale value to a skin-tone group label.

    Schemes:
      light_mid_dark: I-II -> "light", III-IV -> "mid", V-VI -> "dark"
      binary:         I-III -> "light" (<=3), IV-VI -> "dark" (>3)
                       (matches the split used in the original project's README)

    -1 (or any value outside 1-6) always maps to "unknown" regardless of scheme.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown grouping scheme '{scheme}'. Expected one of {SCHEMES}.")

    if fitzpatrick_scale is None or fitzpatrick_scale < 1 or fitzpatrick_scale > 6:
        return UNKNOWN_GROUP

    if scheme == "light_mid_dark":
        if fitzpatrick_scale <= 2:
            return "light"
        if fitzpatrick_scale <= 4:
            return "mid"
        return "dark"

    # binary
    return "light" if fitzpatrick_scale <= 3 else "dark"
