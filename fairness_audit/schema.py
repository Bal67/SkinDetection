"""Loading and validation of predictions CSVs for the fairness audit engine.

The expected schema:
    image_id           str,   required, unique
    true_label         str,   required
    predicted_label    str,   required
    confidence         float, optional, 0-1
    fitzpatrick_scale  int,   required; -1 or blank means "unknown"
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["image_id", "true_label", "predicted_label", "fitzpatrick_scale"]
OPTIONAL_COLUMNS = ["confidence"]
UNKNOWN_FITZPATRICK = -1


class SchemaError(Exception):
    """Raised when a predictions CSV fails validation."""


def load_predictions(path_or_buffer) -> pd.DataFrame:
    """Load and validate a predictions CSV.

    Accepts a file path (str/Path) or a file-like object/buffer (e.g. an
    uploaded-file object from Streamlit, or io.StringIO in tests).

    Returns a validated DataFrame with `fitzpatrick_scale` coerced to int
    (blank/NaN treated as -1, meaning "unknown"). Raises SchemaError with a
    human-readable, itemized message if validation fails.
    """
    errors: list[str] = []

    try:
        df = pd.read_csv(path_or_buffer)
    except pd.errors.EmptyDataError:
        raise SchemaError("The CSV file is empty (no columns/header found).")
    except Exception as exc:
        raise SchemaError(f"Failed to parse CSV: {exc}")

    if df.shape[0] == 0:
        raise SchemaError("The CSV file has no data rows.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}.")

    # Can't validate column contents if the column itself is missing.
    if errors:
        raise SchemaError(_format_errors(errors))

    for col in ("image_id", "true_label", "predicted_label"):
        if df[col].isna().any():
            errors.append(f"Column '{col}' contains blank/missing values.")

    if df["image_id"].duplicated().any():
        dupes = df.loc[df["image_id"].duplicated(), "image_id"].unique()
        preview = ", ".join(str(d) for d in dupes[:5])
        suffix = "..." if len(dupes) > 5 else ""
        errors.append(f"Column 'image_id' must be unique; duplicates found: {preview}{suffix}.")

    # fitzpatrick_scale: blank -> -1 (unknown); anything non-numeric is an error.
    raw_scale = df["fitzpatrick_scale"]
    coerced = pd.to_numeric(raw_scale, errors="coerce")
    non_numeric_mask = coerced.isna() & raw_scale.notna()
    if non_numeric_mask.any():
        bad = raw_scale[non_numeric_mask].unique()
        preview = ", ".join(str(b) for b in bad[:5])
        suffix = "..." if len(bad) > 5 else ""
        errors.append(f"Column 'fitzpatrick_scale' contains non-numeric value(s): {preview}{suffix}.")
    else:
        coerced = coerced.fillna(UNKNOWN_FITZPATRICK)
        non_integer_mask = (coerced % 1 != 0)
        if non_integer_mask.any():
            errors.append("Column 'fitzpatrick_scale' contains non-integer numeric value(s).")
        else:
            df["fitzpatrick_scale"] = coerced.astype(int)
            out_of_range_mask = ~df["fitzpatrick_scale"].isin([-1, 1, 2, 3, 4, 5, 6])
            if out_of_range_mask.any():
                bad = df.loc[out_of_range_mask, "fitzpatrick_scale"].unique()
                preview = ", ".join(str(b) for b in bad[:5])
                errors.append(
                    f"Column 'fitzpatrick_scale' contains out-of-range value(s) "
                    f"(expected 1-6 or -1/blank for unknown): {preview}."
                )

    if "confidence" in df.columns:
        conf = pd.to_numeric(df["confidence"], errors="coerce")
        bad_conf_mask = conf.notna() & ((conf < 0) | (conf > 1))
        non_numeric_conf_mask = conf.isna() & df["confidence"].notna()
        if non_numeric_conf_mask.any():
            errors.append("Column 'confidence' contains non-numeric value(s).")
        if bad_conf_mask.any():
            errors.append("Column 'confidence' contains value(s) outside the 0-1 range.")

    if errors:
        raise SchemaError(_format_errors(errors))

    return df


def _format_errors(errors: list[str]) -> str:
    bullet_list = "\n".join(f"  - {e}" for e in errors)
    return f"Predictions CSV failed validation:\n{bullet_list}"
