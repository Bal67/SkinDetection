"""Fairness/bias auditing engine for dermatology AI model predictions.

UI-agnostic: no plotting, no Streamlit imports, no file I/O beyond reading
the predictions CSV. See schema.py, grouping.py, and metrics.py for the
public API.
"""

from fairness_audit.schema import SchemaError, load_predictions
from fairness_audit.grouping import assign_skin_tone_group
from fairness_audit.metrics import compute_disparity, compute_group_metrics, generate_report

__all__ = [
    "SchemaError",
    "load_predictions",
    "assign_skin_tone_group",
    "compute_group_metrics",
    "compute_disparity",
    "generate_report",
]
