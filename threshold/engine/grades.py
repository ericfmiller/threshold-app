"""Grade-to-numeric conversion utilities for SA factor grades."""

from __future__ import annotations

from typing import Any

from threshold.config.defaults import GRADE_TO_NUM


def sa_grade_to_norm(grade: Any) -> float:
    """Convert an SA letter grade to a normalized 0.0-1.0 score.

    A+ = 1.0, F = 0.0.  None or unrecognized grades return 0.5 (neutral).
    """
    if grade is None:
        return 0.5
    grade_str = str(grade).strip().upper()
    if not grade_str or grade_str in ("N/A", "NONE", "-", ""):
        return 0.5
    num = GRADE_TO_NUM.get(grade_str)
    if num is None:
        return 0.5
    # A+ = 13 maps to 1.0, F = 1 maps to 0.0
    return (num - 1) / 12.0


def sa_grade_to_numeric(grade: Any) -> int:
    """Convert an SA letter grade to its numeric value (A+=13, F=1).

    Returns 0 for None or unrecognized grades.
    """
    if grade is None:
        return 0
    grade_str = str(grade).strip().upper()
    return GRADE_TO_NUM.get(grade_str, 0)
