"""Output generation: dashboards, narratives, alerts, exports.

Re-exports key public functions for convenience.
"""

from threshold.output.alerts import (
    build_scoring_email,
    generate_scoring_alerts,
    load_grade_history,
    load_previous_scores,
    save_score_history,
)
from threshold.output.dashboard import generate_dashboard
from threshold.output.narrative import generate_narrative

__all__ = [
    "generate_scoring_alerts",
    "build_scoring_email",
    "save_score_history",
    "load_previous_scores",
    "load_grade_history",
    "generate_dashboard",
    "generate_narrative",
]
