"""Alias module for `sim_core.scorer` matching Section 4 specification naming."""
from sim_core.scorer import (
    ChartEvaluationResult,
    compute_rl_reward,
    evaluate_chart,
    evaluate_charts_batch,
    export_trajectory,
    main,
)

__all__ = [
    "ChartEvaluationResult",
    "compute_rl_reward",
    "evaluate_chart",
    "evaluate_charts_batch",
    "export_trajectory",
    "main",
]
