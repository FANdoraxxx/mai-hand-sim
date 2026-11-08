"""Human Maimai DX Play Simulator & Transformer Chart Scorer (`sim_core`)."""
from sim_core.config import (
    DEFAULT_HIT_MODE,
    DEFAULT_SKILL_PROFILE,
    SKILL_PROFILES,
    HitMode,
    SkillProfileConfig,
    UnreasonableType,
)
from sim_core.evaluator import (
    ChartEvaluationResult,
    compute_rl_reward,
    evaluate_chart,
    evaluate_charts_batch,
    export_trajectory,
)
from sim_core.geometry import (
    CANONICAL_33_SENSORS,
    OUTER_BUTTONS,
    TOUCH_ZONES,
    build_slide_route,
    circle_polygon_intersection_area,
    get_button_target_xy,
    get_sensor_zone,
    solve_two_palm_coverage,
)
from sim_core.judge import (
    JudgeGrade,
    NoteJudgementResult,
    SlideSensorQueueTracker,
    compute_dx_achievement_rate,
)
from sim_core.kinematics import (
    ArmIKSolution,
    BodyKinematicsState,
    TorsoState,
    solve_arm_ik,
    solve_coupled_body_ik,
)
from sim_core.maidata_parser import (
    ChartEvent,
    NoteType,
    ParsedChart,
    ParsedNote,
    parse_simai_chart,
)
from sim_core.planner import (
    HandAssignment,
    PlannedEventStep,
    PlannerState,
    PlanResult,
    TrajectoryFrame,
    UnreasonableEventRecord,
    plan_chart_movements,
)
from sim_core.player_profile import get_skill_profile
from sim_core.surrogate import predict_chart_surrogate
from sim_core.body_model import BodyState, BodyTimeline, ERGONOMIC_JOINT_SPECS, body_timeline
from sim_core.vocab_and_rarity import window_body_score
from sim_core.version import (
    SIM_CORE_VERSION,
    check_artifact_version,
    get_sim_core_code_hash,
    stamp_artifact_version,
)

__all__ = [
    "SIM_CORE_VERSION",
    "get_sim_core_code_hash",
    "check_artifact_version",
    "stamp_artifact_version",
    "HitMode",
    "SkillProfileConfig",
    "SKILL_PROFILES",
    "DEFAULT_HIT_MODE",
    "DEFAULT_SKILL_PROFILE",
    "UnreasonableType",
    "get_skill_profile",
    "CANONICAL_33_SENSORS",
    "TOUCH_ZONES",
    "OUTER_BUTTONS",
    "get_sensor_zone",
    "get_button_target_xy",
    "circle_polygon_intersection_area",
    "solve_two_palm_coverage",
    "build_slide_route",
    "NoteType",
    "ParsedNote",
    "ChartEvent",
    "ParsedChart",
    "parse_simai_chart",
    "TorsoState",
    "ArmIKSolution",
    "BodyKinematicsState",
    "solve_arm_ik",
    "solve_coupled_body_ik",
    "JudgeGrade",
    "NoteJudgementResult",
    "SlideSensorQueueTracker",
    "compute_dx_achievement_rate",
    "HandAssignment",
    "PlannedEventStep",
    "PlannerState",
    "TrajectoryFrame",
    "UnreasonableEventRecord",
    "PlanResult",
    "plan_chart_movements",
    "ChartEvaluationResult",
    "compute_rl_reward",
    "evaluate_chart",
    "evaluate_charts_batch",
    "export_trajectory",
    "predict_chart_surrogate",
    "BodyState",
    "BodyTimeline",
    "ERGONOMIC_JOINT_SPECS",
    "body_timeline",
    "window_body_score",
    "KinematicVisualizer",
]


def __getattr__(name: str):
    if name == "KinematicVisualizer":
        from sim_core.visualizer import KinematicVisualizer
        return KinematicVisualizer
    raise AttributeError(f"module 'sim_core' has no attribute {name!r}")
