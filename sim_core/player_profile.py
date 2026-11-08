"""Player skill profiles and hit-mode preference wrappers."""
from __future__ import annotations

from sim_core.config import (
    DEFAULT_HIT_MODE,
    DEFAULT_SKILL_PROFILE,
    SKILL_PROFILES,
    HitMode,
    SkillProfileConfig,
)


def get_skill_profile(profile: str | SkillProfileConfig = DEFAULT_SKILL_PROFILE) -> SkillProfileConfig:
    """Resolve a skill profile by name or alias."""
    if isinstance(profile, SkillProfileConfig):
        return profile
    if profile in SKILL_PROFILES:
        return SKILL_PROFILES[profile]
    norm = profile.strip().lower()
    aliases = {
        "lv12": "Lv12_Advanced",
        "advanced": "Lv12_Advanced",
        "lv12_advanced": "Lv12_Advanced",
        "lv13": "Lv13+_Expert",
        "lv13+": "Lv13+_Expert",
        "expert": "Lv13+_Expert",
        "lv13+_expert": "Lv13+_Expert",
        "lv14": "Lv14+_Master",
        "lv14+": "Lv14+_Master",
        "master": "Lv14+_Master",
        "lv14+_master": "Lv14+_Master",
        "lv15": "Lv15_Apex",
        "apex": "Lv15_Apex",
        "lv15_apex": "Lv15_Apex",
    }
    if norm in aliases:
        return SKILL_PROFILES[aliases[norm]]
    raise KeyError(f"Unknown skill profile {profile!r}. Available: {list(SKILL_PROFILES.keys())}")


__all__ = [
    "HitMode",
    "SkillProfileConfig",
    "SKILL_PROFILES",
    "DEFAULT_SKILL_PROFILE",
    "DEFAULT_HIT_MODE",
    "get_skill_profile",
]
