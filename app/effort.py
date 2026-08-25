"""Thinking-effort tiers for the coding agent (Phase 1).

Maps a discrete, user-facing effort level ("quick" / "standard" / "deep") to
concrete parameters the agent loop and LLM client use: how many plan/act/
observe iterations are allowed, and the generation parameters for each call.
"""

from dataclasses import dataclass

DEFAULT_EFFORT = "standard"


@dataclass(frozen=True)
class EffortConfig:
    """Concrete parameters derived from a thinking-effort tier."""

    name: str
    max_iterations: int
    max_tokens: int
    temperature: float
    replan_after_each_step: bool


EFFORT_TIERS: dict[str, EffortConfig] = {
    "quick": EffortConfig(
        name="quick",
        max_iterations=3,
        max_tokens=512,
        temperature=0.4,
        replan_after_each_step=False,
    ),
    "standard": EffortConfig(
        name="standard",
        max_iterations=8,
        max_tokens=1024,
        temperature=0.7,
        replan_after_each_step=True,
    ),
    "deep": EffortConfig(
        name="deep",
        max_iterations=20,
        max_tokens=2048,
        temperature=0.7,
        replan_after_each_step=True,
    ),
}


def get_effort_config(tier: str = DEFAULT_EFFORT) -> EffortConfig:
    """Look up an effort tier by name.

    Args:
        tier: One of "quick", "standard", "deep" (case-insensitive). Defaults
            to DEFAULT_EFFORT if falsy.

    Returns:
        The EffortConfig for the requested tier.

    Raises:
        ValueError: If the tier name is not recognized.
    """
    key = (tier or DEFAULT_EFFORT).lower().strip()
    if key not in EFFORT_TIERS:
        raise ValueError(
            f"Unknown thinking-effort tier: '{tier}'. "
            f"Valid options: {', '.join(EFFORT_TIERS)}"
        )
    return EFFORT_TIERS[key]
