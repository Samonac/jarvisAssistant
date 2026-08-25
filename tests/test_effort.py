"""Unit tests for thinking-effort tiers (app.effort)."""

import pytest

from app.effort import DEFAULT_EFFORT, EFFORT_TIERS, get_effort_config


class TestGetEffortConfig:
    """Effort tier lookup and validation."""

    @pytest.mark.parametrize("tier", ["quick", "standard", "deep"])
    def test_known_tiers_resolve(self, tier):
        cfg = get_effort_config(tier)
        assert cfg.name == tier

    @pytest.mark.parametrize("tier", ["QUICK", " Standard ", "Deep"])
    def test_lookup_is_case_and_whitespace_insensitive(self, tier):
        cfg = get_effort_config(tier)
        assert cfg.name == tier.strip().lower()

    def test_default_tier_used_when_falsy(self):
        assert get_effort_config("").name == DEFAULT_EFFORT
        assert get_effort_config(None).name == DEFAULT_EFFORT

    def test_unknown_tier_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown thinking-effort tier"):
            get_effort_config("ultra-mega-deep")

    def test_iterations_and_tokens_increase_with_effort(self):
        """Higher effort tiers must allow at least as many iterations/tokens as lower ones."""
        quick, standard, deep = (
            EFFORT_TIERS["quick"],
            EFFORT_TIERS["standard"],
            EFFORT_TIERS["deep"],
        )
        assert quick.max_iterations < standard.max_iterations < deep.max_iterations
        assert quick.max_tokens < standard.max_tokens < deep.max_tokens
