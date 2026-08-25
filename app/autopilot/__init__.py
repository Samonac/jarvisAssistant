"""Autonomous nightly self-improvement mode (Phase 5).

Reuses the interactive coding agent's building blocks (Phases 2-4:
CodingAgentLoop, TaskSnapshot, apply_verification_gate) under a stricter,
unattended policy: a dynamic time window, an inactivity check, a persistent
priority task queue, self-discovery when the queue is empty, and mandatory
rollback for anything that isn't self-verified.
"""
