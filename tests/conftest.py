"""Shared pytest configuration.

Relaxes Hypothesis's default 200ms per-example deadline: several property
tests exercise deliberately slow operations (password hashing, SQLite I/O)
whose wall-clock time depends on host hardware, not on correctness.
"""

from hypothesis import settings

settings.register_profile("jarvis", deadline=None)
settings.load_profile("jarvis")
