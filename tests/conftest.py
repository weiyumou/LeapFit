"""Fixtures shared by the test modules; plain builders live in ``helpers``."""

from __future__ import annotations

import pytest

from leapfit import load_student_step

from helpers import EXAMPLE


@pytest.fixture(scope="session")
def example():
    """The example export under its ``Topics`` KC model."""
    return load_student_step(EXAMPLE, kc_model="Topics")
