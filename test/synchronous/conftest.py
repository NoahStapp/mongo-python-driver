from __future__ import annotations

from test import setup, teardown

import pytest

_IS_SYNC = True


@pytest.fixture(scope="session", autouse=True)
def test_setup_and_teardown():
    setup()
    yield
    teardown()
