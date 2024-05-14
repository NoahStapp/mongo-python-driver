from __future__ import annotations

from test import async_setup, async_teardown

import pytest_asyncio


@pytest_asyncio.fixture(scope="session", autouse=True)
async def test_setup_and_teardown():
    await async_setup()
    yield
    await async_teardown()
