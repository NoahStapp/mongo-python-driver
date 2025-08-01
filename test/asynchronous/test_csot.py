# Copyright 2022-present MongoDB, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test the CSOT unified spec tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path[0:0] = [""]

from test.asynchronous import AsyncIntegrationTest, async_client_context, unittest
from test.asynchronous.unified_format import generate_test_classes
from test.asynchronous.utils import flaky

import pymongo
from pymongo import _csot
from pymongo.errors import PyMongoError

_IS_SYNC = False

# Location of JSON test specifications.
if _IS_SYNC:
    TEST_PATH = os.path.join(Path(__file__).resolve().parent, "csot")
else:
    TEST_PATH = os.path.join(Path(__file__).resolve().parent.parent, "csot")

# Generate unified tests.
globals().update(generate_test_classes(TEST_PATH, module=__name__))


class TestCSOT(AsyncIntegrationTest):
    RUN_ON_LOAD_BALANCER = True

    @flaky(reason="PYTHON-3522")
    async def test_timeout_nested(self):
        coll = self.db.coll
        self.assertEqual(_csot.get_timeout(), None)
        self.assertEqual(_csot.get_deadline(), float("inf"))
        self.assertEqual(_csot.get_rtt(), 0.0)
        with pymongo.timeout(10):
            await coll.find_one()
            self.assertEqual(_csot.get_timeout(), 10)
            deadline_10 = _csot.get_deadline()

            # Capped at the original 10 deadline.
            with pymongo.timeout(15):
                await coll.find_one()
                self.assertEqual(_csot.get_timeout(), 15)
                self.assertEqual(_csot.get_deadline(), deadline_10)

            # Should be reset to previous values
            self.assertEqual(_csot.get_timeout(), 10)
            self.assertEqual(_csot.get_deadline(), deadline_10)
            await coll.find_one()

            with pymongo.timeout(5):
                await coll.find_one()
                self.assertEqual(_csot.get_timeout(), 5)
                self.assertLess(_csot.get_deadline(), deadline_10)

            # Should be reset to previous values
            self.assertEqual(_csot.get_timeout(), 10)
            self.assertEqual(_csot.get_deadline(), deadline_10)
            await coll.find_one()

        # Should be reset to previous values
        self.assertEqual(_csot.get_timeout(), None)
        self.assertEqual(_csot.get_deadline(), float("inf"))
        self.assertEqual(_csot.get_rtt(), 0.0)

    @async_client_context.require_change_streams
    @flaky(reason="PYTHON-3522")
    async def test_change_stream_can_resume_after_timeouts(self):
        coll = self.db.test
        await coll.insert_one({})
        async with await coll.watch() as stream:
            with pymongo.timeout(0.1):
                with self.assertRaises(PyMongoError) as ctx:
                    await stream.next()
                self.assertTrue(ctx.exception.timeout)
                self.assertTrue(stream.alive)
                with self.assertRaises(PyMongoError) as ctx:
                    await stream.try_next()
                self.assertTrue(ctx.exception.timeout)
                self.assertTrue(stream.alive)
            # Resume before the insert on 3.6 because 4.0 is required to avoid skipping documents
            if async_client_context.version < (4, 0):
                await stream.try_next()
            await coll.insert_one({})
            with pymongo.timeout(10):
                self.assertTrue(await stream.next())
            self.assertTrue(stream.alive)
            # Timeout applies to entire next() call, not only individual commands.
            with pymongo.timeout(0.5):
                with self.assertRaises(PyMongoError) as ctx:
                    await stream.next()
                self.assertTrue(ctx.exception.timeout)
            self.assertTrue(stream.alive)
        self.assertFalse(stream.alive)


if __name__ == "__main__":
    unittest.main()
