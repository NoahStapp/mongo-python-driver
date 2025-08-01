# Copyright 2017 MongoDB, Inc.
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

"""Test the client_session module."""
from __future__ import annotations

import asyncio
import copy
import sys
import time
from inspect import iscoroutinefunction
from io import BytesIO
from test.helpers import ExceptionCatchingTask
from typing import Any, Callable, List, Set, Tuple

from pymongo.synchronous.mongo_client import MongoClient

sys.path[0:0] = [""]

from test import (
    IntegrationTest,
    SkipTest,
    UnitTest,
    client_context,
    unittest,
)
from test.helpers import client_knobs
from test.utils_shared import (
    EventListener,
    HeartbeatEventListener,
    OvertCommandListener,
    wait_until,
)

from bson import DBRef
from gridfs.synchronous.grid_file import GridFS, GridFSBucket
from pymongo import ASCENDING, MongoClient, _csot, monitoring
from pymongo.common import _MAX_END_SESSIONS
from pymongo.errors import ConfigurationError, InvalidOperation, OperationFailure
from pymongo.operations import IndexModel, InsertOne, UpdateOne
from pymongo.read_concern import ReadConcern
from pymongo.synchronous.command_cursor import CommandCursor
from pymongo.synchronous.cursor import Cursor
from pymongo.synchronous.helpers import next

_IS_SYNC = True


# Ignore auth commands like saslStart, so we can assert lsid is in all commands.
class SessionTestListener(EventListener):
    def started(self, event):
        if not event.command_name.startswith("sasl"):
            super().started(event)

    def succeeded(self, event):
        if not event.command_name.startswith("sasl"):
            super().succeeded(event)

    def failed(self, event):
        if not event.command_name.startswith("sasl"):
            super().failed(event)

    def first_command_started(self):
        assert len(self.started_events) >= 1, "No command-started events"

        return self.started_events[0]


def session_ids(client):
    return [s.session_id for s in copy.copy(client._topology._session_pool)]


class TestSession(IntegrationTest):
    client2: MongoClient
    sensitive_commands: Set[str]

    @client_context.require_sessions
    def setUp(self):
        super().setUp()
        # Create a second client so we can make sure clients cannot share
        # sessions.
        self.client2 = self.rs_or_single_client()

        # Redact no commands, so we can test user-admin commands have "lsid".
        self.sensitive_commands = monitoring._SENSITIVE_COMMANDS.copy()
        monitoring._SENSITIVE_COMMANDS.clear()

        self.listener = SessionTestListener()
        self.session_checker_listener = SessionTestListener()
        self.client = self.rs_or_single_client(
            event_listeners=[self.listener, self.session_checker_listener]
        )
        self.db = self.client.pymongo_test
        self.initial_lsids = {s["id"] for s in session_ids(self.client)}

    def tearDown(self):
        monitoring._SENSITIVE_COMMANDS.update(self.sensitive_commands)
        self.client.drop_database("pymongo_test")
        used_lsids = self.initial_lsids.copy()
        for event in self.session_checker_listener.started_events:
            if "lsid" in event.command:
                used_lsids.add(event.command["lsid"]["id"])

        current_lsids = {s["id"] for s in session_ids(self.client)}
        self.assertLessEqual(used_lsids, current_lsids)

        super().tearDown()

    def _test_ops(self, client, *ops):
        listener = client.options.event_listeners[0]

        for f, args, kw in ops:
            with client.start_session() as s:
                listener.reset()
                s._materialize()
                last_use = s._server_session.last_use
                start = time.monotonic()
                self.assertLessEqual(last_use, start)
                # In case "f" modifies its inputs.
                args = copy.copy(args)
                kw = copy.copy(kw)
                kw["session"] = s
                f(*args, **kw)
                self.assertGreaterEqual(len(listener.started_events), 1)
                for event in listener.started_events:
                    self.assertIn(
                        "lsid",
                        event.command,
                        f"{f.__name__} sent no lsid with {event.command_name}",
                    )

                    self.assertEqual(
                        s.session_id,
                        event.command["lsid"],
                        f"{f.__name__} sent wrong lsid with {event.command_name}",
                    )

                self.assertFalse(s.has_ended)

            self.assertTrue(s.has_ended)
            with self.assertRaisesRegex(InvalidOperation, "ended session"):
                f(*args, **kw)

            # Test a session cannot be used on another client.
            with self.client2.start_session() as s:
                # In case "f" modifies its inputs.
                args = copy.copy(args)
                kw = copy.copy(kw)
                kw["session"] = s
                with self.assertRaisesRegex(
                    InvalidOperation,
                    "Can only use session with the MongoClient that started it",
                ):
                    f(*args, **kw)

        # No explicit session.
        for f, args, kw in ops:
            listener.reset()
            f(*args, **kw)
            self.assertGreaterEqual(len(listener.started_events), 1)
            lsids = []
            for event in listener.started_events:
                self.assertIn(
                    "lsid",
                    event.command,
                    f"{f.__name__} sent no lsid with {event.command_name}",
                )

                lsids.append(event.command["lsid"])

            if not (sys.platform.startswith("java") or "PyPy" in sys.version):
                # Server session was returned to pool. Ignore interpreters with
                # non-deterministic GC.
                for lsid in lsids:
                    self.assertIn(
                        lsid,
                        session_ids(client),
                        f"{f.__name__} did not return implicit session to pool",
                    )

    def test_implicit_sessions_checkout(self):
        # "To confirm that implicit sessions only allocate their server session after a
        # successful connection checkout" test from Driver Sessions Spec.
        succeeded = False
        lsid_set = set()
        listener = OvertCommandListener()
        client = self.rs_or_single_client(event_listeners=[listener], maxPoolSize=1)
        # Retry up to 10 times because there is a known race condition that can cause multiple
        # sessions to be used: connection check in happens before session check in
        for _ in range(10):
            cursor = client.db.test.find({})
            ops: List[Tuple[Callable, List[Any]]] = [
                (client.db.test.find_one, [{"_id": 1}]),
                (client.db.test.delete_one, [{}]),
                (client.db.test.update_one, [{}, {"$set": {"x": 2}}]),
                (client.db.test.bulk_write, [[UpdateOne({}, {"$set": {"x": 2}})]]),
                (client.db.test.find_one_and_delete, [{}]),
                (client.db.test.find_one_and_update, [{}, {"$set": {"x": 1}}]),
                (client.db.test.find_one_and_replace, [{}, {}]),
                (client.db.test.aggregate, [[{"$limit": 1}]]),
                (client.db.test.find, []),
                (client.server_info, []),
                (client.db.aggregate, [[{"$listLocalSessions": {}}, {"$limit": 1}]]),
                (cursor.distinct, ["_id"]),
                (client.db.list_collections, []),
            ]
            tasks = []
            listener.reset()

            def target(op, *args):
                if iscoroutinefunction(op):
                    res = op(*args)
                else:
                    res = op(*args)
                if isinstance(res, (Cursor, CommandCursor)):
                    res.to_list()

            for op, args in ops:
                tasks.append(
                    ExceptionCatchingTask(target=target, args=[op, *args], name=op.__name__)
                )
                tasks[-1].start()
            self.assertEqual(len(tasks), len(ops))
            for t in tasks:
                t.join()
                self.assertIsNone(t.exc)
            lsid_set.clear()
            for i in listener.started_events:
                if i.command.get("lsid"):
                    lsid_set.add(i.command.get("lsid")["id"])
            if len(lsid_set) == 1:
                # Break on first success.
                succeeded = True
                break
        self.assertTrue(succeeded, lsid_set)

    def test_pool_lifo(self):
        # "Pool is LIFO" test from Driver Sessions Spec.
        a = self.client.start_session()
        b = self.client.start_session()
        a_id = a.session_id
        b_id = b.session_id
        a.end_session()
        b.end_session()

        s = self.client.start_session()
        self.assertEqual(b_id, s.session_id)
        self.assertNotEqual(a_id, s.session_id)

        s2 = self.client.start_session()
        self.assertEqual(a_id, s2.session_id)
        self.assertNotEqual(b_id, s2.session_id)

        s.end_session()
        s2.end_session()

    def test_end_session(self):
        # We test elsewhere that using an ended session throws InvalidOperation.
        client = self.client
        s = client.start_session()
        self.assertFalse(s.has_ended)
        self.assertIsNotNone(s.session_id)

        s.end_session()
        self.assertTrue(s.has_ended)

        with self.assertRaisesRegex(InvalidOperation, "ended session"):
            s.session_id

    def test_end_sessions(self):
        # Use a new client so that the tearDown hook does not error.
        listener = SessionTestListener()
        client = self.rs_or_single_client(event_listeners=[listener])
        # Start many sessions.
        sessions = [client.start_session() for _ in range(_MAX_END_SESSIONS + 1)]
        for s in sessions:
            s._materialize()
        for s in sessions:
            s.end_session()

        # Closing the client should end all sessions and clear the pool.
        self.assertEqual(len(client._topology._session_pool), _MAX_END_SESSIONS + 1)
        client.close()
        self.assertEqual(len(client._topology._session_pool), 0)
        end_sessions = [e for e in listener.started_events if e.command_name == "endSessions"]
        self.assertEqual(len(end_sessions), 2)

        # Closing again should not send any commands.
        listener.reset()
        client.close()
        self.assertEqual(len(listener.started_events), 0)

    def test_client(self):
        client = self.client
        ops: list = [
            (client.server_info, [], {}),
            (client.list_database_names, [], {}),
            (client.drop_database, ["pymongo_test"], {}),
        ]

        self._test_ops(client, *ops)

    def test_database(self):
        client = self.client
        db = client.pymongo_test
        ops: list = [
            (db.command, ["ping"], {}),
            (db.create_collection, ["collection"], {}),
            (db.list_collection_names, [], {}),
            (db.validate_collection, ["collection"], {}),
            (db.drop_collection, ["collection"], {}),
            (db.dereference, [DBRef("collection", 1)], {}),
        ]
        self._test_ops(client, *ops)

    @staticmethod
    def collection_write_ops(coll):
        """Generate database write ops for tests."""
        return [
            (coll.drop, [], {}),
            (coll.bulk_write, [[InsertOne({})]], {}),
            (coll.insert_one, [{}], {}),
            (coll.insert_many, [[{}, {}]], {}),
            (coll.replace_one, [{}, {}], {}),
            (coll.update_one, [{}, {"$set": {"a": 1}}], {}),
            (coll.update_many, [{}, {"$set": {"a": 1}}], {}),
            (coll.delete_one, [{}], {}),
            (coll.delete_many, [{}], {}),
            (coll.find_one_and_replace, [{}, {}], {}),
            (coll.find_one_and_update, [{}, {"$set": {"a": 1}}], {}),
            (coll.find_one_and_delete, [{}, {}], {}),
            (coll.rename, ["collection2"], {}),
            # Drop collection2 between tests of "rename", above.
            (coll.database.drop_collection, ["collection2"], {}),
            (coll.create_indexes, [[IndexModel("a")]], {}),
            (coll.create_index, ["a"], {}),
            (coll.drop_index, ["a_1"], {}),
            (coll.drop_indexes, [], {}),
            (coll.aggregate, [[{"$out": "aggout"}]], {}),
        ]

    def test_collection(self):
        client = self.client
        coll = client.pymongo_test.collection

        # Test some collection methods - the rest are in test_cursor.
        ops = self.collection_write_ops(coll)
        ops.extend(
            [
                (coll.distinct, ["a"], {}),
                (coll.find_one, [], {}),
                (coll.count_documents, [{}], {}),
                (coll.list_indexes, [], {}),
                (coll.index_information, [], {}),
                (coll.options, [], {}),
                (coll.aggregate, [[]], {}),
            ]
        )

        self._test_ops(client, *ops)

    def test_cursor_clone(self):
        coll = self.client.pymongo_test.collection
        # Ensure some batches.
        coll.insert_many({} for _ in range(10))
        self.addCleanup(coll.drop)

        with self.client.start_session() as s:
            cursor = coll.find(session=s)
            self.assertTrue(cursor.session is s)
            clone = cursor.clone()
            self.assertTrue(clone.session is s)

        # No explicit session.
        cursor = coll.find(batch_size=2)
        next(cursor)
        # Session is "owned" by cursor.
        self.assertIsNone(cursor.session)
        self.assertIsNotNone(cursor._session)
        clone = cursor.clone()
        next(clone)
        self.assertIsNone(clone.session)
        self.assertIsNotNone(clone._session)
        self.assertFalse(cursor._session is clone._session)
        cursor.close()
        clone.close()

    def test_cursor(self):
        listener = self.listener
        client = self.client
        coll = client.pymongo_test.collection
        coll.insert_many([{} for _ in range(1000)])

        # Test all cursor methods.
        if _IS_SYNC:
            # getitem is only supported in the synchronous API
            ops = [
                ("find", lambda session: coll.find(session=session).to_list()),
                ("getitem", lambda session: coll.find(session=session)[0]),
                ("distinct", lambda session: coll.find(session=session).distinct("a")),
                ("explain", lambda session: coll.find(session=session).explain()),
            ]
        else:
            ops = [
                ("find", lambda session: coll.find(session=session).to_list()),
                ("distinct", lambda session: coll.find(session=session).distinct("a")),
                ("explain", lambda session: coll.find(session=session).explain()),
            ]

        for name, f in ops:
            with client.start_session() as s:
                listener.reset()
                f(session=s)
                self.assertGreaterEqual(len(listener.started_events), 1)
                for event in listener.started_events:
                    self.assertIn(
                        "lsid",
                        event.command,
                        f"{name} sent no lsid with {event.command_name}",
                    )

                    self.assertEqual(
                        s.session_id,
                        event.command["lsid"],
                        f"{name} sent wrong lsid with {event.command_name}",
                    )

            with self.assertRaisesRegex(InvalidOperation, "ended session"):
                f(session=s)

        # No explicit session.
        for name, f in ops:
            listener.reset()
            f(session=None)
            event0 = listener.first_command_started()
            self.assertIn("lsid", event0.command, f"{name} sent no lsid with {event0.command_name}")

            lsid = event0.command["lsid"]

            for event in listener.started_events[1:]:
                self.assertIn(
                    "lsid", event.command, f"{name} sent no lsid with {event.command_name}"
                )

                self.assertEqual(
                    lsid,
                    event.command["lsid"],
                    f"{name} sent wrong lsid with {event.command_name}",
                )

    def test_gridfs(self):
        client = self.client
        fs = GridFS(client.pymongo_test)

        def new_file(session=None):
            grid_file = fs.new_file(_id=1, filename="f", session=session)
            # 1 MB, 5 chunks, to test that each chunk is fetched with same lsid.
            grid_file.write(b"a" * 1048576)
            grid_file.close()

        def find(session=None):
            files = fs.find({"_id": 1}, session=session).to_list()
            for f in files:
                f.read()

        def get(session=None):
            (fs.get(1, session=session)).read()

        def get_version(session=None):
            (fs.get_version("f", session=session)).read()

        def get_last_version(session=None):
            (fs.get_last_version("f", session=session)).read()

        def find_list(session=None):
            fs.find(session=session).to_list()

        self._test_ops(
            client,
            (new_file, [], {}),
            (fs.put, [b"data"], {}),
            (get, [], {}),
            (get_version, [], {}),
            (get_last_version, [], {}),
            (fs.list, [], {}),
            (fs.find_one, [1], {}),
            (find_list, [], {}),
            (fs.exists, [1], {}),
            (find, [], {}),
            (fs.delete, [1], {}),
        )

    def test_gridfs_bucket(self):
        client = self.client
        bucket = GridFSBucket(client.pymongo_test)

        def upload(session=None):
            stream = bucket.open_upload_stream("f", session=session)
            stream.write(b"a" * 1048576)
            stream.close()

        def upload_with_id(session=None):
            stream = bucket.open_upload_stream_with_id(1, "f1", session=session)
            stream.write(b"a" * 1048576)
            stream.close()

        def open_download_stream(session=None):
            stream = bucket.open_download_stream(1, session=session)
            stream.read()

        def open_download_stream_by_name(session=None):
            stream = bucket.open_download_stream_by_name("f", session=session)
            stream.read()

        def find(session=None):
            files = bucket.find({"_id": 1}, session=session).to_list()
            for f in files:
                f.read()

        sio = BytesIO()

        self._test_ops(
            client,
            (upload, [], {}),
            (upload_with_id, [], {}),
            (bucket.upload_from_stream, ["f", b"data"], {}),
            (bucket.upload_from_stream_with_id, [2, "f", b"data"], {}),
            (open_download_stream, [], {}),
            (open_download_stream_by_name, [], {}),
            (bucket.download_to_stream, [1, sio], {}),
            (bucket.download_to_stream_by_name, ["f", sio], {}),
            (find, [], {}),
            (bucket.rename, [1, "f2"], {}),
            (bucket.rename_by_name, ["f2", "f3"], {}),
            # Delete both files so _test_ops can run these operations twice.
            (bucket.delete, [1], {}),
            (bucket.delete_by_name, ["f"], {}),
        )

    def test_gridfsbucket_cursor(self):
        client = self.client
        bucket = GridFSBucket(client.pymongo_test)

        for file_id in 1, 2:
            stream = bucket.open_upload_stream_with_id(file_id, str(file_id))
            stream.write(b"a" * 1048576)
            stream.close()

        with client.start_session() as s:
            cursor = bucket.find(session=s)
            for f in cursor:
                f.read()

            self.assertFalse(s.has_ended)

        self.assertTrue(s.has_ended)

        # No explicit session.
        cursor = bucket.find(batch_size=1)
        files = [cursor.next()]

        s = cursor._session
        self.assertFalse(s.has_ended)
        cursor.__del__()

        self.assertTrue(s.has_ended)
        self.assertIsNone(cursor._session)

        # Files are still valid, they use their own sessions.
        for f in files:
            f.read()

        # Explicit session.
        with client.start_session() as s:
            cursor = bucket.find(session=s)
            assert cursor.session is not None
            s = cursor.session
            files = cursor.to_list()
            cursor.__del__()
            self.assertFalse(s.has_ended)

            for f in files:
                f.read()

        for f in files:
            # Attempt to read the file again.
            f.seek(0)
            with self.assertRaisesRegex(InvalidOperation, "ended session"):
                f.read()

    def test_aggregate(self):
        client = self.client
        coll = client.pymongo_test.collection

        def agg(session=None):
            (coll.aggregate([], batchSize=2, session=session)).to_list()

        # With empty collection.
        self._test_ops(client, (agg, [], {}))

        # Now with documents.
        coll.insert_many([{} for _ in range(10)])
        self.addCleanup(coll.drop)
        self._test_ops(client, (agg, [], {}))

    def test_killcursors(self):
        client = self.client
        coll = client.pymongo_test.collection
        coll.insert_many([{} for _ in range(10)])

        def explicit_close(session=None):
            cursor = coll.find(batch_size=2, session=session)
            next(cursor)
            cursor.close()

        self._test_ops(client, (explicit_close, [], {}))

    def test_aggregate_error(self):
        listener = self.listener
        client = self.client
        coll = client.pymongo_test.collection
        # 3.6.0 mongos only validates the aggregate pipeline when the
        # database exists.
        coll.insert_one({})
        listener.reset()

        with self.assertRaises(OperationFailure):
            coll.aggregate([{"$badOperation": {"bar": 1}}])

        event = listener.first_command_started()
        self.assertEqual(event.command_name, "aggregate")
        lsid = event.command["lsid"]
        # Session was returned to pool despite error.
        self.assertIn(lsid, session_ids(client))

    def _test_cursor_helper(self, create_cursor, close_cursor):
        coll = self.client.pymongo_test.collection
        coll.insert_many([{} for _ in range(1000)])

        cursor = create_cursor(coll, None)
        next(cursor)
        # Session is "owned" by cursor.
        session = cursor._session
        self.assertIsNotNone(session)
        lsid = session.session_id
        next(cursor)

        # Cursor owns its session unto death.
        self.assertNotIn(lsid, session_ids(self.client))
        close_cursor(cursor)
        self.assertIn(lsid, session_ids(self.client))

        # An explicit session is not ended by cursor.close() or list(cursor).
        with self.client.start_session() as s:
            cursor = create_cursor(coll, s)
            next(cursor)
            close_cursor(cursor)
            self.assertFalse(s.has_ended)
            lsid = s.session_id

        self.assertTrue(s.has_ended)
        self.assertIn(lsid, session_ids(self.client))

    def test_cursor_close(self):
        def find(coll, session):
            return coll.find(session=session)

        self._test_cursor_helper(find, lambda cursor: cursor.close())

    def test_command_cursor_close(self):
        def aggregate(coll, session):
            return coll.aggregate([], session=session)

        self._test_cursor_helper(aggregate, lambda cursor: cursor.close())

    def test_cursor_del(self):
        def find(coll, session):
            return coll.find(session=session)

        def delete(cursor):
            return cursor.__del__()

        self._test_cursor_helper(find, delete)

    def test_command_cursor_del(self):
        def aggregate(coll, session):
            return coll.aggregate([], session=session)

        def delete(cursor):
            return cursor.__del__()

        self._test_cursor_helper(aggregate, delete)

    def test_cursor_exhaust(self):
        def find(coll, session):
            return coll.find(session=session)

        self._test_cursor_helper(find, lambda cursor: cursor.to_list())

    def test_command_cursor_exhaust(self):
        def aggregate(coll, session):
            return coll.aggregate([], session=session)

        self._test_cursor_helper(aggregate, lambda cursor: cursor.to_list())

    def test_cursor_limit_reached(self):
        def find(coll, session):
            return coll.find(limit=4, batch_size=2, session=session)

        self._test_cursor_helper(
            find,
            lambda cursor: cursor.to_list(),
        )

    def test_command_cursor_limit_reached(self):
        def aggregate(coll, session):
            return coll.aggregate([], batchSize=900, session=session)

        self._test_cursor_helper(
            aggregate,
            lambda cursor: cursor.to_list(),
        )

    def _test_unacknowledged_ops(self, client, *ops):
        listener = client.options.event_listeners[0]

        for f, args, kw in ops:
            with client.start_session() as s:
                listener.reset()
                # In case "f" modifies its inputs.
                args = copy.copy(args)
                kw = copy.copy(kw)
                kw["session"] = s
                with self.assertRaises(
                    ConfigurationError, msg=f"{f.__name__} did not raise ConfigurationError"
                ):
                    f(*args, **kw)
                if f.__name__ == "create_collection":
                    # create_collection runs listCollections first.
                    event = listener.started_events.pop(0)
                    self.assertEqual("listCollections", event.command_name)
                    self.assertIn(
                        "lsid",
                        event.command,
                        f"{f.__name__} sent no lsid with {event.command_name}",
                    )

                # Should not run any command before raising an error.
                self.assertFalse(listener.started_events, f"{f.__name__} sent command")

            self.assertTrue(s.has_ended)

        # Unacknowledged write without a session does not send an lsid.
        for f, args, kw in ops:
            listener.reset()
            f(*args, **kw)
            self.assertGreaterEqual(len(listener.started_events), 1)

            if f.__name__ == "create_collection":
                # create_collection runs listCollections first.
                event = listener.started_events.pop(0)
                self.assertEqual("listCollections", event.command_name)
                self.assertIn(
                    "lsid",
                    event.command,
                    f"{f.__name__} sent no lsid with {event.command_name}",
                )

            for event in listener.started_events:
                self.assertNotIn(
                    "lsid", event.command, f"{f.__name__} sent lsid with {event.command_name}"
                )

    def test_unacknowledged_writes(self):
        # Ensure the collection exists.
        self.client.pymongo_test.test_unacked_writes.insert_one({})
        client = self.rs_or_single_client(w=0, event_listeners=[self.listener])
        db = client.pymongo_test
        coll = db.test_unacked_writes
        ops: list = [
            (client.drop_database, [db.name], {}),
            (db.create_collection, ["collection"], {}),
            (db.drop_collection, ["collection"], {}),
        ]
        ops.extend(self.collection_write_ops(coll))
        self._test_unacknowledged_ops(client, *ops)

        def drop_db():
            try:
                self.client.drop_database(db.name)
                return True
            except OperationFailure as exc:
                # Try again on BackgroundOperationInProgressForDatabase and
                # BackgroundOperationInProgressForNamespace.
                if exc.code in (12586, 12587):
                    return False
                raise

        wait_until(drop_db, "dropped database after w=0 writes")

    def test_snapshot_incompatible_with_causal_consistency(self):
        with self.client.start_session(causal_consistency=False, snapshot=False):
            pass
        with self.client.start_session(causal_consistency=False, snapshot=True):
            pass
        with self.client.start_session(causal_consistency=True, snapshot=False):
            pass
        with self.assertRaises(ConfigurationError):
            with self.client.start_session(causal_consistency=True, snapshot=True):
                pass

    def test_session_not_copyable(self):
        client = self.client
        with client.start_session() as s:
            self.assertRaises(TypeError, lambda: copy.copy(s))


class TestCausalConsistency(UnitTest):
    listener: SessionTestListener
    client: MongoClient

    @client_context.require_sessions
    def setUp(self):
        super().setUp()
        self.listener = SessionTestListener()
        self.client = self.rs_or_single_client(event_listeners=[self.listener])

    @client_context.require_no_standalone
    def test_core(self):
        with self.client.start_session() as sess:
            self.assertIsNone(sess.cluster_time)
            self.assertIsNone(sess.operation_time)
            self.listener.reset()
            self.client.pymongo_test.test.find_one(session=sess)
            started = self.listener.started_events[0]
            cmd = started.command
            self.assertIsNone(cmd.get("readConcern"))
            op_time = sess.operation_time
            self.assertIsNotNone(op_time)
            succeeded = self.listener.succeeded_events[0]
            reply = succeeded.reply
            self.assertEqual(op_time, reply.get("operationTime"))

            # No explicit session
            self.client.pymongo_test.test.insert_one({})
            self.assertEqual(sess.operation_time, op_time)
            self.listener.reset()
            try:
                self.client.pymongo_test.command("doesntexist", session=sess)
            except:
                pass
            failed = self.listener.failed_events[0]
            failed_op_time = failed.failure.get("operationTime")
            # Some older builds of MongoDB 3.5 / 3.6 return None for
            # operationTime when a command fails. Make sure we don't
            # change operation_time to None.
            if failed_op_time is None:
                self.assertIsNotNone(sess.operation_time)
            else:
                self.assertEqual(sess.operation_time, failed_op_time)

            with self.client.start_session() as sess2:
                self.assertIsNone(sess2.cluster_time)
                self.assertIsNone(sess2.operation_time)
                self.assertRaises(TypeError, sess2.advance_cluster_time, 1)
                self.assertRaises(ValueError, sess2.advance_cluster_time, {})
                self.assertRaises(TypeError, sess2.advance_operation_time, 1)
                # No error
                assert sess.cluster_time is not None
                assert sess.operation_time is not None
                sess2.advance_cluster_time(sess.cluster_time)
                sess2.advance_operation_time(sess.operation_time)
                self.assertEqual(sess.cluster_time, sess2.cluster_time)
                self.assertEqual(sess.operation_time, sess2.operation_time)

    def _test_reads(self, op, exception=None):
        coll = self.client.pymongo_test.test
        with self.client.start_session() as sess:
            coll.find_one({}, session=sess)
            operation_time = sess.operation_time
            self.assertIsNotNone(operation_time)
            self.listener.reset()
            if exception:
                with self.assertRaises(exception):
                    op(coll, sess)
            else:
                op(coll, sess)
            act = (
                self.listener.started_events[0]
                .command.get("readConcern", {})
                .get("afterClusterTime")
            )
            self.assertEqual(operation_time, act)

    @client_context.require_no_standalone
    def test_reads(self):
        # Make sure the collection exists.
        self.client.pymongo_test.test.insert_one({})

        def aggregate(coll, session):
            return (coll.aggregate([], session=session)).to_list()

        def aggregate_raw(coll, session):
            return (coll.aggregate_raw_batches([], session=session)).to_list()

        def find_raw(coll, session):
            return coll.find_raw_batches({}, session=session).to_list()

        self._test_reads(aggregate)
        self._test_reads(lambda coll, session: coll.find({}, session=session).to_list())
        self._test_reads(lambda coll, session: coll.find_one({}, session=session))
        self._test_reads(lambda coll, session: coll.count_documents({}, session=session))
        self._test_reads(lambda coll, session: coll.distinct("foo", session=session))
        self._test_reads(aggregate_raw)
        self._test_reads(find_raw)

        with self.assertRaises(ConfigurationError):
            self._test_reads(lambda coll, session: coll.estimated_document_count(session=session))

    def _test_writes(self, op):
        coll = self.client.pymongo_test.test
        with self.client.start_session() as sess:
            op(coll, sess)
            operation_time = sess.operation_time
            self.assertIsNotNone(operation_time)
            self.listener.reset()
            coll.find_one({}, session=sess)
            act = (
                self.listener.started_events[0]
                .command.get("readConcern", {})
                .get("afterClusterTime")
            )
            self.assertEqual(operation_time, act)

    @client_context.require_no_standalone
    def test_writes(self):
        self._test_writes(
            lambda coll, session: coll.bulk_write([InsertOne[dict]({})], session=session)
        )
        self._test_writes(lambda coll, session: coll.insert_one({}, session=session))
        self._test_writes(lambda coll, session: coll.insert_many([{}], session=session))
        self._test_writes(
            lambda coll, session: coll.replace_one({"_id": 1}, {"x": 1}, session=session)
        )
        self._test_writes(
            lambda coll, session: coll.update_one({}, {"$set": {"X": 1}}, session=session)
        )
        self._test_writes(
            lambda coll, session: coll.update_many({}, {"$set": {"x": 1}}, session=session)
        )
        self._test_writes(lambda coll, session: coll.delete_one({}, session=session))
        self._test_writes(lambda coll, session: coll.delete_many({}, session=session))
        self._test_writes(
            lambda coll, session: coll.find_one_and_replace({"x": 1}, {"y": 1}, session=session)
        )
        self._test_writes(
            lambda coll, session: coll.find_one_and_update(
                {"y": 1}, {"$set": {"x": 1}}, session=session
            )
        )
        self._test_writes(lambda coll, session: coll.find_one_and_delete({"x": 1}, session=session))
        self._test_writes(lambda coll, session: coll.create_index("foo", session=session))
        self._test_writes(
            lambda coll, session: coll.create_indexes(
                [IndexModel([("bar", ASCENDING)])], session=session
            )
        )
        self._test_writes(lambda coll, session: coll.drop_index("foo_1", session=session))
        self._test_writes(lambda coll, session: coll.drop_indexes(session=session))

    def _test_no_read_concern(self, op):
        coll = self.client.pymongo_test.test
        with self.client.start_session() as sess:
            coll.find_one({}, session=sess)
            operation_time = sess.operation_time
            self.assertIsNotNone(operation_time)
            self.listener.reset()
            op(coll, sess)
            rc = self.listener.started_events[0].command.get("readConcern")
            self.assertIsNone(rc)

    @client_context.require_no_standalone
    def test_writes_do_not_include_read_concern(self):
        self._test_no_read_concern(
            lambda coll, session: coll.bulk_write([InsertOne[dict]({})], session=session)
        )
        self._test_no_read_concern(lambda coll, session: coll.insert_one({}, session=session))
        self._test_no_read_concern(lambda coll, session: coll.insert_many([{}], session=session))
        self._test_no_read_concern(
            lambda coll, session: coll.replace_one({"_id": 1}, {"x": 1}, session=session)
        )
        self._test_no_read_concern(
            lambda coll, session: coll.update_one({}, {"$set": {"X": 1}}, session=session)
        )
        self._test_no_read_concern(
            lambda coll, session: coll.update_many({}, {"$set": {"x": 1}}, session=session)
        )
        self._test_no_read_concern(lambda coll, session: coll.delete_one({}, session=session))
        self._test_no_read_concern(lambda coll, session: coll.delete_many({}, session=session))
        self._test_no_read_concern(
            lambda coll, session: coll.find_one_and_replace({"x": 1}, {"y": 1}, session=session)
        )
        self._test_no_read_concern(
            lambda coll, session: coll.find_one_and_update(
                {"y": 1}, {"$set": {"x": 1}}, session=session
            )
        )
        self._test_no_read_concern(
            lambda coll, session: coll.find_one_and_delete({"x": 1}, session=session)
        )
        self._test_no_read_concern(lambda coll, session: coll.create_index("foo", session=session))
        self._test_no_read_concern(
            lambda coll, session: coll.create_indexes(
                [IndexModel([("bar", ASCENDING)])], session=session
            )
        )
        self._test_no_read_concern(lambda coll, session: coll.drop_index("foo_1", session=session))
        self._test_no_read_concern(lambda coll, session: coll.drop_indexes(session=session))

        # Not a write, but explain also doesn't support readConcern.
        self._test_no_read_concern(lambda coll, session: coll.find({}, session=session).explain())

    @client_context.require_no_standalone
    def test_get_more_does_not_include_read_concern(self):
        coll = self.client.pymongo_test.test
        with self.client.start_session() as sess:
            coll.find_one({}, session=sess)
            operation_time = sess.operation_time
            self.assertIsNotNone(operation_time)
            coll.insert_many([{}, {}])
            cursor = coll.find({}).batch_size(1)
            next(cursor)
            self.listener.reset()
            cursor.to_list()
            started = self.listener.started_events[0]
            self.assertEqual(started.command_name, "getMore")
            self.assertIsNone(started.command.get("readConcern"))

    def test_session_not_causal(self):
        with self.client.start_session(causal_consistency=False) as s:
            self.client.pymongo_test.test.insert_one({}, session=s)
            self.listener.reset()
            self.client.pymongo_test.test.find_one({}, session=s)
            act = (
                self.listener.started_events[0]
                .command.get("readConcern", {})
                .get("afterClusterTime")
            )
            self.assertIsNone(act)

    @client_context.require_standalone
    def test_server_not_causal(self):
        with self.client.start_session(causal_consistency=True) as s:
            self.client.pymongo_test.test.insert_one({}, session=s)
            self.listener.reset()
            self.client.pymongo_test.test.find_one({}, session=s)
            act = (
                self.listener.started_events[0]
                .command.get("readConcern", {})
                .get("afterClusterTime")
            )
            self.assertIsNone(act)

    @client_context.require_no_standalone
    def test_read_concern(self):
        with self.client.start_session(causal_consistency=True) as s:
            coll = self.client.pymongo_test.test
            coll.insert_one({}, session=s)
            self.listener.reset()
            coll.find_one({}, session=s)
            read_concern = self.listener.started_events[0].command.get("readConcern")
            self.assertIsNotNone(read_concern)
            self.assertIsNone(read_concern.get("level"))
            self.assertIsNotNone(read_concern.get("afterClusterTime"))

            coll = coll.with_options(read_concern=ReadConcern("majority"))
            self.listener.reset()
            coll.find_one({}, session=s)
            read_concern = self.listener.started_events[0].command.get("readConcern")
            self.assertIsNotNone(read_concern)
            self.assertEqual(read_concern.get("level"), "majority")
            self.assertIsNotNone(read_concern.get("afterClusterTime"))

    @client_context.require_no_standalone
    def test_cluster_time_with_server_support(self):
        self.client.pymongo_test.test.insert_one({})
        self.listener.reset()
        self.client.pymongo_test.test.find_one({})
        after_cluster_time = self.listener.started_events[0].command.get("$clusterTime")
        self.assertIsNotNone(after_cluster_time)

    @client_context.require_standalone
    def test_cluster_time_no_server_support(self):
        self.client.pymongo_test.test.insert_one({})
        self.listener.reset()
        self.client.pymongo_test.test.find_one({})
        after_cluster_time = self.listener.started_events[0].command.get("$clusterTime")
        self.assertIsNone(after_cluster_time)


class TestClusterTime(IntegrationTest):
    def setUp(self):
        super().setUp()
        if "$clusterTime" not in (client_context.hello):
            raise SkipTest("$clusterTime not supported")

    # Sessions prose test: 3) $clusterTime in commands
    def test_cluster_time(self):
        listener = SessionTestListener()
        client = self.rs_or_single_client(event_listeners=[listener])
        collection = client.pymongo_test.collection
        # Prepare for tests of find() and aggregate().
        collection.insert_many([{} for _ in range(10)])
        self.addCleanup(collection.drop)
        self.addCleanup(client.pymongo_test.collection2.drop)

        def rename_and_drop():
            # Ensure collection exists.
            collection.insert_one({})
            collection.rename("collection2")
            client.pymongo_test.collection2.drop()

        def insert_and_find():
            cursor = collection.find().batch_size(1)
            for _ in range(10):
                # Advance the cluster time.
                collection.insert_one({})
                next(cursor)

            cursor.close()

        def insert_and_aggregate():
            cursor = (collection.aggregate([], batchSize=1)).batch_size(1)
            for _ in range(5):
                # Advance the cluster time.
                collection.insert_one({})
                next(cursor)

            cursor.close()

        def aggregate():
            (collection.aggregate([])).to_list()

        ops = [
            # Tests from Driver Sessions Spec.
            ("ping", lambda: client.admin.command("ping")),
            ("aggregate", lambda: aggregate()),
            ("find", lambda: collection.find().to_list()),
            ("insert_one", lambda: collection.insert_one({})),
            # Additional PyMongo tests.
            ("insert_and_find", insert_and_find),
            ("insert_and_aggregate", insert_and_aggregate),
            ("update_one", lambda: collection.update_one({}, {"$set": {"x": 1}})),
            ("update_many", lambda: collection.update_many({}, {"$set": {"x": 1}})),
            ("delete_one", lambda: collection.delete_one({})),
            ("delete_many", lambda: collection.delete_many({})),
            ("bulk_write", lambda: collection.bulk_write([InsertOne({})])),
            ("rename_and_drop", rename_and_drop),
        ]

        for _name, f in ops:
            listener.reset()
            # Call f() twice, insert to advance clusterTime, call f() again.
            f()
            f()
            collection.insert_one({})
            f()

            self.assertGreaterEqual(len(listener.started_events), 1)
            for i, event in enumerate(listener.started_events):
                self.assertIn(
                    "$clusterTime",
                    event.command,
                    f"{f.__name__} sent no $clusterTime with {event.command_name}",
                )

                if i > 0:
                    succeeded = listener.succeeded_events[i - 1]
                    self.assertIn(
                        "$clusterTime",
                        succeeded.reply,
                        f"{f.__name__} received no $clusterTime with {succeeded.command_name}",
                    )

                    self.assertTrue(
                        event.command["$clusterTime"]["clusterTime"]
                        >= succeeded.reply["$clusterTime"]["clusterTime"],
                        f"{f.__name__} sent wrong $clusterTime with {event.command_name}",
                    )

    # Sessions prose test: 20) Drivers do not gossip `$clusterTime` on SDAM commands
    def test_cluster_time_not_used_by_sdam(self):
        heartbeat_listener = HeartbeatEventListener()
        cmd_listener = OvertCommandListener()
        with client_knobs(min_heartbeat_interval=0.01):
            c1 = self.single_client(
                event_listeners=[heartbeat_listener, cmd_listener], heartbeatFrequencyMS=10
            )
            cluster_time = (c1.admin.command({"ping": 1}))["$clusterTime"]
            self.assertEqual(c1._topology.max_cluster_time(), cluster_time)

            # Advance the server's $clusterTime by performing an insert via another client.
            self.db.test.insert_one({"advance": "$clusterTime"})
            # Wait until the client C1 processes the next pair of SDAM heartbeat started + succeeded events.
            heartbeat_listener.reset()

            def next_heartbeat():
                events = heartbeat_listener.events
                for i in range(len(events) - 1):
                    if isinstance(events[i], monitoring.ServerHeartbeatStartedEvent):
                        if isinstance(events[i + 1], monitoring.ServerHeartbeatSucceededEvent):
                            return True
                return False

            wait_until(next_heartbeat, "never found pair of heartbeat started + succeeded events")
            # Assert that C1's max $clusterTime is still the same and has not been updated by SDAM.
            cmd_listener.reset()
            c1.admin.command({"ping": 1})
            started = cmd_listener.started_events[0]
            self.assertEqual(started.command_name, "ping")
            self.assertEqual(started.command["$clusterTime"], cluster_time)


if __name__ == "__main__":
    unittest.main()
