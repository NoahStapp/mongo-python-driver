# Copyright 2023-present MongoDB, Inc.
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

"""Test MONGODB-OIDC Authentication."""
from __future__ import annotations

import os
import sys
import time
import unittest
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from test.asynchronous import AsyncPyMongoTestCase
from test.asynchronous.helpers import ConcurrentRunner
from typing import Dict

import pytest

sys.path[0:0] = [""]

from test.asynchronous.unified_format import generate_test_classes
from test.utils_shared import EventListener, OvertCommandListener

from bson import SON
from pymongo import AsyncMongoClient
from pymongo._azure_helpers import _get_azure_response
from pymongo._gcp_helpers import _get_gcp_response
from pymongo.asynchronous.auth_oidc import (
    OIDCCallback,
    OIDCCallbackContext,
    OIDCCallbackResult,
    _get_authenticator,
)
from pymongo.auth_oidc_shared import _get_k8s_token
from pymongo.auth_shared import _build_credentials_tuple
from pymongo.cursor_shared import CursorType
from pymongo.errors import AutoReconnect, ConfigurationError, OperationFailure
from pymongo.hello import HelloCompat
from pymongo.operations import InsertOne
from pymongo.synchronous.uri_parser import parse_uri

_IS_SYNC = False

ROOT = Path(__file__).parent.parent.resolve()
TEST_PATH = ROOT / "auth" / "unified"
ENVIRON = os.environ.get("OIDC_ENV", "test")
DOMAIN = os.environ.get("OIDC_DOMAIN", "")
TOKEN_DIR = os.environ.get("OIDC_TOKEN_DIR", "")
TOKEN_FILE = os.environ.get("OIDC_TOKEN_FILE", "")

# Generate unified tests.
globals().update(generate_test_classes(str(TEST_PATH), module=__name__))

pytestmark = pytest.mark.auth_oidc


class OIDCTestBase(AsyncPyMongoTestCase):
    @classmethod
    def setUpClass(cls):
        cls.uri_single = os.environ["MONGODB_URI_SINGLE"]
        cls.uri_multiple = os.environ.get("MONGODB_URI_MULTI")
        cls.uri_admin = os.environ["MONGODB_URI"]
        if ENVIRON == "test":
            if not TOKEN_DIR:
                raise ValueError("Please set OIDC_TOKEN_DIR")
            if not TOKEN_FILE:
                raise ValueError("Please set OIDC_TOKEN_FILE")

    async def asyncSetUp(self):
        self.request_called = 0

    def get_token(self, username=None):
        """Get a token for the current provider."""
        if ENVIRON == "test":
            if username is None:
                token_file = TOKEN_FILE
            else:
                token_file = os.path.join(TOKEN_DIR, username)
            with open(token_file) as fid:  # noqa: ASYNC101,RUF100
                return fid.read()
        elif ENVIRON == "azure":
            opts = parse_uri(self.uri_single)["options"]
            token_aud = opts["authMechanismProperties"]["TOKEN_RESOURCE"]
            return _get_azure_response(token_aud, username)["access_token"]
        elif ENVIRON == "gcp":
            opts = parse_uri(self.uri_single)["options"]
            token_aud = opts["authMechanismProperties"]["TOKEN_RESOURCE"]
            return _get_gcp_response(token_aud, username)["access_token"]
        elif ENVIRON == "k8s":
            return _get_k8s_token()
        else:
            raise ValueError(f"Unknown ENVIRON: {ENVIRON}")

    @asynccontextmanager
    async def fail_point(self, command_args):
        cmd_on = SON([("configureFailPoint", "failCommand")])
        cmd_on.update(command_args)
        client = AsyncMongoClient(self.uri_admin)
        await client.admin.command(cmd_on)
        try:
            yield
        finally:
            await client.admin.command(
                "configureFailPoint", cmd_on["configureFailPoint"], mode="off"
            )
            await client.close()


class TestAuthOIDCHuman(OIDCTestBase):
    uri: str

    @classmethod
    def setUpClass(cls):
        if ENVIRON != "test":
            raise unittest.SkipTest("Human workflows are only tested with the test environment")
        if DOMAIN is None:
            raise ValueError("Missing OIDC_DOMAIN")
        super().setUpClass()

    async def asyncSetUp(self):
        self.refresh_present = 0
        await super().asyncSetUp()

    def create_request_cb(self, username="test_user1", sleep=0):
        def request_token(context: OIDCCallbackContext):
            # Validate the info.
            self.assertIsInstance(context.idp_info.issuer, str)
            if context.idp_info.clientId is not None:
                self.assertIsInstance(context.idp_info.clientId, str)

            # Validate the timeout.
            timeout_seconds = context.timeout_seconds
            self.assertEqual(timeout_seconds, 60 * 5)

            if context.refresh_token:
                self.refresh_present += 1

            token = self.get_token(username)
            resp = OIDCCallbackResult(access_token=token, refresh_token=token)

            time.sleep(sleep)
            self.request_called += 1
            return resp

        class Inner(OIDCCallback):
            def fetch(self, context):
                return request_token(context)

        return Inner()

    async def create_client(self, *args, **kwargs):
        username = kwargs.get("username", "test_user1")
        if kwargs.get("username") in ["test_user1", "test_user2"]:
            kwargs["username"] = f"{username}@{DOMAIN}"
        request_cb = kwargs.pop("request_cb", self.create_request_cb(username=username))
        props = kwargs.pop("authmechanismproperties", {"OIDC_HUMAN_CALLBACK": request_cb})
        kwargs["retryReads"] = False
        if not len(args):
            args = [self.uri_single]

        client = self.simple_client(*args, authmechanismproperties=props, **kwargs)

        return client

    async def test_1_1_single_principal_implicit_username(self):
        # Create default OIDC client with authMechanism=MONGODB-OIDC.
        client = await self.create_client()
        # Perform a find operation that succeeds.
        await client.test.test.find_one()
        # Close the client.
        await client.close()

    async def test_1_2_single_principal_explicit_username(self):
        # Create a client with MONGODB_URI_SINGLE, a username of test_user1, authMechanism=MONGODB-OIDC, and the OIDC human callback.
        client = await self.create_client(username="test_user1")
        # Perform a find operation that succeeds.
        await client.test.test.find_one()
        # Close the client.
        await client.close()

    async def test_1_3_multiple_principal_user_1(self):
        if not self.uri_multiple:
            raise unittest.SkipTest("Test Requires Server with Multiple Workflow IdPs")
        # Create a client with MONGODB_URI_MULTI, a username of test_user1, authMechanism=MONGODB-OIDC, and the OIDC human callback.
        client = await self.create_client(self.uri_multiple, username="test_user1")
        # Perform a find operation that succeeds.
        await client.test.test.find_one()
        # Close the client.
        await client.close()

    async def test_1_4_multiple_principal_user_2(self):
        if not self.uri_multiple:
            raise unittest.SkipTest("Test Requires Server with Multiple Workflow IdPs")
        # Create a human callback that reads in the generated test_user2 token file.
        # Create a client with MONGODB_URI_MULTI, a username of test_user2, authMechanism=MONGODB-OIDC, and the OIDC human callback.
        client = await self.create_client(self.uri_multiple, username="test_user2")
        # Perform a find operation that succeeds.
        await client.test.test.find_one()
        # Close the client.
        await client.close()

    async def test_1_5_multiple_principal_no_user(self):
        if not self.uri_multiple:
            raise unittest.SkipTest("Test Requires Server with Multiple Workflow IdPs")
        # Create a client with MONGODB_URI_MULTI, no username, authMechanism=MONGODB-OIDC, and the OIDC human callback.
        client = await self.create_client(self.uri_multiple)
        # Assert that a find operation fails.
        with self.assertRaises(OperationFailure):
            await client.test.test.find_one()
        # Close the client.
        await client.close()

    async def test_1_6_allowed_hosts_blocked(self):
        # Create a default OIDC client, with an ALLOWED_HOSTS that is an empty list.
        request_token = self.create_request_cb()
        props: Dict = {"OIDC_HUMAN_CALLBACK": request_token, "ALLOWED_HOSTS": []}
        client = await self.create_client(authmechanismproperties=props)
        # Assert that a find operation fails with a client-side error.
        with self.assertRaises(ConfigurationError):
            await client.test.test.find_one()
        # Close the client.
        await client.close()

        # Create a client that uses the URL mongodb://localhost/?authMechanism=MONGODB-OIDC&ignored=example.com,
        # a human callback, and an ALLOWED_HOSTS that contains ["example.com"].
        props: Dict = {
            "OIDC_HUMAN_CALLBACK": request_token,
            "ALLOWED_HOSTS": ["example.com"],
        }
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            client = await self.create_client(
                self.uri_single + "&ignored=example.com",
                authmechanismproperties=props,
                connect=False,
            )
            # Assert that a find operation fails with a client-side error.
            with self.assertRaises(ConfigurationError):
                await client.test.test.find_one()
        # Close the client.
        await client.close()

    async def test_1_7_allowed_hosts_in_connection_string_ignored(self):
        # Create an OIDC configured client with the connection string: `mongodb+srv://example.com/?authMechanism=MONGODB-OIDC&authMechanismProperties=ALLOWED_HOSTS:%5B%22example.com%22%5D` and a Human Callback.
        # Assert that the creation of the client raises a configuration error.
        uri = "mongodb+srv://example.com?authMechanism=MONGODB-OIDC&authMechanismProperties=ALLOWED_HOSTS:%5B%22example.com%22%5D"
        with self.assertRaises(ConfigurationError), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = AsyncMongoClient(
                uri,
                authmechanismproperties=dict(OIDC_HUMAN_CALLBACK=self.create_request_cb()),
            )
            await c.aconnect()

    async def test_1_8_machine_idp_human_callback(self):
        if not os.environ.get("OIDC_IS_LOCAL"):
            raise unittest.SkipTest("Test Requires Local OIDC server")
        # Create a client with MONGODB_URI_SINGLE, a username of test_machine, authMechanism=MONGODB-OIDC, and the OIDC human callback.
        client = await self.create_client(username="test_machine")
        # Perform a find operation that succeeds.
        await client.test.test.find_one()
        # Close the client.
        await client.close()

    async def test_2_1_valid_callback_inputs(self):
        # Create a AsyncMongoClient with a human callback that validates its inputs and returns a valid access token.
        client = await self.create_client()
        # Perform a find operation that succeeds. Verify that the human callback was called with the appropriate inputs, including the timeout parameter if possible.
        # Ensure that there are no unexpected fields.
        await client.test.test.find_one()
        # Close the client.
        await client.close()

    async def test_2_2_callback_returns_missing_data(self):
        # Create a AsyncMongoClient with a human callback that returns data not conforming to the OIDCCredential with missing fields.
        class CustomCB(OIDCCallback):
            def fetch(self, ctx):
                return dict()

        client = await self.create_client(request_cb=CustomCB())
        # Perform a find operation that fails.
        with self.assertRaises(ValueError):
            await client.test.test.find_one()
        # Close the client.
        await client.close()

    async def test_2_3_refresh_token_is_passed_to_the_callback(self):
        # Create a AsyncMongoClient with a human callback that checks for the presence of a refresh token.
        client = await self.create_client()

        # Perform a find operation that succeeds.
        await client.test.test.find_one()

        # Set a fail point for ``find`` commands.
        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391},
            }
        ):
            # Perform a ``find`` operation that succeeds.
            await client.test.test.find_one()

        # Assert that the callback has been called twice.
        self.assertEqual(self.request_called, 2)

        # Assert that the refresh token was used once.
        self.assertEqual(self.refresh_present, 1)

    async def test_3_1_uses_speculative_authentication_if_there_is_a_cached_token(self):
        # Create a client with a human callback that returns a valid token.
        client = await self.create_client()

        # Set a fail point for ``find`` commands.
        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391, "closeConnection": True},
            }
        ):
            # Perform a ``find`` operation that fails.
            with self.assertRaises(AutoReconnect):
                await client.test.test.find_one()

        # Set a fail point for ``saslStart`` commands.
        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["saslStart"], "errorCode": 18},
            }
        ):
            # Perform a ``find`` operation that succeeds
            await client.test.test.find_one()

        # Close the client.
        await client.close()

    async def test_3_2_does_not_use_speculative_authentication_if_there_is_no_cached_token(self):
        # Create a ``AsyncMongoClient`` with a human callback that returns a valid token
        client = await self.create_client()

        # Set a fail point for ``saslStart`` commands.
        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["saslStart"], "errorCode": 18},
            }
        ):
            # Perform a ``find`` operation that fails.
            with self.assertRaises(OperationFailure):
                await client.test.test.find_one()

        # Close the client.
        await client.close()

    async def test_4_1_reauthenticate_succeeds(self):
        # Create a default OIDC client and add an event listener.
        # The following assumes that the driver does not emit saslStart or saslContinue events.
        # If the driver does emit those events, ignore/filter them for the purposes of this test.
        listener = OvertCommandListener()
        client = await self.create_client(event_listeners=[listener])

        # Perform a find operation that succeeds.
        await client.test.test.find_one()

        # Assert that the human callback has been called once.
        self.assertEqual(self.request_called, 1)

        # Clear the listener state if possible.
        listener.reset()

        # Force a reauthenication using a fail point.
        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391},
            }
        ):
            # Perform another find operation that succeeds.
            await client.test.test.find_one()

        # Assert that the human callback has been called twice.
        self.assertEqual(self.request_called, 2)

        # Assert that the ordering of list started events is [find, find].
        # Note that if the listener stat could not be cleared then there will be an extra find command.
        started_events = [
            i.command_name for i in listener.started_events if not i.command_name.startswith("sasl")
        ]
        succeeded_events = [
            i.command_name
            for i in listener.succeeded_events
            if not i.command_name.startswith("sasl")
        ]
        failed_events = [
            i.command_name for i in listener.failed_events if not i.command_name.startswith("sasl")
        ]

        self.assertEqual(
            started_events,
            [
                "find",
                "find",
            ],
        )
        # Assert that the list of command succeeded events is [find].
        self.assertEqual(succeeded_events, ["find"])
        # Assert that a find operation failed once during the command execution.
        self.assertEqual(failed_events, ["find"])
        # Close the client.
        await client.close()

    async def test_4_2_reauthenticate_succeeds_no_refresh(self):
        # Create a default OIDC client with a human callback that does not return a refresh token.
        cb = self.create_request_cb()

        class CustomRequest(OIDCCallback):
            def fetch(self, *args, **kwargs):
                result = cb.fetch(*args, **kwargs)
                result.refresh_token = None
                return result

        client = await self.create_client(request_cb=CustomRequest())

        # Perform a find operation that succeeds.
        await client.test.test.find_one()

        # Assert that the human callback has been called once.
        self.assertEqual(self.request_called, 1)

        # Force a reauthenication using a fail point.
        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391},
            }
        ):
            # Perform a find operation that succeeds.
            await client.test.test.find_one()

        # Assert that the human callback has been called twice.
        self.assertEqual(self.request_called, 2)
        # Close the client.
        await client.close()

    async def test_4_3_reauthenticate_succeeds_after_refresh_fails(self):
        # Create a default OIDC client with a human callback that returns an invalid refresh token
        cb = self.create_request_cb()

        class CustomRequest(OIDCCallback):
            def fetch(self, *args, **kwargs):
                result = cb.fetch(*args, **kwargs)
                result.refresh_token = "bad"
                return result

        client = await self.create_client(request_cb=CustomRequest())

        # Perform a find operation that succeeds.
        await client.test.test.find_one()

        # Assert that the human callback has been called once.
        self.assertEqual(self.request_called, 1)

        # Force a reauthenication using a fail point.
        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391},
            }
        ):
            # Perform a find operation that succeeds.
            await client.test.test.find_one()

        # Assert that the human callback has been called 2 times.
        self.assertEqual(self.request_called, 2)

        # Close the client.
        await client.close()

    async def test_4_4_reauthenticate_fails(self):
        # Create a default OIDC client with a human callback that returns invalid refresh tokens and
        # Returns invalid access tokens after the first access.
        cb = self.create_request_cb()

        class CustomRequest(OIDCCallback):
            fetch_called = 0

            def fetch(self, *args, **kwargs):
                self.fetch_called += 1
                result = cb.fetch(*args, **kwargs)
                result.refresh_token = "bad"
                if self.fetch_called > 1:
                    result.access_token = "bad"
                return result

        client = await self.create_client(request_cb=CustomRequest())

        # Perform a find operation that succeeds (to force a speculative auth).
        await client.test.test.find_one()
        # Assert that the human callback has been called once.
        self.assertEqual(self.request_called, 1)

        # Force a reauthentication using a failCommand.
        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391},
            }
        ):
            # Perform a find operation that fails.
            with self.assertRaises(OperationFailure):
                await client.test.test.find_one()

        # Assert that the human callback has been called three times.
        self.assertEqual(self.request_called, 3)

        # Close the client.
        await client.close()

    async def test_request_callback_returns_null(self):
        class RequestTokenNull(OIDCCallback):
            def fetch(self, a):
                return None

        client = await self.create_client(request_cb=RequestTokenNull())
        with self.assertRaises(ValueError):
            await client.test.test.find_one()
        await client.close()

    async def test_request_callback_invalid_result(self):
        class CallbackInvalidToken(OIDCCallback):
            def fetch(self, a):
                return {}

        client = await self.create_client(request_cb=CallbackInvalidToken())
        with self.assertRaises(ValueError):
            await client.test.test.find_one()
        await client.close()

    async def test_reauthentication_succeeds_multiple_connections(self):
        request_cb = self.create_request_cb()

        # Create a client with the callback.
        client1 = await self.create_client(request_cb=request_cb)
        client2 = await self.create_client(request_cb=request_cb)

        # Perform an insert operation.
        await client1.test.test.insert_many([{"a": 1}, {"a": 1}])
        await client2.test.test.find_one()
        self.assertEqual(self.request_called, 2)

        # Use the same authenticator for both clients
        # to simulate a race condition with separate connections.
        # We should only see one extra callback despite both connections
        # needing to reauthenticate.
        client2.options.pool_options._credentials.cache.data = (
            client1.options.pool_options._credentials.cache.data
        )

        await client1.test.test.find_one()
        await client2.test.test.find_one()

        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391},
            }
        ):
            await client1.test.test.find_one()

        self.assertEqual(self.request_called, 3)

        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391},
            }
        ):
            await client2.test.test.find_one()

        self.assertEqual(self.request_called, 3)
        await client1.close()
        await client2.close()

    # PyMongo specific tests, since we have multiple code paths for reauth handling.

    async def test_reauthenticate_succeeds_bulk_write(self):
        # Create a client.
        client = await self.create_client()

        # Perform a find operation.
        await client.test.test.find_one()

        # Assert that the request callback has been called once.
        self.assertEqual(self.request_called, 1)

        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["insert"], "errorCode": 391},
            }
        ):
            # Perform a bulk write operation.
            await client.test.test.bulk_write([InsertOne({})])  # type:ignore[type-var]

        # Assert that the request callback has been called twice.
        self.assertEqual(self.request_called, 2)
        await client.close()

    async def test_reauthenticate_succeeds_bulk_read(self):
        # Create a client.
        client = await self.create_client()

        # Perform a find operation.
        await client.test.test.find_one()

        # Perform a bulk write operation.
        await client.test.test.bulk_write([InsertOne({})])  # type:ignore[type-var]

        # Assert that the request callback has been called once.
        self.assertEqual(self.request_called, 1)

        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391},
            }
        ):
            # Perform a bulk read operation.
            cursor = client.test.test.find_raw_batches({})
            await cursor.to_list()

        # Assert that the request callback has been called twice.
        self.assertEqual(self.request_called, 2)
        await client.close()

    async def test_reauthenticate_succeeds_cursor(self):
        # Create a client.
        client = await self.create_client()

        # Perform an insert operation.
        await client.test.test.insert_one({"a": 1})

        # Assert that the request callback has been called once.
        self.assertEqual(self.request_called, 1)

        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391},
            }
        ):
            # Perform a find operation.
            cursor = client.test.test.find({"a": 1})
            self.assertGreaterEqual(len(await cursor.to_list()), 1)

        # Assert that the request callback has been called twice.
        self.assertEqual(self.request_called, 2)
        await client.close()

    async def test_reauthenticate_succeeds_get_more(self):
        # Create a client.
        client = await self.create_client()

        # Perform an insert operation.
        await client.test.test.insert_many([{"a": 1}, {"a": 1}])

        # Assert that the request callback has been called once.
        self.assertEqual(self.request_called, 1)

        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["getMore"], "errorCode": 391},
            }
        ):
            # Perform a find operation.
            cursor = client.test.test.find({"a": 1}, batch_size=1)
            self.assertGreaterEqual(len(await cursor.to_list()), 1)

        # Assert that the request callback has been called twice.
        self.assertEqual(self.request_called, 2)
        await client.close()

    async def test_reauthenticate_succeeds_get_more_exhaust(self):
        # Ensure no mongos
        client = await self.create_client()
        hello = await client.admin.command(HelloCompat.LEGACY_CMD)
        if hello.get("msg") != "isdbgrid":
            raise unittest.SkipTest("Must not be a mongos")

        # Create a client with the callback.
        client = await self.create_client()

        # Perform an insert operation.
        await client.test.test.insert_many([{"a": 1}, {"a": 1}])

        # Assert that the request callback has been called once.
        self.assertEqual(self.request_called, 1)

        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["getMore"], "errorCode": 391},
            }
        ):
            # Perform a find operation.
            cursor = client.test.test.find({"a": 1}, batch_size=1, cursor_type=CursorType.EXHAUST)
            self.assertGreaterEqual(len(await cursor.to_list()), 1)

        # Assert that the request callback has been called twice.
        self.assertEqual(self.request_called, 2)
        await client.close()

    async def test_reauthenticate_succeeds_command(self):
        # Create a client.
        client = await self.create_client()

        # Perform an insert operation.
        await client.test.test.insert_one({"a": 1})

        # Assert that the request callback has been called once.
        self.assertEqual(self.request_called, 1)

        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["count"], "errorCode": 391},
            }
        ):
            # Perform a count operation.
            cursor = await client.test.command({"count": "test"})

        self.assertGreaterEqual(len(cursor), 1)

        # Assert that the request callback has been called twice.
        self.assertEqual(self.request_called, 2)
        await client.close()


class TestAuthOIDCMachine(OIDCTestBase):
    uri: str

    async def asyncSetUp(self):
        self.request_called = 0

    def create_request_cb(self, username=None, sleep=0):
        def request_token(context):
            assert isinstance(context.timeout_seconds, int)
            assert context.version == 1
            assert context.refresh_token is None
            assert context.idp_info is None
            token = self.get_token(username)
            time.sleep(sleep)
            self.request_called += 1
            return OIDCCallbackResult(access_token=token)

        class Inner(OIDCCallback):
            def fetch(self, context):
                return request_token(context)

        return Inner()

    async def create_client(self, *args, **kwargs):
        request_cb = kwargs.pop("request_cb", self.create_request_cb())
        props = kwargs.pop("authmechanismproperties", {"OIDC_CALLBACK": request_cb})
        kwargs["retryReads"] = False
        if not len(args):
            args = [self.uri_single]
        client = AsyncMongoClient(*args, authmechanismproperties=props, **kwargs)
        self.addAsyncCleanup(client.close)
        return client

    async def test_1_1_callback_is_called_during_reauthentication(self):
        # Create a ``AsyncMongoClient`` configured with a custom OIDC callback that
        # implements the provider logic.
        client = await self.create_client()
        # Perform a ``find`` operation that succeeds.
        await client.test.test.find_one()
        # Assert that the callback was called 1 time.
        self.assertEqual(self.request_called, 1)

    async def test_1_2_callback_is_called_once_for_multiple_connections(self):
        # Create a ``AsyncMongoClient`` configured with a custom OIDC callback that
        # implements the provider logic.
        client = await self.create_client()
        await client.aconnect()

        # Start 10 tasks and run 100 find operations that all succeed in each task.
        async def target():
            for _ in range(100):
                await client.test.test.find_one()

        tasks = []
        for i in range(10):
            tasks.append(ConcurrentRunner(target=target))
        for t in tasks:
            await t.start()
        for t in tasks:
            await t.join()
        # Assert that the callback was called 1 time.
        self.assertEqual(self.request_called, 1)

    async def test_2_1_valid_callback_inputs(self):
        # Create a AsyncMongoClient configured with an OIDC callback that validates its inputs and returns a valid access token.
        client = await self.create_client()
        # Perform a find operation that succeeds.
        await client.test.test.find_one()
        # Assert that the OIDC callback was called with the appropriate inputs, including the timeout parameter if possible. Ensure that there are no unexpected fields.
        self.assertEqual(self.request_called, 1)

    async def test_2_2_oidc_callback_returns_null(self):
        # Create a AsyncMongoClient configured with an OIDC callback that returns null.
        class CallbackNullToken(OIDCCallback):
            def fetch(self, a):
                return None

        client = await self.create_client(request_cb=CallbackNullToken())
        # Perform a find operation that fails.
        with self.assertRaises(ValueError):
            await client.test.test.find_one()

    async def test_2_3_oidc_callback_returns_missing_data(self):
        # Create a AsyncMongoClient configured with an OIDC callback that returns data not conforming to the OIDCCredential with missing fields.
        class CustomCallback(OIDCCallback):
            count = 0

            def fetch(self, a):
                self.count += 1
                return object()

        client = await self.create_client(request_cb=CustomCallback())
        # Perform a find operation that fails.
        with self.assertRaises(ValueError):
            await client.test.test.find_one()

    async def test_2_4_invalid_client_configuration_with_callback(self):
        # Create a AsyncMongoClient configured with an OIDC callback and auth mechanism property ENVIRONMENT:test.
        request_cb = self.create_request_cb()
        props: Dict = {"OIDC_CALLBACK": request_cb, "ENVIRONMENT": "test"}
        # Assert it returns a client configuration error.
        with self.assertRaises(ConfigurationError):
            await self.create_client(authmechanismproperties=props)

    async def test_2_5_invalid_use_of_ALLOWED_HOSTS(self):
        # Create an OIDC configured client with auth mechanism properties `{"ENVIRONMENT": "test", "ALLOWED_HOSTS": []}`.
        props: Dict = {"ENVIRONMENT": "test", "ALLOWED_HOSTS": []}
        # Assert it returns a client configuration error.
        with self.assertRaises(ConfigurationError):
            await self.create_client(authmechanismproperties=props)

        # Create an OIDC configured client with auth mechanism properties `{"OIDC_CALLBACK": "<my_callback>", "ALLOWED_HOSTS": []}`.
        props: Dict = {"OIDC_CALLBACK": self.create_request_cb(), "ALLOWED_HOSTS": []}
        # Assert it returns a client configuration error.
        with self.assertRaises(ConfigurationError):
            await self.create_client(authmechanismproperties=props)

    async def test_2_6_ALLOWED_HOSTS_defaults_ignored(self):
        # Create a MongoCredential for OIDC with a machine callback.
        props = {"OIDC_CALLBACK": self.create_request_cb()}
        extra = dict(authmechanismproperties=props)
        mongo_creds = _build_credentials_tuple("MONGODB-OIDC", None, "foo", None, extra, "test")
        # Assert that creating an authenticator for example.com does not result in an error.
        authenticator = _get_authenticator(mongo_creds, ("example.com", 30))
        assert authenticator.properties.username == "foo"

        # Create a MongoCredential for OIDC with an ENVIRONMENT.
        props = {"ENVIRONMENT": "test"}
        extra = dict(authmechanismproperties=props)
        mongo_creds = _build_credentials_tuple("MONGODB-OIDC", None, None, None, extra, "test")
        # Assert that creating an authenticator for example.com does not result in an error.
        authenticator = _get_authenticator(mongo_creds, ("example.com", 30))
        assert authenticator.properties.username == ""

    async def test_3_1_authentication_failure_with_cached_tokens_fetch_a_new_token_and_retry(self):
        # Create a AsyncMongoClient and an OIDC callback that implements the provider logic.
        client = await self.create_client()
        await client.aconnect()
        # Poison the cache with an invalid access token.
        # Set a fail point for ``find`` command.
        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391, "closeConnection": True},
            }
        ):
            # Perform a ``find`` operation that fails. This is to force the ``AsyncMongoClient``
            # to cache an access token.
            with self.assertRaises(AutoReconnect):
                await client.test.test.find_one()
        # Poison the cache of the client.
        client.options.pool_options._credentials.cache.data.access_token = "bad"
        # Reset the request count.
        self.request_called = 0
        # Verify that a find succeeds.
        await client.test.test.find_one()
        # Verify that the callback was called 1 time.
        self.assertEqual(self.request_called, 1)

    async def test_3_2_authentication_failures_without_cached_tokens_returns_an_error(self):
        # Create a AsyncMongoClient configured with retryReads=false and an OIDC callback that always returns invalid access tokens.
        class CustomCallback(OIDCCallback):
            count = 0

            def fetch(self, a):
                self.count += 1
                return OIDCCallbackResult(access_token="bad value")

        callback = CustomCallback()
        client = await self.create_client(request_cb=callback)
        # Perform a ``find`` operation that fails.
        with self.assertRaises(OperationFailure):
            await client.test.test.find_one()
        # Verify that the callback was called 1 time.
        self.assertEqual(callback.count, 1)

    async def test_3_3_unexpected_error_code_does_not_clear_cache(self):
        # Create a ``AsyncMongoClient`` with a human callback that returns a valid token
        client = await self.create_client()

        # Set a fail point for ``saslStart`` commands.
        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["saslStart"], "errorCode": 20},
            }
        ):
            # Perform a ``find`` operation that fails.
            with self.assertRaises(OperationFailure):
                await client.test.test.find_one()

        # Assert that the callback has been called once.
        self.assertEqual(self.request_called, 1)

        # Perform a ``find`` operation that succeeds.
        await client.test.test.find_one()

        # Assert that the callback has been called once.
        self.assertEqual(self.request_called, 1)

    async def test_4_1_reauthentication_succeeds(self):
        # Create a ``AsyncMongoClient`` configured with a custom OIDC callback that
        # implements the provider logic.
        client = await self.create_client()
        await client.aconnect()

        # Set a fail point for the find command.
        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391},
            }
        ):
            # Perform a ``find`` operation that succeeds.
            await client.test.test.find_one()

        # Verify that the callback was called 2 times (once during the connection
        # handshake, and again during reauthentication).
        self.assertEqual(self.request_called, 2)

    async def test_4_2_read_commands_fail_if_reauthentication_fails(self):
        # Create a ``AsyncMongoClient`` whose OIDC callback returns one good token and then
        # bad tokens after the first call.
        get_token = self.get_token

        class CustomCallback(OIDCCallback):
            count = 0

            def fetch(self, _):
                self.count += 1
                if self.count == 1:
                    access_token = get_token()
                else:
                    access_token = "bad value"
                return OIDCCallbackResult(access_token=access_token)

        callback = CustomCallback()
        client = await self.create_client(request_cb=callback)

        # Perform a read operation that succeeds.
        await client.test.test.find_one()

        # Set a fail point for the find command.
        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391},
            }
        ):
            # Perform a ``find`` operation that fails.
            with self.assertRaises(OperationFailure):
                await client.test.test.find_one()

        # Verify that the callback was called 2 times.
        self.assertEqual(callback.count, 2)

    async def test_4_3_write_commands_fail_if_reauthentication_fails(self):
        # Create a ``AsyncMongoClient`` whose OIDC callback returns one good token and then
        # bad token after the first call.
        get_token = self.get_token

        class CustomCallback(OIDCCallback):
            count = 0

            def fetch(self, _):
                self.count += 1
                if self.count == 1:
                    access_token = get_token()
                else:
                    access_token = "bad value"
                return OIDCCallbackResult(access_token=access_token)

        callback = CustomCallback()
        client = await self.create_client(request_cb=callback)

        # Perform an insert operation that succeeds.
        await client.test.test.insert_one({})

        # Set a fail point for the find command.
        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["insert"], "errorCode": 391},
            }
        ):
            # Perform a ``insert`` operation that fails.
            with self.assertRaises(OperationFailure):
                await client.test.test.insert_one({})

        # Verify that the callback was called 2 times.
        self.assertEqual(callback.count, 2)

    async def test_4_4_speculative_authentication_should_be_ignored_on_reauthentication(self):
        # Create an OIDC configured client that can listen for `SaslStart` commands.
        listener = EventListener()
        client = await self.create_client(event_listeners=[listener])
        await client.aconnect()

        # Preload the *Client Cache* with a valid access token to enforce Speculative Authentication.
        client2 = await self.create_client()
        await client2.test.test.find_one()
        client.options.pool_options._credentials.cache.data = (
            client2.options.pool_options._credentials.cache.data
        )
        await client2.close()
        self.request_called = 0

        # Perform an `insert` operation that succeeds.
        await client.test.test.insert_one({})

        # Assert that the callback was not called.
        self.assertEqual(self.request_called, 0)

        # Assert there were no `SaslStart` commands executed.
        assert not any(
            event.command_name.lower() == "saslstart" for event in listener.started_events
        )
        listener.reset()

        # Set a fail point for `insert` commands of the form:
        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["insert"], "errorCode": 391},
            }
        ):
            # Perform an `insert` operation that succeeds.
            await client.test.test.insert_one({})

        # Assert that the callback was called once.
        self.assertEqual(self.request_called, 1)

        # Assert there were `SaslStart` commands executed.
        assert any(event.command_name.lower() == "saslstart" for event in listener.started_events)

    async def test_4_5_reauthentication_succeeds_when_a_session_is_involved(self):
        # Create an OIDC configured client.
        client = await self.create_client()

        # Set a fail point for `find` commands of the form:
        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391},
            }
        ):
            # Start a new session.
            async with client.start_session() as session:
                # In the started session perform a `find` operation that succeeds.
                await client.test.test.find_one({}, session=session)

        # Assert that the callback was called 2 times (once during the connection handshake, and again during reauthentication).
        self.assertEqual(self.request_called, 2)

    async def test_5_1_azure_with_no_username(self):
        if ENVIRON != "azure":
            raise unittest.SkipTest("Test is only supported on Azure")
        opts = parse_uri(self.uri_single)["options"]
        resource = opts["authMechanismProperties"]["TOKEN_RESOURCE"]

        props = dict(TOKEN_RESOURCE=resource, ENVIRONMENT="azure")
        client = await self.create_client(authMechanismProperties=props)
        await client.test.test.find_one()

    async def test_5_2_azure_with_bad_username(self):
        if ENVIRON != "azure":
            raise unittest.SkipTest("Test is only supported on Azure")

        opts = parse_uri(self.uri_single)["options"]
        token_aud = opts["authMechanismProperties"]["TOKEN_RESOURCE"]

        props = dict(TOKEN_RESOURCE=token_aud, ENVIRONMENT="azure")
        client = await self.create_client(username="bad", authmechanismproperties=props)
        with self.assertRaises(ValueError):
            await client.test.test.find_one()

    async def test_speculative_auth_success(self):
        client1 = await self.create_client()
        await client1.test.test.find_one()
        client2 = await self.create_client()
        await client2.aconnect()

        # Prime the cache of the second client.
        client2.options.pool_options._credentials.cache.data = (
            client1.options.pool_options._credentials.cache.data
        )

        # Set a fail point for saslStart commands.
        async with self.fail_point(
            {
                "mode": {"times": 2},
                "data": {"failCommands": ["saslStart"], "errorCode": 18},
            }
        ):
            # Perform a find operation.
            await client2.test.test.find_one()

    async def test_reauthentication_succeeds_multiple_connections(self):
        client1 = await self.create_client()
        client2 = await self.create_client()

        # Perform an insert operation.
        await client1.test.test.insert_many([{"a": 1}, {"a": 1}])
        await client2.test.test.find_one()
        self.assertEqual(self.request_called, 2)

        # Use the same authenticator for both clients
        # to simulate a race condition with separate connections.
        # We should only see one extra callback despite both connections
        # needing to reauthenticate.
        client2.options.pool_options._credentials.cache.data = (
            client1.options.pool_options._credentials.cache.data
        )

        await client1.test.test.find_one()
        await client2.test.test.find_one()

        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391},
            }
        ):
            await client1.test.test.find_one()

        self.assertEqual(self.request_called, 3)

        async with self.fail_point(
            {
                "mode": {"times": 1},
                "data": {"failCommands": ["find"], "errorCode": 391},
            }
        ):
            await client2.test.test.find_one()

        self.assertEqual(self.request_called, 3)


if __name__ == "__main__":
    unittest.main()
