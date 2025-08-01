# Copyright 2010-present MongoDB, Inc.
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

"""Synchronous test suite for pymongo, bson, and gridfs."""
from __future__ import annotations

import asyncio
import gc
import inspect
import logging
import multiprocessing
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import unittest
import warnings
from inspect import iscoroutinefunction

from pymongo.errors import AutoReconnect
from pymongo.synchronous.uri_parser import parse_uri

try:
    import ipaddress

    HAVE_IPADDRESS = True
except ImportError:
    HAVE_IPADDRESS = False
from contextlib import contextmanager
from functools import partial, wraps
from typing import Any, Callable, Dict, Generator, overload
from unittest import SkipTest
from urllib.parse import quote_plus

import pymongo
import pymongo.errors
from bson.son import SON
from pymongo.common import partition_node
from pymongo.hello import HelloCompat
from pymongo.server_api import ServerApi
from pymongo.ssl_support import HAVE_SSL, _ssl  # type:ignore[attr-defined]
from pymongo.synchronous.database import Database
from pymongo.synchronous.mongo_client import MongoClient

sys.path[0:0] = [""]

from test.helpers import (
    COMPRESSORS,
    IS_SRV,
    MONGODB_API_VERSION,
    MULTI_MONGOS_LB_URI,
    TEST_LOADBALANCER,
    TLS_OPTIONS,
    SystemCertsPatcher,
    client_knobs,
    db_pwd,
    db_user,
    global_knobs,
    host,
    is_server_resolvable,
    port,
    print_running_topology,
    print_thread_stacks,
    print_thread_tracebacks,
    sanitize_cmd,
    sanitize_reply,
)
from test.version import Version

_IS_SYNC = True


def _connection_string(h):
    if h.startswith(("mongodb://", "mongodb+srv://")):
        return h
    return f"mongodb://{h!s}"


class ClientContext:
    client: MongoClient

    MULTI_MONGOS_LB_URI = MULTI_MONGOS_LB_URI

    def __init__(self):
        """Create a client and grab essential information from the server."""
        self.connection_attempts = []
        self.connected = False
        self.w = None
        self.nodes = set()
        self.replica_set_name = None
        self.cmd_line = None
        self.server_status = None
        self.version = Version(-1)  # Needs to be comparable with Version
        self.auth_enabled = False
        self.test_commands_enabled = False
        self.server_parameters = {}
        self._hello = None
        self.is_mongos = False
        self.mongoses = []
        self.is_rs = False
        self.has_ipv6 = False
        self.tls = False
        self.tlsCertificateKeyFile = False
        self.server_is_resolvable = is_server_resolvable()
        self.default_client_options: Dict = {}
        self.sessions_enabled = False
        self.client = None  # type: ignore
        self.conn_lock = threading.Lock()
        self.is_data_lake = False
        self.load_balancer = TEST_LOADBALANCER
        self._fips_enabled = None
        if self.load_balancer:
            self.default_client_options["loadBalanced"] = True
        if COMPRESSORS:
            self.default_client_options["compressors"] = COMPRESSORS
        if MONGODB_API_VERSION:
            server_api = ServerApi(MONGODB_API_VERSION)
            self.default_client_options["server_api"] = server_api

    @property
    def client_options(self):
        """Return the MongoClient options for creating a duplicate client."""
        opts = client_context.default_client_options.copy()
        opts["host"] = host
        opts["port"] = port
        if client_context.auth_enabled:
            opts["username"] = db_user
            opts["password"] = db_pwd
        if self.replica_set_name:
            opts["replicaSet"] = self.replica_set_name
        return opts

    @property
    def uri(self):
        """Return the MongoClient URI for creating a duplicate client."""
        opts = client_context.default_client_options.copy()
        opts.pop("server_api", None)  # Cannot be set from the URI
        opts_parts = []
        for opt, val in opts.items():
            strval = str(val)
            if isinstance(val, bool):
                strval = strval.lower()
            opts_parts.append(f"{opt}={quote_plus(strval)}")
        opts_part = "&".join(opts_parts)
        auth_part = ""
        if client_context.auth_enabled:
            auth_part = f"{quote_plus(db_user)}:{quote_plus(db_pwd)}@"
        pair = self.pair
        return f"mongodb://{auth_part}{pair}/?{opts_part}"

    @property
    def hello(self):
        if not self._hello:
            if self.load_balancer:
                self._hello = self.client.admin.command(HelloCompat.CMD)
            else:
                self._hello = self.client.admin.command(HelloCompat.LEGACY_CMD)
        return self._hello

    def _connect(self, host, port, **kwargs):
        kwargs.update(self.default_client_options)
        client: MongoClient = pymongo.MongoClient(
            host, port, serverSelectionTimeoutMS=5000, **kwargs
        )
        try:
            try:
                client.admin.command("ping")  # Can we connect?
            except pymongo.errors.OperationFailure as exc:
                # SERVER-32063
                self.connection_attempts.append(
                    f"connected client {client!r}, but legacy hello failed: {exc}"
                )
            else:
                self.connection_attempts.append(f"successfully connected client {client!r}")
            # If connected, then return client with default timeout
            return pymongo.MongoClient(host, port, **kwargs)
        except pymongo.errors.ConnectionFailure as exc:
            self.connection_attempts.append(f"failed to connect client {client!r}: {exc}")
            return None
        finally:
            client.close()

    def _init_client(self):
        self.mongoses = []
        self.connection_attempts = []
        self.client = self._connect(host, port)
        if self.client is not None:
            # Return early when connected to dataLake as mongohoused does not
            # support the getCmdLineOpts command and is tested without TLS.
            if os.environ.get("TEST_DATA_LAKE"):
                self.is_data_lake = True
                self.auth_enabled = True
                self.client.close()
                self.client = self._connect(host, port, username=db_user, password=db_pwd)
                self.connected = True
                return

        if HAVE_SSL and not self.client:
            # Is MongoDB configured for SSL?
            self.client = self._connect(host, port, **TLS_OPTIONS)
            if self.client:
                self.tls = True
                self.default_client_options.update(TLS_OPTIONS)
                self.tlsCertificateKeyFile = True

        if self.client:
            self.connected = True

            try:
                self.cmd_line = self.client.admin.command("getCmdLineOpts")
            except pymongo.errors.OperationFailure as e:
                assert e.details is not None
                msg = e.details.get("errmsg", "")
                if e.code == 13 or "unauthorized" in msg or "login" in msg:
                    # Unauthorized.
                    self.auth_enabled = True
                else:
                    raise
            else:
                self.auth_enabled = self._server_started_with_auth()

            if self.auth_enabled:
                if not IS_SRV:
                    # See if db_user already exists.
                    if not self._check_user_provided():
                        _create_user(self.client.admin, db_user, db_pwd)

                if self.client:
                    self.client.close()

                self.client = self._connect(
                    host,
                    port,
                    username=db_user,
                    password=db_pwd,
                    replicaSet=self.replica_set_name,
                    **self.default_client_options,
                )

                # May not have this if OperationFailure was raised earlier.
                self.cmd_line = self.client.admin.command("getCmdLineOpts")

            self.server_status = self.client.admin.command("serverStatus")
            if self.storage_engine == "mmapv1":
                # MMAPv1 does not support retryWrites=True.
                self.default_client_options["retryWrites"] = False

            hello = self.hello
            self.sessions_enabled = "logicalSessionTimeoutMinutes" in hello

            if "setName" in hello:
                self.replica_set_name = str(hello["setName"])
                self.is_rs = True
                if self.client:
                    self.client.close()
                if self.auth_enabled:
                    # It doesn't matter which member we use as the seed here.
                    self.client = pymongo.MongoClient(
                        host,
                        port,
                        username=db_user,
                        password=db_pwd,
                        replicaSet=self.replica_set_name,
                        **self.default_client_options,
                    )
                else:
                    self.client = pymongo.MongoClient(
                        host, port, replicaSet=self.replica_set_name, **self.default_client_options
                    )

                # Get the authoritative hello result from the primary.
                self._hello = None
                hello = self.hello
                nodes = [partition_node(node.lower()) for node in hello.get("hosts", [])]
                nodes.extend([partition_node(node.lower()) for node in hello.get("passives", [])])
                nodes.extend([partition_node(node.lower()) for node in hello.get("arbiters", [])])
                self.nodes = set(nodes)
            else:
                self.nodes = {(host, port)}
            self.w = len(hello.get("hosts", [])) or 1
            self.version = Version.from_client(self.client)

            self.server_parameters = self.client.admin.command("getParameter", "*")
            assert self.cmd_line is not None
            if self.server_parameters["enableTestCommands"]:
                self.test_commands_enabled = True
            elif "parsed" in self.cmd_line:
                params = self.cmd_line["parsed"].get("setParameter", [])
                if "enableTestCommands=1" in params:
                    self.test_commands_enabled = True
                else:
                    params = self.cmd_line["parsed"].get("setParameter", {})
                    if params.get("enableTestCommands") == "1":
                        self.test_commands_enabled = True
            self.has_ipv6 = self._server_started_with_ipv6()

            self.is_mongos = (self.hello).get("msg") == "isdbgrid"
            if self.is_mongos:
                address = self.client.address
                self.mongoses.append(address)
                # Check for another mongos on the next port.
                assert address is not None
                next_address = address[0], address[1] + 1
                mongos_client = self._connect(*next_address, **self.default_client_options)
                if mongos_client:
                    hello = mongos_client.admin.command(HelloCompat.LEGACY_CMD)
                    if hello.get("msg") == "isdbgrid":
                        self.mongoses.append(next_address)
                    mongos_client.close()

    def init(self):
        with self.conn_lock:
            if not self.client and not self.connection_attempts:
                self._init_client()

    def connection_attempt_info(self):
        return "\n".join(self.connection_attempts)

    @property
    def host(self):
        if self.is_rs and not IS_SRV:
            primary = self.client.primary
            return str(primary[0]) if primary is not None else host
        return host

    @property
    def port(self):
        if self.is_rs and not IS_SRV:
            primary = self.client.primary
            return primary[1] if primary is not None else port
        return port

    @property
    def pair(self):
        return "%s:%d" % (self.host, self.port)

    @property
    def has_secondaries(self):
        if not self.client:
            return False
        return bool(len(self.client.secondaries))

    @property
    def storage_engine(self):
        try:
            return self.server_status.get("storageEngine", {}).get(  # type:ignore[union-attr]
                "name"
            )
        except AttributeError:
            # Raised if self.server_status is None.
            return None

    @property
    def fips_enabled(self):
        if self._fips_enabled is not None:
            return self._fips_enabled
        try:
            subprocess.run(["fips-mode-setup", "--is-enabled"], check=True)
            self._fips_enabled = True
        except (subprocess.SubprocessError, FileNotFoundError):
            self._fips_enabled = False
        if os.environ.get("REQUIRE_FIPS") and not self._fips_enabled:
            raise RuntimeError("Expected FIPS to be enabled")
        return self._fips_enabled

    def check_auth_type(self, auth_type):
        auth_mechs = self.server_parameters.get("authenticationMechanisms", [])
        return auth_type in auth_mechs

    def _check_user_provided(self):
        """Return True if db_user/db_password is already an admin user."""
        client: MongoClient = pymongo.MongoClient(
            host,
            port,
            username=db_user,
            password=db_pwd,
            **self.default_client_options,
        )

        try:
            return db_user in _all_users(client.admin)
        except pymongo.errors.OperationFailure as e:
            assert e.details is not None
            msg = e.details.get("errmsg", "")
            if e.code == 18 or "auth fails" in msg:
                # Auth failed.
                return False
            else:
                raise
        finally:
            client.close()

    def _server_started_with_auth(self):
        # MongoDB >= 2.0
        assert self.cmd_line is not None
        if "parsed" in self.cmd_line:
            parsed = self.cmd_line["parsed"]
            # MongoDB >= 2.6
            if "security" in parsed:
                security = parsed["security"]
                # >= rc3
                if "authorization" in security:
                    return security["authorization"] == "enabled"
                # < rc3
                return security.get("auth", False) or bool(security.get("keyFile"))
            return parsed.get("auth", False) or bool(parsed.get("keyFile"))
        # Legacy
        argv = self.cmd_line["argv"]
        return "--auth" in argv or "--keyFile" in argv

    def _server_started_with_ipv6(self):
        if not socket.has_ipv6:
            return False

        assert self.cmd_line is not None
        if "parsed" in self.cmd_line:
            if not self.cmd_line["parsed"].get("net", {}).get("ipv6"):
                return False
        else:
            if "--ipv6" not in self.cmd_line["argv"]:
                return False

        # The server was started with --ipv6. Is there an IPv6 route to it?
        try:
            for info in socket.getaddrinfo(self.host, self.port):
                if info[0] == socket.AF_INET6:
                    return True
        except OSError:
            pass

        return False

    def _require(self, condition, msg, func=None):
        def make_wrapper(f):
            if iscoroutinefunction(f):
                wraps_async = True
            else:
                wraps_async = False

            @wraps(f)
            def wrap(*args, **kwargs):
                self.init()
                # Always raise SkipTest if we can't connect to MongoDB
                if not self.connected:
                    pair = self.pair
                    raise SkipTest(f"Cannot connect to MongoDB on {pair}")
                if iscoroutinefunction(condition):
                    if condition():
                        if wraps_async:
                            return f(*args, **kwargs)
                        else:
                            return f(*args, **kwargs)
                elif condition():
                    if wraps_async:
                        return f(*args, **kwargs)
                    else:
                        return f(*args, **kwargs)
                if "self.pair" in msg:
                    new_msg = msg.replace("self.pair", self.pair)
                else:
                    new_msg = msg
                raise SkipTest(new_msg)

            return wrap

        if func is None:

            def decorate(f):
                return make_wrapper(f)

            return decorate
        return make_wrapper(func)

    def create_user(self, dbname, user, pwd=None, roles=None, **kwargs):
        kwargs["writeConcern"] = {"w": self.w}
        return _create_user(self.client[dbname], user, pwd, roles, **kwargs)

    def drop_user(self, dbname, user):
        self.client[dbname].command("dropUser", user, writeConcern={"w": self.w})

    def require_connection(self, func):
        """Run a test only if we can connect to MongoDB."""
        return self._require(
            lambda: True,  # _require checks if we're connected
            "Cannot connect to MongoDB on self.pair",
            func=func,
        )

    def require_data_lake(self, func):
        """Run a test only if we are connected to Atlas Data Lake."""
        return self._require(
            lambda: self.is_data_lake,
            "Not connected to Atlas Data Lake on self.pair",
            func=func,
        )

    def require_version_min(self, *ver):
        """Run a test only if the server version is at least ``version``."""
        other_version = Version(*ver)
        return self._require(
            lambda: self.version >= other_version,
            "Server version must be at least %s" % str(other_version),
        )

    def require_version_max(self, *ver):
        """Run a test only if the server version is at most ``version``."""
        other_version = Version(*ver)
        return self._require(
            lambda: self.version <= other_version,
            "Server version must be at most %s" % str(other_version),
        )

    def require_auth(self, func):
        """Run a test only if the server is running with auth enabled."""
        return self._require(
            lambda: self.auth_enabled, "Authentication is not enabled on the server", func=func
        )

    def require_no_auth(self, func):
        """Run a test only if the server is running without auth enabled."""
        return self._require(
            lambda: not self.auth_enabled,
            "Authentication must not be enabled on the server",
            func=func,
        )

    def require_no_fips(self, func):
        """Run a test only if the host does not have FIPS enabled."""
        return self._require(
            lambda: not self.fips_enabled, "Test cannot run on a FIPS-enabled host", func=func
        )

    def require_replica_set(self, func):
        """Run a test only if the client is connected to a replica set."""
        return self._require(lambda: self.is_rs, "Not connected to a replica set", func=func)

    def require_secondaries_count(self, count):
        """Run a test only if the client is connected to a replica set that has
        `count` secondaries.
        """

        def sec_count():
            return 0 if not self.client else len(self.client.secondaries)

        def check():
            return sec_count() >= count

        return self._require(check, "Not enough secondaries available")

    @property
    def supports_secondary_read_pref(self):
        if self.has_secondaries:
            return True
        if self.is_mongos:
            shard = (self.client.config.shards.find_one())["host"]  # type:ignore[index]
            num_members = shard.count(",") + 1
            return num_members > 1
        return False

    def require_secondary_read_pref(self):
        """Run a test only if the client is connected to a cluster that
        supports secondary read preference
        """
        return self._require(
            lambda: self.supports_secondary_read_pref,
            "This cluster does not support secondary read preference",
        )

    def require_no_replica_set(self, func):
        """Run a test if the client is *not* connected to a replica set."""
        return self._require(
            lambda: not self.is_rs, "Connected to a replica set, not a standalone mongod", func=func
        )

    def require_ipv6(self, func):
        """Run a test only if the client can connect to a server via IPv6."""
        return self._require(lambda: self.has_ipv6, "No IPv6", func=func)

    def require_no_mongos(self, func):
        """Run a test only if the client is not connected to a mongos."""
        return self._require(
            lambda: not self.is_mongos, "Must be connected to a mongod, not a mongos", func=func
        )

    def require_mongos(self, func):
        """Run a test only if the client is connected to a mongos."""
        return self._require(lambda: self.is_mongos, "Must be connected to a mongos", func=func)

    def require_multiple_mongoses(self, func):
        """Run a test only if the client is connected to a sharded cluster
        that has 2 mongos nodes.
        """
        return self._require(
            lambda: len(self.mongoses) > 1, "Must have multiple mongoses available", func=func
        )

    def require_standalone(self, func):
        """Run a test only if the client is connected to a standalone."""
        return self._require(
            lambda: not (self.is_mongos or self.is_rs),
            "Must be connected to a standalone",
            func=func,
        )

    def require_no_standalone(self, func):
        """Run a test only if the client is not connected to a standalone."""
        return self._require(
            lambda: self.is_mongos or self.is_rs,
            "Must be connected to a replica set or mongos",
            func=func,
        )

    def require_load_balancer(self, func):
        """Run a test only if the client is connected to a load balancer."""
        return self._require(
            lambda: self.load_balancer, "Must be connected to a load balancer", func=func
        )

    def require_no_load_balancer(self, func):
        """Run a test only if the client is not connected to a load balancer."""
        return self._require(
            lambda: not self.load_balancer, "Must not be connected to a load balancer", func=func
        )

    def require_change_streams(self, func):
        """Run a test only if the server supports change streams."""
        return self.require_no_standalone(func)

    def is_topology_type(self, topologies):
        unknown = set(topologies) - {
            "single",
            "replicaset",
            "sharded",
            "load-balanced",
        }
        if unknown:
            raise AssertionError(f"Unknown topologies: {unknown!r}")
        if self.load_balancer:
            if "load-balanced" in topologies:
                return True
            return False
        if "single" in topologies and not (self.is_mongos or self.is_rs):
            return True
        if "replicaset" in topologies and self.is_rs:
            return True
        if "sharded" in topologies and self.is_mongos:
            return True
        return False

    def require_cluster_type(self, topologies=None):
        """Run a test only if the client is connected to a cluster that
        conforms to one of the specified topologies. Acceptable topologies
        are 'single', 'replicaset', and 'sharded'.
        """
        topologies = topologies or []

        def _is_valid_topology():
            return self.is_topology_type(topologies)

        return self._require(_is_valid_topology, "Cluster type not in %s" % (topologies))

    def require_test_commands(self, func):
        """Run a test only if the server has test commands enabled."""
        return self._require(
            lambda: self.test_commands_enabled, "Test commands must be enabled", func=func
        )

    def require_failCommand_fail_point(self, func):
        """Run a test only if the server supports the failCommand fail
        point.
        """
        return self._require(
            lambda: self.supports_failCommand_fail_point,
            "failCommand fail point must be supported",
            func=func,
        )

    def require_failCommand_appName(self, func):
        """Run a test only if the server supports the failCommand appName."""
        # SERVER-47195 and SERVER-49336.
        return self._require(
            lambda: (self.test_commands_enabled and self.version >= (4, 4, 7)),
            "failCommand appName must be supported",
            func=func,
        )

    def require_failCommand_blockConnection(self, func):
        """Run a test only if the server supports failCommand blockConnection."""
        return self._require(
            lambda: (
                self.test_commands_enabled
                and (
                    (not self.is_mongos and self.version >= (4, 2, 9))
                    or (self.is_mongos and self.version >= (4, 4))
                )
            ),
            "failCommand blockConnection is not supported",
            func=func,
        )

    def require_tls(self, func):
        """Run a test only if the client can connect over TLS."""
        return self._require(lambda: self.tls, "Must be able to connect via TLS", func=func)

    def require_no_tls(self, func):
        """Run a test only if the client can connect over TLS."""
        return self._require(lambda: not self.tls, "Must be able to connect without TLS", func=func)

    def require_tlsCertificateKeyFile(self, func):
        """Run a test only if the client can connect with tlsCertificateKeyFile."""
        return self._require(
            lambda: self.tlsCertificateKeyFile,
            "Must be able to connect with tlsCertificateKeyFile",
            func=func,
        )

    def require_server_resolvable(self, func):
        """Run a test only if the hostname 'server' is resolvable."""
        return self._require(
            lambda: self.server_is_resolvable,
            "No hosts entry for 'server'. Cannot validate hostname in the certificate",
            func=func,
        )

    def require_sessions(self, func):
        """Run a test only if the deployment supports sessions."""
        return self._require(lambda: self.sessions_enabled, "Sessions not supported", func=func)

    def supports_retryable_writes(self):
        if not self.sessions_enabled:
            return False
        return self.is_mongos or self.is_rs

    def require_retryable_writes(self, func):
        """Run a test only if the deployment supports retryable writes."""
        return self._require(
            self.supports_retryable_writes,
            "This server does not support retryable writes",
            func=func,
        )

    def supports_transactions(self):
        if self.version.at_least(4, 1, 8):
            return self.is_mongos or self.is_rs

        if self.version.at_least(4, 0):
            return self.is_rs

        return False

    def require_transactions(self, func):
        """Run a test only if the deployment might support transactions.

        *Might* because this does not test the storage engine or FCV.
        """
        return self._require(
            self.supports_transactions, "Transactions are not supported", func=func
        )

    def require_no_api_version(self, func):
        """Skip this test when testing with requireApiVersion."""
        return self._require(
            lambda: not MONGODB_API_VERSION,
            "This test does not work with requireApiVersion",
            func=func,
        )

    def require_sync(self, func):
        """Run a test only if using the synchronous API."""
        return self._require(
            lambda: _IS_SYNC, "This test only works with the synchronous API", func=func
        )

    def require_async(self, func):
        """Run a test only if using the asynchronous API."""  # unasync: off
        return self._require(
            lambda: not _IS_SYNC,
            "This test only works with the asynchronous API",  # unasync: off
            func=func,
        )

    def mongos_seeds(self):
        return ",".join("{}:{}".format(*address) for address in self.mongoses)

    @property
    def supports_failCommand_fail_point(self):
        """Does the server support the failCommand fail point?"""
        if self.is_mongos:
            return self.version.at_least(4, 1, 5) and self.test_commands_enabled
        else:
            return self.version.at_least(4, 0) and self.test_commands_enabled

    @property
    def requires_hint_with_min_max_queries(self):
        """Does the server require a hint with min/max queries."""
        # Changed in SERVER-39567.
        return self.version.at_least(4, 1, 10)

    @property
    def max_bson_size(self):
        return (self.hello)["maxBsonObjectSize"]

    @property
    def max_write_batch_size(self):
        return (self.hello)["maxWriteBatchSize"]

    @property
    def max_message_size_bytes(self):
        return (self.hello)["maxMessageSizeBytes"]


# Reusable client context
client_context = ClientContext()

# Global event loop for async tests.
LOOP = None


def get_loop() -> asyncio.AbstractEventLoop:
    """Get the test suite's global event loop."""
    global LOOP
    if LOOP is None:
        try:
            LOOP = asyncio.get_running_loop()
        except RuntimeError:
            # no running event loop, fallback to get_event_loop.
            try:
                # Ignore DeprecationWarning: There is no current event loop
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    LOOP = asyncio.get_event_loop()
            except RuntimeError:
                LOOP = asyncio.new_event_loop()
                asyncio.set_event_loop(LOOP)
    return LOOP


class PyMongoTestCase(unittest.TestCase):
    if not _IS_SYNC:
        # An async TestCase that uses a single event loop for all tests.
        # Inspired by TestCase.
        def setUp(self):
            pass

        def tearDown(self):
            pass

        def addCleanup(self, func, /, *args, **kwargs):
            self.addCleanup(*(func, *args), **kwargs)

        def _callSetUp(self):
            self.setUp()
            self._callAsync(self.setUp)

        def _callTestMethod(self, method):
            self._callMaybeAsync(method)

        def _callTearDown(self):
            self._callAsync(self.tearDown)
            self.tearDown()

        def _callCleanup(self, function, *args, **kwargs):
            self._callMaybeAsync(function, *args, **kwargs)

        def _callAsync(self, func, /, *args, **kwargs):
            assert inspect.iscoroutinefunction(func), f"{func!r} is not an async function"
            return get_loop().run_until_complete(func(*args, **kwargs))

        def _callMaybeAsync(self, func, /, *args, **kwargs):
            if inspect.iscoroutinefunction(func):
                return get_loop().run_until_complete(func(*args, **kwargs))
            else:
                return func(*args, **kwargs)

    def assertEqualCommand(self, expected, actual, msg=None):
        self.assertEqual(sanitize_cmd(expected), sanitize_cmd(actual), msg)

    def assertEqualReply(self, expected, actual, msg=None):
        self.assertEqual(sanitize_reply(expected), sanitize_reply(actual), msg)

    @staticmethod
    def configure_fail_point(client, command_args, off=False):
        cmd = {"configureFailPoint": "failCommand"}
        cmd.update(command_args)
        if off:
            cmd["mode"] = "off"
            cmd.pop("data", None)
        client.admin.command(cmd)

    @contextmanager
    def fail_point(self, command_args):
        self.configure_fail_point(client_context.client, command_args)
        try:
            yield
        finally:
            self.configure_fail_point(client_context.client, command_args, off=True)

    @contextmanager
    def fork(
        self, target: Callable, timeout: float = 60
    ) -> Generator[multiprocessing.Process, None, None]:
        """Helper for tests that use os.fork()

        Use in a with statement:

            with self.fork(target=lambda: print('in child')) as proc:
                self.assertTrue(proc.pid)  # Child process was started
        """

        def _print_threads(*args: object) -> None:
            if _print_threads.called:  # type:ignore[attr-defined]
                return
            _print_threads.called = True  # type:ignore[attr-defined]
            print_thread_tracebacks()

        _print_threads.called = False  # type:ignore[attr-defined]

        def _target() -> None:
            signal.signal(signal.SIGUSR1, _print_threads)
            try:
                target()
            except Exception as exc:
                sys.stderr.write(f"Child process failed with: {exc}\n")
                _print_threads()
                # Sleep for a while to let the parent attach via GDB.
                time.sleep(2 * timeout)
                raise

        ctx = multiprocessing.get_context("fork")
        proc = ctx.Process(target=_target)
        proc.start()
        try:
            yield proc  # type: ignore
        finally:
            proc.join(timeout)
            pid = proc.pid
            assert pid
            if proc.exitcode is None:
                # gdb to get C-level tracebacks
                print_thread_stacks(pid)
                # If it failed, SIGUSR1 to get thread tracebacks.
                os.kill(pid, signal.SIGUSR1)
                proc.join(5)
                if proc.exitcode is None:
                    # SIGINT to get main thread traceback in case SIGUSR1 didn't work.
                    os.kill(pid, signal.SIGINT)
                    proc.join(5)
                if proc.exitcode is None:
                    # SIGKILL in case SIGINT didn't work.
                    proc.kill()
                    proc.join(1)
                self.fail(f"child timed out after {timeout}s (see traceback in logs): deadlock?")
            self.assertEqual(proc.exitcode, 0)

    @classmethod
    def _unmanaged_async_mongo_client(
        cls, host, port, authenticate=True, directConnection=None, **kwargs
    ):
        """Create a new client over SSL/TLS if necessary."""
        host = host or client_context.host
        port = port or client_context.port
        client_options: dict = client_context.default_client_options.copy()
        if client_context.replica_set_name and not directConnection:
            client_options["replicaSet"] = client_context.replica_set_name
        if directConnection is not None:
            client_options["directConnection"] = directConnection
        client_options.update(kwargs)

        uri = _connection_string(host)
        auth_mech = kwargs.get("authMechanism", "")
        if client_context.auth_enabled and authenticate and auth_mech != "MONGODB-OIDC":
            # Only add the default username or password if one is not provided.
            res = parse_uri(uri)
            if (
                not res["username"]
                and not res["password"]
                and "username" not in client_options
                and "password" not in client_options
            ):
                client_options["username"] = db_user
                client_options["password"] = db_pwd
        client = MongoClient(uri, port, **client_options)
        if client._options.connect:
            client._connect()
        return client

    def _async_mongo_client(self, host, port, authenticate=True, directConnection=None, **kwargs):
        """Create a new client over SSL/TLS if necessary."""
        host = host or client_context.host
        port = port or client_context.port
        client_options: dict = client_context.default_client_options.copy()
        if client_context.replica_set_name and not directConnection:
            client_options["replicaSet"] = client_context.replica_set_name
        if directConnection is not None:
            client_options["directConnection"] = directConnection
        client_options.update(kwargs)

        uri = _connection_string(host)
        auth_mech = kwargs.get("authMechanism", "")
        if client_context.auth_enabled and authenticate and auth_mech != "MONGODB-OIDC":
            # Only add the default username or password if one is not provided.
            res = parse_uri(uri)
            if (
                not res["username"]
                and not res["password"]
                and "username" not in client_options
                and "password" not in client_options
            ):
                client_options["username"] = db_user
                client_options["password"] = db_pwd
        client = MongoClient(uri, port, **client_options)
        if client._options.connect:
            client._connect()
        self.addCleanup(client.close)
        return client

    @classmethod
    def unmanaged_single_client_noauth(
        cls, h: Any = None, p: Any = None, **kwargs: Any
    ) -> MongoClient[dict]:
        """Make a direct connection. Don't authenticate."""
        return cls._unmanaged_async_mongo_client(
            h, p, authenticate=False, directConnection=True, **kwargs
        )

    @classmethod
    def unmanaged_single_client(
        cls, h: Any = None, p: Any = None, **kwargs: Any
    ) -> MongoClient[dict]:
        """Make a direct connection. Don't authenticate."""
        return cls._unmanaged_async_mongo_client(h, p, directConnection=True, **kwargs)

    @classmethod
    def unmanaged_rs_client(cls, h: Any = None, p: Any = None, **kwargs: Any) -> MongoClient[dict]:
        """Connect to the replica set and authenticate if necessary."""
        return cls._unmanaged_async_mongo_client(h, p, **kwargs)

    @classmethod
    def unmanaged_rs_client_noauth(
        cls, h: Any = None, p: Any = None, **kwargs: Any
    ) -> MongoClient[dict]:
        """Make a direct connection. Don't authenticate."""
        return cls._unmanaged_async_mongo_client(h, p, authenticate=False, **kwargs)

    @classmethod
    def unmanaged_rs_or_single_client_noauth(
        cls, h: Any = None, p: Any = None, **kwargs: Any
    ) -> MongoClient[dict]:
        """Make a direct connection. Don't authenticate."""
        return cls._unmanaged_async_mongo_client(h, p, authenticate=False, **kwargs)

    @classmethod
    def unmanaged_rs_or_single_client(
        cls, h: Any = None, p: Any = None, **kwargs: Any
    ) -> MongoClient[dict]:
        """Make a direct connection. Don't authenticate."""
        return cls._unmanaged_async_mongo_client(h, p, **kwargs)

    def single_client_noauth(
        self, h: Any = None, p: Any = None, **kwargs: Any
    ) -> MongoClient[dict]:
        """Make a direct connection. Don't authenticate."""
        return self._async_mongo_client(h, p, authenticate=False, directConnection=True, **kwargs)

    def single_client(self, h: Any = None, p: Any = None, **kwargs: Any) -> MongoClient[dict]:
        """Make a direct connection, and authenticate if necessary."""
        return self._async_mongo_client(h, p, directConnection=True, **kwargs)

    def rs_client_noauth(self, h: Any = None, p: Any = None, **kwargs: Any) -> MongoClient[dict]:
        """Connect to the replica set. Don't authenticate."""
        return self._async_mongo_client(h, p, authenticate=False, **kwargs)

    def rs_client(self, h: Any = None, p: Any = None, **kwargs: Any) -> MongoClient[dict]:
        """Connect to the replica set and authenticate if necessary."""
        return self._async_mongo_client(h, p, **kwargs)

    def rs_or_single_client_noauth(
        self, h: Any = None, p: Any = None, **kwargs: Any
    ) -> MongoClient[dict]:
        """Connect to the replica set if there is one, otherwise the standalone.

        Like rs_or_single_client, but does not authenticate.
        """
        return self._async_mongo_client(h, p, authenticate=False, **kwargs)

    def rs_or_single_client(self, h: Any = None, p: Any = None, **kwargs: Any) -> MongoClient[Any]:
        """Connect to the replica set if there is one, otherwise the standalone.

        Authenticates if necessary.
        """
        return self._async_mongo_client(h, p, **kwargs)

    def simple_client(self, h: Any = None, p: Any = None, **kwargs: Any) -> MongoClient:
        if not h and not p:
            client = MongoClient(**kwargs)
        else:
            client = MongoClient(h, p, **kwargs)
        self.addCleanup(client.close)
        return client

    @classmethod
    def unmanaged_simple_client(cls, h: Any = None, p: Any = None, **kwargs: Any) -> MongoClient:
        if not h and not p:
            client = MongoClient(**kwargs)
        else:
            client = MongoClient(h, p, **kwargs)
        return client

    def disable_replication(self, client):
        """Disable replication on all secondaries."""
        for h, p in client.secondaries:
            secondary = self.single_client(h, p)
            secondary.admin.command("configureFailPoint", "stopReplProducer", mode="alwaysOn")

    def enable_replication(self, client):
        """Enable replication on all secondaries."""
        for h, p in client.secondaries:
            secondary = self.single_client(h, p)
            secondary.admin.command("configureFailPoint", "stopReplProducer", mode="off")


class UnitTest(PyMongoTestCase):
    """Async base class for TestCases that don't require a connection to MongoDB."""

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass


class IntegrationTest(PyMongoTestCase):
    """Async base class for TestCases that need a connection to MongoDB to pass."""

    client: MongoClient[dict]
    db: Database
    credentials: Dict[str, str]

    @client_context.require_connection
    def setUp(self) -> None:
        if client_context.load_balancer and not getattr(self, "RUN_ON_LOAD_BALANCER", False):
            raise SkipTest("this test does not support load balancers")
        self.client = client_context.client
        self.db = self.client.pymongo_test
        if client_context.auth_enabled:
            self.credentials = {"username": db_user, "password": db_pwd}
        else:
            self.credentials = {}

    def cleanup_colls(self, *collections):
        """Cleanup collections faster than drop_collection."""
        for c in collections:
            c = self.client[c.database.name][c.name]
            c.delete_many({})
            c.drop_indexes()

    def patch_system_certs(self, ca_certs):
        patcher = SystemCertsPatcher(ca_certs)
        self.addCleanup(patcher.disable)


class MockClientTest(UnitTest):
    """Base class for TestCases that use MockClient.

    This class is *not* an IntegrationTest: if properly written, MockClient
    tests do not require a running server.

    The class temporarily overrides HEARTBEAT_FREQUENCY to speed up tests.
    """

    # MockClients tests that use replicaSet, directConnection=True, pass
    # multiple seed addresses, or wait for heartbeat events are incompatible
    # with loadBalanced=True.
    @client_context.require_no_load_balancer
    def setUp(self) -> None:
        super().setUp()

        self.client_knobs = client_knobs(heartbeat_frequency=0.001, min_heartbeat_interval=0.001)
        self.client_knobs.enable()

    def tearDown(self) -> None:
        self.client_knobs.disable()
        super().tearDown()


def setup():
    if not _IS_SYNC:
        # Set up the event loop.
        get_loop()
    client_context.init()
    warnings.resetwarnings()
    warnings.simplefilter("always")
    global_knobs.enable()


def teardown():
    global_knobs.disable()
    garbage = []
    for g in gc.garbage:
        garbage.append(f"GARBAGE: {g!r}")
        garbage.append(f"  gc.get_referents: {gc.get_referents(g)!r}")
        garbage.append(f"  gc.get_referrers: {gc.get_referrers(g)!r}")
    if garbage:
        raise AssertionError("\n".join(garbage))
    c = client_context.client
    if c:
        if not client_context.is_data_lake:
            try:
                c.drop_database("pymongo-pooling-tests")
                c.drop_database("pymongo_test")
                c.drop_database("pymongo_test1")
                c.drop_database("pymongo_test2")
                c.drop_database("pymongo_test_mike")
                c.drop_database("pymongo_test_bernie")
            except AutoReconnect:
                # PYTHON-4982
                if sys.implementation.name.lower() != "pypy":
                    raise
        c.close()
    print_running_clients()


def test_cases(suite):
    """Iterator over all TestCases within a TestSuite."""
    for suite_or_case in suite._tests:
        if isinstance(suite_or_case, unittest.TestCase):
            # unittest.TestCase
            yield suite_or_case
        else:
            # unittest.TestSuite
            yield from test_cases(suite_or_case)


def print_running_clients():
    from pymongo.synchronous.topology import Topology

    processed = set()
    # Avoid false positives on the main test client.
    # XXX: Can be removed after PYTHON-1634 or PYTHON-1896.
    c = client_context.client
    if c:
        processed.add(c._topology._topology_id)
    # Call collect to manually cleanup any would-be gc'd clients to avoid
    # false positives.
    gc.collect()
    for obj in gc.get_objects():
        try:
            if isinstance(obj, Topology):
                # Avoid printing the same Topology multiple times.
                if obj._topology_id in processed:
                    continue
                print_running_topology(obj)
                processed.add(obj._topology_id)
        except ReferenceError:
            pass


def _all_users(db):
    return {u["user"] for u in (db.command("usersInfo")).get("users", [])}


def _create_user(authdb, user, pwd=None, roles=None, **kwargs):
    cmd = SON([("createUser", user)])
    # X509 doesn't use a password
    if pwd:
        cmd["pwd"] = pwd
    cmd["roles"] = roles or ["root"]
    cmd.update(**kwargs)
    return authdb.command(cmd)


def connected(client):
    """Convenience to wait for a newly-constructed client to connect."""
    with warnings.catch_warnings():
        # Ignore warning that ping is always routed to primary even
        # if client's read preference isn't PRIMARY.
        warnings.simplefilter("ignore", UserWarning)
        client.admin.command("ping")  # Force connection.

    return client


def drop_collections(db: Database):
    # Drop all non-system collections in this database.
    for coll in db.list_collection_names(filter={"name": {"$regex": r"^(?!system\.)"}}):
        db.drop_collection(coll)


def remove_all_users(db: Database):
    db.command("dropAllUsersFromDatabase", 1, writeConcern={"w": client_context.w})
