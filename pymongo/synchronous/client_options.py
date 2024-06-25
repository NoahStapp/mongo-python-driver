# Copyright 2014-present MongoDB, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License.  You
# may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.

"""Tools to parse mongo client options."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

from bson.codec_options import _parse_codec_options
from pymongo.read_concern import ReadConcern
from pymongo.synchronous import common
from pymongo.synchronous.read_preferences import (
    _ServerMode,
    make_read_preference,
    read_pref_mode_from_name,
)
from pymongo.write_concern import WriteConcern

if TYPE_CHECKING:
    from bson.codec_options import CodecOptions
    from pymongo.synchronous.encryption_options import AutoEncryptionOpts

_IS_SYNC = True


def _parse_read_preference(options: Mapping[str, Any]) -> _ServerMode:
    """Parse read preference options."""
    if "read_preference" in options:
        return options["read_preference"]

    name = options.get("readpreference", "primary")
    mode = read_pref_mode_from_name(name)
    tags = options.get("readpreferencetags")
    max_staleness = options.get("maxstalenessseconds", -1)
    return make_read_preference(mode, tags, max_staleness)


def _parse_write_concern(options: Mapping[str, Any]) -> WriteConcern:
    """Parse write concern options."""
    concern = options.get("w")
    wtimeout = options.get("wtimeoutms")
    j = options.get("journal")
    fsync = options.get("fsync")
    return WriteConcern(concern, wtimeout, j, fsync)


def _parse_read_concern(options: Mapping[str, Any]) -> ReadConcern:
    """Parse read concern options."""
    concern = options.get("readconcernlevel")
    return ReadConcern(concern)


class ClientOptions:
    """Read only configuration options for a MongoClient.

    Should not be instantiated directly by application developers. Access
    a client's options via :attr:`pymongo.mongo_client.MongoClient.options`
    instead.
    """

    def __init__(self, options: Mapping[str, Any]):
        self.__options = options
        self.__codec_options = _parse_codec_options(options)
        self.__read_preference = _parse_read_preference(options)
        self.__write_concern = _parse_write_concern(options)
        self.__read_concern = _parse_read_concern(options)
        self.__connect = options.get("connect")
        self.__retry_writes = options.get("retrywrites", common.RETRY_WRITES)
        self.__retry_reads = options.get("retryreads", common.RETRY_READS)
        self.__auto_encryption_opts = options.get("auto_encryption_opts")
        self.__timeout = options.get("timeoutms")

    @property
    def _options(self) -> Mapping[str, Any]:
        """The original options used to create this ClientOptions."""
        return self.__options

    @property
    def connect(self) -> Optional[bool]:
        """Whether to begin discovering a MongoDB topology automatically."""
        return self.__connect

    @property
    def codec_options(self) -> CodecOptions:
        """A :class:`~bson.codec_options.CodecOptions` instance."""
        return self.__codec_options

    @property
    def read_preference(self) -> _ServerMode:
        """A read preference instance."""
        return self.__read_preference

    @property
    def write_concern(self) -> WriteConcern:
        """A :class:`~pymongo.write_concern.WriteConcern` instance."""
        return self.__write_concern

    @property
    def read_concern(self) -> ReadConcern:
        """A :class:`~pymongo.read_concern.ReadConcern` instance."""
        return self.__read_concern

    @property
    def timeout(self) -> Optional[float]:
        """The configured timeoutMS converted to seconds, or None.

        .. versionadded:: 4.2
        """
        return self.__timeout

    @property
    def retry_writes(self) -> bool:
        """If this instance should retry supported write operations."""
        return self.__retry_writes

    @property
    def retry_reads(self) -> bool:
        """If this instance should retry supported read operations."""
        return self.__retry_reads

    @property
    def auto_encryption_opts(self) -> Optional[AutoEncryptionOpts]:
        """A :class:`~pymongo.encryption.AutoEncryptionOpts` or None."""
        return self.__auto_encryption_opts
