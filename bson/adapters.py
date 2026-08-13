# Copyright 2026-present MongoDB, Inc.
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

"""Support for decoding BSON replies directly into user document classes.

A class opts into user-defined decoding by setting
``_type_marker = 102`` and providing::

    @classmethod
    def from_bson(cls, doc, codec_options):
        ...

where ``doc`` is one fully decoded document.

Plain dataclasses and pydantic v2 models are automatically handled by the driver in the adapters below.
"""

from __future__ import annotations

import dataclasses
import datetime
import inspect
import sys
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from bson.codec_options import CodecOptions

_BSON_DESERIALIZABLE_MARKER = 102


def _bson_deserializable_class(document_class: Any) -> bool:
    """Return True if `document_class` implements the typed decoding protocol."""
    return getattr(document_class, "_type_marker", None) == _BSON_DESERIALIZABLE_MARKER


def _convert_typed_document(document: Any, codec_options: CodecOptions[Any]) -> Any:
    if isinstance(document, Mapping):
        return document
    adapter = codec_options._document_adapter
    if adapter is None:
        return document
    # Shipped adapters wrap the user type; a raw marker-102 protocol class is
    # its own document type. A protocol class may define a `document_type`
    # attribute for its own purposes, so never sniff the attribute off it.
    document_type = adapter.document_type if isinstance(adapter, _DocumentAdapter) else adapter
    if isinstance(document, document_type):
        if hasattr(adapter, "to_bson"):
            return adapter.to_bson(document, codec_options)
        raise TypeError(
            f"{document_type!r} does not implement to_bson, cannot serialize to a document."
        )
    return document


class _DocumentAdapter:
    """Wraps a user document type in the ``from_bson`` protocol."""

    _type_marker = _BSON_DESERIALIZABLE_MARKER

    def __init__(self, document_type: type[Any]) -> None:
        self.document_type = document_type

    def from_bson(self, doc: dict[str, Any], codec_options: CodecOptions[Any]) -> Any:
        """Construct one ``document_type`` instance from a decoded document."""
        raise NotImplementedError

    def to_bson(self, doc: Any, codec_options: CodecOptions[Any]) -> Any:
        """Convert one ``document_type`` instance into a dictionary document."""
        raise NotImplementedError

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, _DocumentAdapter):
            return type(self) is type(other) and self.document_type == other.document_type
        return NotImplemented

    def __hash__(self) -> int:
        return hash((type(self), self.document_type))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.document_type!r})"


# Exact types _dataclass_to_document returns by reference, mirroring the
# fast path of dataclasses.asdict (which instead deep-copies anything it
# does not recognize). Membership is a speed optimization only: unknown
# types fall through the dispatch below and are likewise shared, not copied.
_ATOMIC_TYPES: set[type] = {
    type(None),
    bool,
    int,
    float,
    str,
    bytes,
    datetime.datetime,
    uuid.UUID,
}

_DATACLASS_FIELD_NAMES: dict[type, tuple[str, ...]] = {}


def _register_bson_atomic_types() -> None:
    """Add the BSON leaf types to ``_ATOMIC_TYPES``.

    Deferred to first adapter construction because objectid and decimal128
    transitively import bson.codec_options, which imports this module.
    """
    from bson.binary import Binary
    from bson.datetime_ms import DatetimeMS
    from bson.decimal128 import Decimal128
    from bson.int64 import Int64
    from bson.objectid import ObjectId
    from bson.regex import Regex
    from bson.timestamp import Timestamp

    _ATOMIC_TYPES.update((Binary, DatetimeMS, Decimal128, Int64, ObjectId, Regex, Timestamp))


def _dataclass_to_document(value: Any) -> Any:
    """Recursively convert nested dataclass instances into dicts.

    Mirrors the dispatch of ``dataclasses.asdict`` except that leaf values
    are returned by reference instead of deep-copied: the BSON encoder only
    reads the result, and copying Binary/ObjectId/datetime values dominates
    asdict's cost. Tuples (including namedtuples) become lists, which encode
    to the same BSON arrays.
    """
    tp = type(value)
    if tp in _ATOMIC_TYPES:
        return value
    if hasattr(tp, "__dataclass_fields__"):
        names = _DATACLASS_FIELD_NAMES.get(tp)
        if names is None:
            names = tuple(f.name for f in dataclasses.fields(value))
            _DATACLASS_FIELD_NAMES[tp] = names
        return {name: _dataclass_to_document(getattr(value, name)) for name in names}
    # Exact container types first for speed; subclasses handled below.
    if tp is list:
        return [_dataclass_to_document(v) for v in value]
    if tp is dict:
        return {k: _dataclass_to_document(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dataclass_to_document(v) for v in value]
    if isinstance(value, dict):
        return {k: _dataclass_to_document(v) for k, v in value.items()}
    return value


class _DataclassAdapter(_DocumentAdapter):
    """Decodes BSON into a plain dataclass via ``cls(**doc)``.

    Extra wire keys raise ``TypeError`` from the constructor and missing
    keys fall through to field defaults.
    """

    def __init__(self, document_type: type[Any]) -> None:
        super().__init__(document_type)
        fields = dataclasses.fields(document_type)
        self._field_names = [f.name for f in fields]
        if "_id" not in self._field_names:
            raise TypeError(f"{self.document_type!r} must define an `_id` field.")
        no_init = [f.name for f in fields if not f.init]
        if no_init:
            raise TypeError(
                f"{self.document_type!r} field(s) {no_init} set init=False: documents "
                "written by the driver would include them, but cls(**doc) cannot pass "
                "them back, so every decode would fail."
            )
        required_init_only = [
            name
            for name, param in inspect.signature(document_type).parameters.items()
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
            and param.default is inspect.Parameter.empty
            and name not in self._field_names
        ]
        if required_init_only:
            raise TypeError(
                f"{self.document_type!r} requires init-only argument(s) {required_init_only} "
                "that are never stored in documents (e.g. an InitVar without a default), "
                "so every decode would fail."
            )
        # Idempotent, and by now bson is fully imported.
        _register_bson_atomic_types()

    def from_bson(self, doc: dict[str, Any], codec_options: CodecOptions[Any]) -> Any:
        return self.document_type(**doc)

    def to_bson(self, doc: Any, codec_options: CodecOptions[Any]) -> Any:
        doc = _dataclass_to_document(doc)
        if "_id" in doc and doc["_id"] is None:
            del doc["_id"]
        return doc


def _pydantic_field_consumes_id(field: Any) -> bool:
    """Return True if a pydantic Field consumes the top-level ``_id`` document key."""
    alias = field.validation_alias if field.validation_alias is not None else field.alias
    if isinstance(alias, str):
        return alias == "_id"
    # AliasChoices has .choices and AliasPath has .path; an AliasPath consumes
    # the document key that is its first element.
    choices = getattr(alias, "choices", None)
    if choices is not None:
        return any(
            choice == "_id" or getattr(choice, "path", [None])[0] == "_id" for choice in choices
        )
    path = getattr(alias, "path", None)
    return bool(path and path[0] == "_id")


class _PydanticAdapter(_DocumentAdapter):
    """Decodes BSON into a pydantic v2 model via ``model_validate``.

    The model must alias the ``_id`` field or use an extra to opt out of strict decoding.
    """

    def __init__(self, document_type: type[Any]) -> None:
        super().__init__(document_type)
        extra = document_type.model_config.get("extra")
        id_fields = [
            field
            for field in getattr(document_type, "model_fields", {}).values()
            if _pydantic_field_consumes_id(field)
        ]
        if not id_fields:
            if extra is None:
                raise TypeError(
                    f"pydantic model {document_type.__name__!r} has no field aliased to '_id' and "
                    "does not set 'extra' in model_config, so decoded documents would silently "
                    "lose their '_id' key. Map the key explicitly, e.g. "
                    "id: ObjectId = Field(alias='_id'), or opt out of strict decoding with "
                    "model_config = ConfigDict(extra='ignore')"
                )
            if extra == "forbid":
                raise TypeError(
                    f"pydantic model {document_type.__name__!r} has no field aliased to '_id' but "
                    "sets extra='forbid', so validation would reject every decoded document "
                    "over its '_id' key. Map the key explicitly, e.g. "
                    "id: ObjectId = Field(alias='_id')"
                )
        # A field that consumes '_id' on decode must also serialize back to
        # '_id' or writes cannot round-trip; see to_bson.
        self._id_round_trips = all(
            (field.serialization_alias if field.serialization_alias is not None else field.alias)
            == "_id"
            for field in id_fields
        )
        self._extra: Optional[str] = None
        if extra is None:
            if "extra" not in inspect.signature(document_type.model_validate).parameters:
                raise TypeError(
                    f"pydantic model {document_type.__name__!r} does not set 'extra' in "
                    "model_config, and this pydantic version cannot enforce strict "
                    "decoding (model_validate gained its 'extra' argument in pydantic "
                    "2.12), so unknown document keys would be silently dropped. Upgrade "
                    "pydantic, or state the decoding policy explicitly, e.g. "
                    "model_config = ConfigDict(extra='ignore')"
                )
            self._extra = "forbid"

    def from_bson(self, doc: dict[str, Any], codec_options: CodecOptions[Any]) -> Any:
        if self._extra is not None:
            return self.document_type.model_validate(doc, extra=self._extra)
        return self.document_type.model_validate(doc)

    def to_bson(self, doc: Any, codec_options: CodecOptions[Any]) -> Any:
        if not self._id_round_trips:
            raise TypeError(
                f"pydantic model {self.document_type.__name__!r} consumes '_id' through a "
                "validation-only alias, so its documents cannot be written back with the "
                "same key; add serialization_alias='_id' or use alias='_id'"
            )
        doc = doc.model_dump(by_alias=True)
        if "_id" in doc and doc["_id"] is None:
            del doc["_id"]
        return doc


def _resolve_document_class(document_class: Any) -> Optional[Any]:
    """Resolve a non-mapping document_class to a typed decoding implementation.

    Returns the argument unchanged if it already implements the protocol, a shipped adapter
    for dataclasses and pydantic v2 models, or ``None`` if unsupported.
    Raises ``TypeError`` for pydantic v1 models.
    """
    if _bson_deserializable_class(document_class):
        if getattr(document_class, "from_bson", None) is None:
            raise TypeError(
                f"{document_class!r} sets _type_marker = {_BSON_DESERIALIZABLE_MARKER} "
                "but does not implement from_bson"
            )
        return document_class
    if isinstance(document_class, type) and dataclasses.is_dataclass(document_class):
        return _DataclassAdapter(document_class)
    # Never import pydantic: if it isn't already in sys.modules the user
    # cannot be passing a pydantic model.
    pydantic = sys.modules.get("pydantic")
    if (
        pydantic is not None
        and isinstance(document_class, type)
        and issubclass(document_class, pydantic.BaseModel)
    ):
        if not hasattr(document_class, "model_validate"):
            raise TypeError(
                "pydantic v1 models are not supported as a document_class, upgrade to pydantic v2"
            )
        return _PydanticAdapter(document_class)
    return None
