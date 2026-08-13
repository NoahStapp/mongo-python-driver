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
import inspect
import sys
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
    if adapter is not None and isinstance(document, getattr(adapter, "document_type", adapter)):
        if hasattr(adapter, "to_bson"):
            return adapter.to_bson(document, codec_options)
        else:
            raise TypeError(
                f"{getattr(adapter, 'document_type', adapter)!r} does not implement to_bson, cannot serialize to a document."
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


class _DataclassAdapter(_DocumentAdapter):
    """Decodes BSON into a plain dataclass via ``cls(**doc)``.

    Extra wire keys raise ``TypeError`` from the constructor and missing
    keys fall through to field defaults.
    """

    def __init__(self, document_type: type[Any]) -> None:
        super().__init__(document_type)
        self._field_names = [f.name for f in dataclasses.fields(document_type)]
        if "_id" not in self._field_names:
            raise TypeError(f"{self.document_type!r} must define an `_id` field.")

    def from_bson(self, doc: dict[str, Any], codec_options: CodecOptions[Any]) -> Any:
        return self.document_type(**doc)

    def to_bson(self, doc: Any, codec_options: CodecOptions[Any]) -> Any:
        doc = dataclasses.asdict(doc)
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
        if extra is None and not any(
            _pydantic_field_consumes_id(field) for field in document_type.model_fields.values()
        ):
            raise TypeError(
                f"pydantic model {document_type.__name__!r} has no field aliased to '_id' and "
                "does not set 'extra' in model_config, so decoded documents would silently "
                "lose their '_id' key. Map the key explicitly, e.g. "
                "id: ObjectId = Field(alias='_id'), or opt out of strict decoding with "
                "model_config = ConfigDict(extra='ignore')"
            )
        self._extra: Optional[str] = None
        if extra is None and "extra" in inspect.signature(document_type.model_validate).parameters:
            self._extra = "forbid"

    def from_bson(self, doc: dict[str, Any], codec_options: CodecOptions[Any]) -> Any:
        if self._extra is not None:
            return self.document_type.model_validate(doc, extra=self._extra)
        return self.document_type.model_validate(doc)

    def to_bson(self, doc: Any, codec_options: CodecOptions[Any]) -> Any:
        return doc.model_dump(by_alias=True)


def _resolve_document_class(document_class: Any) -> Optional[Any]:
    """Resolve a non-mapping document_class to a typed decoding implementation.

    Returns the argument unchanged if it already implements the protocol, a shipped adapter
    for dataclasses and pydantic v2 models, or ``None`` if unsupported.
    Raises ``TypeError`` for pydantic v1 models.
    """
    if _bson_deserializable_class(document_class):
        if (
            getattr(document_class, "from_bson", None) is None
            and getattr(document_class, "to_bson", None) is None
        ):
            raise TypeError(
                f"{document_class!r} sets _type_marker = {_BSON_DESERIALIZABLE_MARKER} "
                "but does not implement from_bson or to_bson"
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
