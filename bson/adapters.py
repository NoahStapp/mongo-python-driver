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
import sys
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from bson.codec_options import CodecOptions

_BSON_DESERIALIZABLE_MARKER = 102


def _bson_deserializable_class(document_class: Any) -> bool:
    """Return True if `document_class` implements the typed decoding protocol."""
    return getattr(document_class, "_type_marker", None) == _BSON_DESERIALIZABLE_MARKER


class _DocumentAdapter:
    """Wraps a user document type in the ``from_bson`` protocol."""

    _type_marker = _BSON_DESERIALIZABLE_MARKER

    def __init__(self, document_type: type[Any]) -> None:
        self.document_type = document_type

    def from_bson(self, doc: dict[str, Any], codec_options: CodecOptions[Any]) -> Any:
        """Construct one ``document_type`` instance from a decoded document."""
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

    def from_bson(self, doc: dict[str, Any], codec_options: CodecOptions[Any]) -> Any:
        return self.document_type(**doc)


class _PydanticAdapter(_DocumentAdapter):
    """Decodes BSON into a pydantic v2 model via ``model_validate``."""

    def from_bson(self, doc: dict[str, Any], codec_options: CodecOptions[Any]) -> Any:
        return self.document_type.model_validate(doc)


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
