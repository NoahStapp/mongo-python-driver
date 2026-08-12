# Copyright 2023-Present MongoDB, Inc.
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

"""Type aliases used by bson"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, Union

if TYPE_CHECKING:
    from array import array
    from mmap import mmap

    from _typeshed import DataclassInstance

    from bson.raw_bson import RawBSONDocument

    class _SupportsFromBson(Protocol):
        """A class opting into typed decoding via the ``from_bson`` hook."""

        _type_marker: int

        @classmethod
        def from_bson(cls, doc: dict[str, Any], codec_options: Any) -> Any: ...

    class _PydanticModelLike(Protocol):
        """Structural stand-in for pydantic.BaseModel, which bson cannot import."""

        @classmethod
        def model_validate(cls, obj: Any, /) -> Any: ...

    # Mirrors the runtime document_class acceptance rules in
    # bson.adapters._resolve_document_class.
    _DocumentTypeBound = Union[
        Mapping[str, Any], DataclassInstance, _SupportsFromBson, _PydanticModelLike
    ]


# Common Shared Types.
_DocumentOut = Union[MutableMapping[str, Any], "RawBSONDocument"]
_DocumentType = TypeVar("_DocumentType", bound="_DocumentTypeBound")
_DocumentTypeArg = TypeVar("_DocumentTypeArg", bound="_DocumentTypeBound")
# For decode surfaces with no typed-document support (bson.decode and friends),
# which instantiate document_class as a mapping.
_MappingDocumentType = TypeVar("_MappingDocumentType", bound=Mapping[str, Any])
_ReadableBuffer = Union[bytes, memoryview, bytearray, "mmap", "array"]  # type: ignore[type-arg]
