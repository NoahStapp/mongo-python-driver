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

PYTHON-4192 proof of concept. A class opts into typed decoding by setting
``_type_marker = 102`` and providing::

    @classmethod
    def from_bson(cls, data, codec_options):
        ...

where ``data`` is the raw BSON bytes of one document. A class may also
implement the optional batch hook::

    @classmethod
    def from_bson_batch(cls, data, codec_options):
        ...

where ``data`` is a buffer of N raw BSON documents laid out back-to-back;
it must return a list of N instances. When absent, the driver slices the
buffer and calls ``from_bson`` once per document.

A class that wants decoded documents rather than raw bytes implements the
dict-level hook instead::

    @classmethod
    def from_bson_dict(cls, doc, codec_options):
        ...

where ``doc`` is one fully decoded document: the driver decodes each
cursor batch itself (one batched decode, no raw-batch handling) and calls
the hook once per document. This is the easiest tier to implement — no
BSON handling in user code — and the fastest. With it present,
``from_bson`` is not required.

The tiers form a ladder of decreasing driver involvement, and the driver
uses the highest rung a class implements: ``from_bson_dict`` (driver
decodes, class constructs), then ``from_bson_batch`` (class decodes the
batch buffer), then ``from_bson`` (class decodes each document).

Plain dataclasses and pydantic v2 models are auto-wrapped in the shipped
adapters below at the ``CodecOptions`` validation gate.
"""

from __future__ import annotations

import dataclasses
import struct
import sys
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from bson.codec_options import CodecOptions

_BSON_DESERIALIZABLE_MARKER = 102


def _bson_deserializable_class(document_class: Any) -> bool:
    """Return True if `document_class` implements the ``from_bson`` protocol."""
    return getattr(document_class, "_type_marker", None) == _BSON_DESERIALIZABLE_MARKER


def _dict_batch_constructor(
    document_class: Any,
    codec_options: CodecOptions[Any],
) -> Optional[Callable[[list[dict[str, Any]]], list[Any]]]:
    """Return a callable constructing instances from already-decoded documents.

    Classes on the ``from_bson_dict`` tier (which includes the shipped
    adapters) need no raw batch bytes, which lets reply unpacking decode
    the whole reply to dicts in a single pass (see
    ``pymongo.message._unpack_typed_response``). Returns ``None`` for
    ``from_bson``/``from_bson_batch``-only implementations: those are owed
    raw BSON bytes.
    """
    from_bson_dict = getattr(document_class, "from_bson_dict", None)
    if from_bson_dict is not None:
        return lambda docs: [from_bson_dict(decoded, codec_options) for decoded in docs]
    return None


class _DocumentAdapter:
    """Wraps a user document type in the ``from_bson`` protocol.

    ``__eq__``/``__hash__``/``__repr__`` are required because an adapter is
    stored as the ``document_class`` field of the ``CodecOptions`` namedtuple,
    which is compared and repr'd (client repr, ``with_options``).
    """

    _type_marker = _BSON_DESERIALIZABLE_MARKER

    def __init__(self, document_type: type[Any]) -> None:
        self.document_type = document_type
        self._dict_options_cache: Optional[tuple[CodecOptions[Any], CodecOptions[Any]]] = None

    def _as_dict_options(self, codec_options: CodecOptions[Any]) -> CodecOptions[Any]:
        """These codec options with ``document_class`` replaced by ``dict``.

        ``from_bson`` is called once per document with the same
        ``codec_options`` object (the one this adapter is stored in), so the
        derived options are cached: ``with_options`` reruns the full
        ``CodecOptions`` validation gate, which is too expensive per document.
        """
        cached = self._dict_options_cache
        if cached is not None and cached[0] is codec_options:
            return cached[1]
        dict_options = codec_options.with_options(document_class=dict)
        self._dict_options_cache = (codec_options, dict_options)
        return dict_options

    def from_bson_dict(self, doc: dict[str, Any], codec_options: CodecOptions[Any]) -> Any:
        """Construct one ``document_type`` instance from a decoded document."""
        raise NotImplementedError

    def from_bson(self, data: Any, codec_options: CodecOptions[Any]) -> Any:
        import bson

        return self.from_bson_dict(
            bson.decode(data, self._as_dict_options(codec_options)), codec_options
        )

    def from_bson_batch(self, data: Any, codec_options: CodecOptions[Any]) -> list[Any]:
        import bson

        from_bson_dict = self.from_bson_dict
        return [
            from_bson_dict(decoded, codec_options)
            for decoded in bson.decode_all(data, self._as_dict_options(codec_options))
        ]

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

    def from_bson_dict(self, doc: dict[str, Any], codec_options: CodecOptions[Any]) -> Any:
        return self.document_type(**doc)


class _PydanticAdapter(_DocumentAdapter):
    """Decodes BSON into a pydantic v2 model via ``model_validate``."""

    def from_bson_dict(self, doc: dict[str, Any], codec_options: CodecOptions[Any]) -> Any:
        return self.document_type.model_validate(doc)


def _decode_typed_batch(
    document_class: Any, data: bytes | memoryview, codec_options: CodecOptions[Any]
) -> list[Any]:
    """Decode a buffer of back-to-back raw BSON documents into instances.

    Dispatches to the richest protocol tier the class provides:
    ``from_bson_dict`` (one batched decode, one hook call per decoded
    document), then ``from_bson_batch`` (one batched decode by the class),
    then slicing the buffer and calling ``from_bson`` once per document.
    """
    from_bson_dict = getattr(document_class, "from_bson_dict", None)
    if from_bson_dict is not None:
        import bson

        dict_options = codec_options.with_options(document_class=dict)
        return [
            from_bson_dict(decoded, codec_options)
            for decoded in bson.decode_all(data, dict_options)
        ]
    from_bson_batch = getattr(document_class, "from_bson_batch", None)
    if from_bson_batch is not None:
        return from_bson_batch(data, codec_options)
    view = memoryview(data)
    docs = []
    position = 0
    obj_end = len(view)
    while position < obj_end:
        # The first four bytes of a BSON document hold its total size.
        # bson._get_object_size() is not used here: it assumes position 0
        # marks the start of a single top-level document spanning the whole
        # buffer, which does not hold for a concatenation of N documents
        # (mirrors the multi-document loop in bson._decode_all()).
        obj_size = struct.unpack_from("<i", view, position)[0]
        docs.append(document_class.from_bson(view[position : position + obj_size], codec_options))
        position += obj_size
    return docs


def _resolve_document_class(document_class: Any) -> Optional[Any]:
    """Resolve a non-mapping document_class to a ``from_bson`` implementation.

    Returns the argument unchanged if it already implements the protocol
    (explicit implementations are never second-guessed), a shipped adapter
    for dataclasses and pydantic v2 models, or ``None`` if unsupported (the
    caller raises today's ``TypeError``). Raises ``TypeError`` for pydantic
    v1 models.
    """
    if _bson_deserializable_class(document_class):
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
