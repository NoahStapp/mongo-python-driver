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

Finally, a class whose instances are built by plain assignment of decoded
fields may advertise a static construction layout::

    @classmethod
    def __bson_construct_plan__(cls):
        return BsonConstructPlan(fields=("_id", "name", "age"))

which lets the driver construct instances itself, one batched decode per
cursor batch with no per-document protocol calls (and, in a future C fast
path, no intermediate dict). The plan takes precedence over
``from_bson_batch``, which takes precedence over ``from_bson``; each tier
is optional with the tier below as its fallback.

Plain dataclasses and pydantic v2 models are auto-wrapped in the shipped
adapters below at the ``CodecOptions`` validation gate.
"""

from __future__ import annotations

import dataclasses
import struct
import sys
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

if TYPE_CHECKING:
    from bson.codec_options import CodecOptions

_BSON_DESERIALIZABLE_MARKER = 102


def _bson_deserializable_class(document_class: Any) -> bool:
    """Return True if `document_class` implements the ``from_bson`` protocol."""
    return getattr(document_class, "_type_marker", None) == _BSON_DESERIALIZABLE_MARKER


class BsonConstructPlan(NamedTuple):
    """Static construction layout advertised via ``__bson_construct_plan__``.

    The plan is queried once per batch, never per document; it must be
    immutable and describe construction only — validation or coercion belongs
    in the class's constructor.

    Under the ``"call"`` strategy every decoded key is passed to the factory
    as a keyword argument (after ``fields``->``params`` renaming), so extra
    wire keys raise ``TypeError`` from the constructor and missing keys fall
    through to constructor defaults — the same semantics as ``cls(**decoded)``.
    Under ``"setattr"`` the instance is allocated with ``__new__`` and fields
    are assigned directly; ``__init__`` never runs.
    """

    fields: tuple[str, ...]
    """Expected wire keys, in expected order."""

    params: Optional[tuple[str, ...]] = None
    """Constructor keyword names for ``fields``; ``None`` means identical."""

    strategy: str = "call"
    """``"call"`` (invoke the factory) or ``"setattr"`` (allocate + assign)."""

    factory: Optional[Any] = None
    """Callable constructing one instance; ``None`` means the class itself."""


def _construct_batch_with_plan(
    document_class: Any,
    plan: BsonConstructPlan,
    data: bytes | memoryview,
    codec_options: CodecOptions[Any],
) -> list[Any]:
    """Pure-Python reference executor for ``__bson_construct_plan__``.

    Decodes the whole buffer with one ``decode_all`` call and constructs the
    instances per the plan. A C fast path would instead decode straight into
    a vectorcall argument array, skipping the intermediate dicts.
    """
    import bson

    if plan.strategy not in ("call", "setattr"):
        raise ValueError(f"unknown BsonConstructPlan strategy: {plan.strategy!r}")
    target = plan.factory if plan.factory is not None else document_class
    rename = None
    if plan.params is not None and plan.params != plan.fields:
        rename = dict(zip(plan.fields, plan.params))
    dict_options = codec_options.with_options(document_class=dict)
    docs = []
    for decoded in bson.decode_all(data, dict_options):
        if rename is None:
            fields = decoded
        else:
            fields = {rename.get(key, key): value for key, value in decoded.items()}
        if plan.strategy == "call":
            docs.append(target(**fields))
        else:
            obj = target.__new__(target)
            for key, value in fields.items():
                setattr(obj, key, value)
            docs.append(obj)
    return docs


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

    def _from_dict(self, decoded: dict[str, Any]) -> Any:
        """Construct one ``document_type`` instance from a decoded document."""
        raise NotImplementedError

    def from_bson(self, data: Any, codec_options: CodecOptions[Any]) -> Any:
        import bson

        return self._from_dict(bson.decode(data, self._as_dict_options(codec_options)))

    def from_bson_batch(self, data: Any, codec_options: CodecOptions[Any]) -> list[Any]:
        import bson

        from_dict = self._from_dict
        return [
            from_dict(decoded)
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
    """Decodes BSON into a plain dataclass.

    Built on the construct-plan layer: a dataclass has a static field set
    and a named-parameter ``__init__``, so the driver can construct it
    directly. ``_from_dict`` remains the fallback tier.
    """

    def __init__(self, document_type: type[Any]) -> None:
        super().__init__(document_type)
        self._plan = BsonConstructPlan(
            fields=tuple(field.name for field in dataclasses.fields(document_type)),
            factory=document_type,
        )

    def __bson_construct_plan__(self) -> BsonConstructPlan:
        return self._plan

    def _from_dict(self, decoded: dict[str, Any]) -> Any:
        return self.document_type(**decoded)


class _PydanticAdapter(_DocumentAdapter):
    """Decodes BSON into a pydantic v2 model via ``model_validate``."""

    def _from_dict(self, decoded: dict[str, Any]) -> Any:
        return self.document_type.model_validate(decoded)


def _decode_typed_batch(
    document_class: Any, data: bytes | memoryview, codec_options: CodecOptions[Any]
) -> list[Any]:
    """Decode a buffer of back-to-back raw BSON documents into instances.

    Dispatches to the richest protocol tier the class provides: a
    ``__bson_construct_plan__`` layout (driver-side construction), then
    ``from_bson_batch`` (one batched decode), then slicing the buffer and
    calling ``from_bson`` once per document.
    """
    plan_hook = getattr(document_class, "__bson_construct_plan__", None)
    if plan_hook is not None:
        return _construct_batch_with_plan(document_class, plan_hook(), data, codec_options)
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
