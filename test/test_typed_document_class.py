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

"""Test typed document_class support (PYTHON-4192 PoC)."""

from __future__ import annotations

import datetime
import inspect
import sys
import types
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Optional
from unittest import mock

sys.path[0:0] = [""]

import bson
from bson import encode
from bson.adapters import (
    _BSON_DESERIALIZABLE_MARKER,
    _bson_deserializable_class,
    _convert_typed_document,
    _DataclassAdapter,
    _DocumentAdapter,
    _PydanticAdapter,
    _resolve_document_class,
)
from bson.codec_options import CodecOptions
from bson.int64 import Int64
from bson.objectid import ObjectId
from bson.raw_bson import RawBSONDocument
from bson.son import SON
from pymongo.common import validate_document_class
from pymongo.errors import OperationFailure
from pymongo.message import _unpack_typed_response
from pymongo.operations import InsertOne, ReplaceOne
from test import (
    IntegrationTest,
    UnitTest,
    client_context,
    unittest,
)
from test.utils_shared import OvertCommandListener

_IS_SYNC = True


@dataclass
class UserDC:
    _id: ObjectId
    name: str
    age: int


@dataclass
class AutoIdDC:
    """A round-trip-capable dataclass whose ``_id`` defaults to None."""

    name: str
    age: int
    _id: Optional[ObjectId] = None


@dataclass
class OtherDC:
    _id: ObjectId
    color: str


class ProtocolDoc:
    """A hand-rolled implementation of the from_bson protocol."""

    _type_marker = _BSON_DESERIALIZABLE_MARKER

    def __init__(self, fields: dict[str, Any]) -> None:
        self.fields = fields

    @classmethod
    def from_bson(cls, doc: dict[str, Any], codec_options: CodecOptions[Any]) -> ProtocolDoc:
        return cls(doc)


class EncodableProtocolDoc(ProtocolDoc):
    """A protocol document class that also implements the to_bson hook."""

    @classmethod
    def to_bson(cls, doc: EncodableProtocolDoc, codec_options: CodecOptions[Any]) -> dict[str, Any]:
        return dict(doc.fields)


class NotADocumentClass:
    pass


try:
    from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

    _HAVE_PYDANTIC = True
    # model_validate accepts a runtime `extra` argument from pydantic 2.12.
    _PYDANTIC_RUNTIME_EXTRA = "extra" in inspect.signature(BaseModel.model_validate).parameters

    class UserModel(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
        id: ObjectId = Field(alias="_id")
        name: str
        age: int

except ImportError:
    _HAVE_PYDANTIC = False
    _PYDANTIC_RUNTIME_EXTRA = False


class TestDocumentClassResolution(UnitTest):
    def test_protocol_class_used_as_is(self):
        self.assertIs(_resolve_document_class(ProtocolDoc), ProtocolDoc)

    def test_marker_class_without_hook_rejected(self):
        class HooklessDoc:
            _type_marker = _BSON_DESERIALIZABLE_MARKER

        with self.assertRaisesRegex(TypeError, "does not implement from_bson"):
            _resolve_document_class(HooklessDoc)

    def test_dataclass_wrapped_in_adapter(self):
        resolved = _resolve_document_class(UserDC)
        self.assertIsInstance(resolved, _DataclassAdapter)
        self.assertIs(resolved.document_type, UserDC)
        self.assertTrue(_bson_deserializable_class(resolved))

    def test_unsupported_class_returns_none(self):
        self.assertIsNone(_resolve_document_class(NotADocumentClass))

    def test_adapter_instance_resolves_to_itself(self):
        adapter = _DataclassAdapter(UserDC)
        self.assertIs(_resolve_document_class(adapter), adapter)

    def test_pydantic_not_imported_returns_none(self):
        # Without pydantic in sys.modules, an arbitrary class is not sniffed.
        with mock.patch.dict(sys.modules):
            sys.modules.pop("pydantic", None)
            self.assertIsNone(_resolve_document_class(NotADocumentClass))

    def test_pydantic_v1_model_rejected(self):
        fake_pydantic = types.ModuleType("pydantic")

        class FakeBaseModel:
            pass

        fake_pydantic.BaseModel = FakeBaseModel  # type: ignore[attr-defined]

        class V1Model(FakeBaseModel):
            pass

        with mock.patch.dict(sys.modules, {"pydantic": fake_pydantic}):
            with self.assertRaisesRegex(TypeError, "pydantic v2"):
                _resolve_document_class(V1Model)

    def test_pydantic_v2_model_wrapped_in_adapter(self):
        fake_pydantic = types.ModuleType("pydantic")

        class FakeBaseModel:
            model_config = {"extra": "ignore"}

            @classmethod
            def model_validate(cls, obj: Any) -> Any:
                return obj

        fake_pydantic.BaseModel = FakeBaseModel  # type: ignore[attr-defined]

        class V2Model(FakeBaseModel):
            pass

        with mock.patch.dict(sys.modules, {"pydantic": fake_pydantic}):
            resolved = _resolve_document_class(V2Model)
        self.assertIsInstance(resolved, _PydanticAdapter)
        self.assertIs(resolved.document_type, V2Model)


class TestAdapters(UnitTest):
    def test_adapter_eq_hash_repr(self):
        class OtherAdapter(_DocumentAdapter):
            pass

        a, b = _DataclassAdapter(UserDC), _DataclassAdapter(UserDC)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, OtherAdapter(UserDC))
        self.assertIn("UserDC", repr(a))

    def test_dataclass_extra_keys_raise_type_error(self):
        # Same semantics as cls(**decoded): unexpected wire keys are passed
        # through and the constructor rejects them.
        with self.assertRaises(TypeError):
            _DataclassAdapter(UserDC).from_bson(
                {"_id": ObjectId(), "name": "x", "age": 1, "extra": True}, CodecOptions()
            )

    def test_dataclass_missing_keys_use_constructor_defaults(self):
        @dataclass
        class DefaultedDC:
            _id: ObjectId
            name: str
            age: int = -1

        doc = _DataclassAdapter(DefaultedDC).from_bson(
            {"_id": ObjectId(), "name": "x"}, CodecOptions()
        )
        self.assertEqual(doc.age, -1)

    def test_bson_scalars_survive_dataclass_path(self):
        oid = ObjectId()
        when = datetime.datetime(2026, 7, 23, 12, 0, 0)

        @dataclass
        class Event:
            _id: ObjectId
            when: datetime.datetime

        event = _DataclassAdapter(Event).from_bson(
            bson.decode(encode({"_id": oid, "when": when})), CodecOptions()
        )
        self.assertEqual(event._id, oid)
        self.assertEqual(event.when, when)

    def test_adapters_expose_from_bson(self):
        oid = ObjectId()
        user = _DataclassAdapter(UserDC).from_bson(
            {"_id": oid, "name": "Ada", "age": 36}, CodecOptions()
        )
        self.assertEqual(user, UserDC(oid, "Ada", 36))

        class FakeModel:
            model_config = {"extra": "ignore"}

            @classmethod
            def model_validate(cls, obj: Any) -> Any:
                return ("validated", obj["name"])

        self.assertEqual(
            _PydanticAdapter(FakeModel).from_bson({"name": "x"}, CodecOptions()),
            ("validated", "x"),
        )


@unittest.skipUnless(_HAVE_PYDANTIC, "pydantic v2 is not installed")
class TestPydanticAdapterContract(UnitTest):
    """The _id / unknown-key contract for pydantic models (PYTHON-4192 finding #9).

    Pydantic cannot declare ``_id`` as a plain field (leading underscore means
    private attribute), so a model must either map the ``_id`` document key
    through a field alias or opt out of strict decoding explicitly.
    """

    def test_model_without_id_mapping_rejected(self):
        class Natural(BaseModel):
            name: str
            age: int

        with self.assertRaisesRegex(TypeError, "aliased to '_id'"):
            CodecOptions(document_class=Natural)

    def test_explicit_extra_ignore_opts_out(self):
        class Lenient(BaseModel):
            model_config = ConfigDict(extra="ignore")
            name: str

        opts = CodecOptions(document_class=Lenient)
        doc = opts._document_adapter.from_bson({"_id": ObjectId(), "name": "x"}, opts)
        self.assertEqual(doc, Lenient(name="x"))

    def test_explicit_extra_allow_preserves_id(self):
        class Open(BaseModel):
            model_config = ConfigDict(extra="allow")
            name: str

        opts = CodecOptions(document_class=Open)
        oid = ObjectId()
        doc = opts._document_adapter.from_bson({"_id": oid, "name": "x"}, opts)
        self.assertEqual(doc.model_extra, {"_id": oid})

    def test_aliased_model_decodes_id(self):
        opts = CodecOptions(document_class=UserModel)
        oid = ObjectId()
        user = opts._document_adapter.from_bson({"_id": oid, "name": "x", "age": 1}, opts)
        self.assertEqual(user.id, oid)

    @unittest.skipUnless(_PYDANTIC_RUNTIME_EXTRA, "requires pydantic >= 2.12")
    def test_unknown_keys_rejected_by_default(self):
        opts = CodecOptions(document_class=UserModel)
        with self.assertRaises(ValidationError):
            opts._document_adapter.from_bson(
                {"_id": ObjectId(), "name": "x", "age": 1, "stray": True}, opts
            )

    def test_explicit_extra_config_beats_default_strictness(self):
        class Lenient(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")
            id: ObjectId = Field(alias="_id")
            name: str

        opts = CodecOptions(document_class=Lenient)
        doc = opts._document_adapter.from_bson(
            {"_id": ObjectId(), "name": "x", "stray": True}, opts
        )
        self.assertEqual(doc.name, "x")

    def test_validation_alias_forms_recognized(self):
        class ViaValidationAlias(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            ident: ObjectId = Field(validation_alias="_id")

        class ViaAliasChoices(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            ident: ObjectId = Field(validation_alias=AliasChoices("_id", "ident"))

        for model in (ViaValidationAlias, ViaAliasChoices):
            with self.subTest(model=model.__name__):
                opts = CodecOptions(document_class=model)
                self.assertIs(opts.document_class, model)
                self.assertIsInstance(opts._document_adapter, _PydanticAdapter)


class TestFromBsonHook(UnitTest):
    """Semantics of the protocol hook itself."""

    def test_from_bson_class(self):
        doc = ProtocolDoc.from_bson({"a": 1}, CodecOptions())
        self.assertIsInstance(doc, ProtocolDoc)
        self.assertEqual(doc.fields, {"a": 1})

    def test_from_bson_polymorphic_dispatch(self):
        # Discriminated unions: the hook sees the decoded document, so it
        # can dispatch construction on a type field.
        class Shape:
            _type_marker = _BSON_DESERIALIZABLE_MARKER

            def __init__(self, doc: dict[str, Any]) -> None:
                self.doc = doc

            @classmethod
            def from_bson(cls, doc: dict[str, Any], codec_options: CodecOptions[Any]) -> Shape:
                return shapes[doc["kind"]](doc)

        class Circle(Shape):
            pass

        class Square(Shape):
            pass

        shapes = {"circle": Circle, "square": Square}
        circle = Shape.from_bson({"kind": "circle", "r": 1.0}, CodecOptions())
        square = Shape.from_bson({"kind": "square", "s": 2.0}, CodecOptions())
        self.assertIsInstance(circle, Circle)
        self.assertIsInstance(square, Square)


class TestConvertTypedDocument(UnitTest):
    """Write-path conversion of typed document instances into documents."""

    def test_mappings_pass_through_unchanged(self):
        # Mappings never convert, even with a typed adapter configured.
        opts = CodecOptions(document_class=UserDC)
        for doc in ({"x": 1}, SON([("x", 1)]), RawBSONDocument(encode({"x": 1}))):
            with self.subTest(type=type(doc).__name__):
                self.assertIs(_convert_typed_document(doc, opts), doc)

    def test_typed_instance_converted_to_document(self):
        oid = ObjectId()
        opts = CodecOptions(document_class=UserDC)
        converted = _convert_typed_document(UserDC(oid, "Ada", 36), opts)
        self.assertEqual(converted, {"_id": oid, "name": "Ada", "age": 36})

    def test_no_adapter_returns_document_unchanged(self):
        # Untyped codec options never convert; downstream validation rejects.
        user = UserDC(ObjectId(), "Ada", 36)
        self.assertIs(_convert_typed_document(user, CodecOptions()), user)

    def test_unrelated_instance_not_converted(self):
        # Only document_type instances are encoded: an instance of a
        # different (even encodable) class passes through for validation
        # to reject instead of being silently encoded with its own fields.
        opts = CodecOptions(document_class=UserDC)
        other = OtherDC(ObjectId(), "red")
        self.assertIs(_convert_typed_document(other, opts), other)

    def test_protocol_class_with_to_bson_converts(self):
        opts = CodecOptions(document_class=EncodableProtocolDoc)
        converted = _convert_typed_document(EncodableProtocolDoc({"x": 1}), opts)
        self.assertEqual(converted, {"x": 1})

    def test_protocol_class_without_to_bson_rejected(self):
        # A from_bson-only protocol class is decode-only: writes must fail
        # with a clear error, not an adapter-internal AttributeError.
        opts = CodecOptions(document_class=ProtocolDoc)
        with self.assertRaisesRegex(TypeError, "does not implement to_bson"):
            _convert_typed_document(ProtocolDoc({"x": 1}), opts)


class TestToBsonHook(UnitTest):
    """Semantics of the adapters' to_bson hook."""

    def test_dataclass_to_bson_dumps_fields(self):
        oid = ObjectId()
        dumped = _DataclassAdapter(UserDC).to_bson(UserDC(oid, "Ada", 36), CodecOptions())
        self.assertEqual(dumped, {"_id": oid, "name": "Ada", "age": 36})

    def test_dataclass_to_bson_drops_none_id(self):
        # A default None _id must not reach the server: call sites see a
        # missing _id and generate an ObjectId client side instead of
        # storing _id: null.
        dumped = _DataclassAdapter(AutoIdDC).to_bson(AutoIdDC("Ada", 36), CodecOptions())
        self.assertEqual(dumped, {"name": "Ada", "age": 36})

    def test_dataclass_to_bson_recurses_into_nested_dataclasses(self):
        @dataclass
        class Address:
            city: str

        @dataclass
        class Person:
            _id: ObjectId
            address: Address

        oid = ObjectId()
        dumped = _DataclassAdapter(Person).to_bson(Person(oid, Address("NYC")), CodecOptions())
        self.assertEqual(dumped, {"_id": oid, "address": {"city": "NYC"}})
        # The dumped document must survive wire encoding.
        encode(dumped)

    def test_dataclass_without_id_field_rejected(self):
        # A dataclass with no _id field would write documents its own
        # from_bson could never read back, so it is rejected up front.
        @dataclass
        class NoIdDC:
            name: str

        with self.assertRaisesRegex(TypeError, "_id"):
            _DataclassAdapter(NoIdDC)
        with self.assertRaisesRegex(TypeError, "_id"):
            CodecOptions(document_class=NoIdDC)

    @unittest.skipUnless(_HAVE_PYDANTIC, "pydantic v2 is not installed")
    def test_pydantic_to_bson_dumps_wire_names(self):
        # Aliased fields (id -> _id) are dumped under their wire names, so
        # a dumped document round-trips through from_bson.
        oid = ObjectId()
        adapter = _PydanticAdapter(UserModel)
        dumped = adapter.to_bson(UserModel(id=oid, name="Ada", age=36), CodecOptions())
        self.assertEqual(dumped, {"_id": oid, "name": "Ada", "age": 36})
        self.assertEqual(
            adapter.from_bson(dumped, CodecOptions()), UserModel(id=oid, name="Ada", age=36)
        )


class TestUnpackTypedResponseSinglePass(UnitTest):
    """Reply unpacking for typed document classes."""

    USER_FIELDS = {"cursor": {"firstBatch": 1}}

    def setUp(self):
        super().setUp()
        self.oids = [ObjectId() for _ in range(2)]
        self.batch = [{"_id": oid, "name": f"user{i}", "age": i} for i, oid in enumerate(self.oids)]

    def _payload(self, **extra_cursor_fields: Any) -> bytes:
        cursor: dict[str, Any] = {"firstBatch": self.batch, "id": Int64(0), "ns": "db.coll"}
        cursor.update(extra_cursor_fields)
        return encode({"cursor": cursor, "ok": 1.0})

    def test_batch_documents_constructed(self):
        opts = CodecOptions(document_class=UserDC)
        (envelope,) = _unpack_typed_response(self._payload(), opts, self.USER_FIELDS)
        self.assertEqual(
            envelope["cursor"]["firstBatch"],
            [UserDC(oid, f"user{i}", i) for i, oid in enumerate(self.oids)],
        )

    def test_envelope_subdocuments_are_plain_dicts(self):
        # Non-user envelope fields must decode like the plain-dict path:
        # nested documents (e.g. postBatchResumeToken) come back as dicts,
        # not RawBSONDocument.
        opts = CodecOptions(document_class=UserDC)
        payload = self._payload(postBatchResumeToken={"_data": "abc"})
        (envelope,) = _unpack_typed_response(payload, opts, self.USER_FIELDS)
        token = envelope["cursor"]["postBatchResumeToken"]
        self.assertIsInstance(token, dict)
        self.assertEqual(token, {"_data": "abc"})

    def test_envelope_respects_codec_options(self):
        # Envelope fields outside user_fields must honor the user's codec
        # options (here tz_aware), like the plain-dict path does.
        when = datetime.datetime(2026, 8, 6, tzinfo=datetime.timezone.utc)
        opts: CodecOptions[Any] = CodecOptions(document_class=UserDC, tz_aware=True)
        payload = self._payload(atTime=when)
        (envelope,) = _unpack_typed_response(payload, opts, self.USER_FIELDS)
        self.assertEqual(envelope["cursor"]["atTime"], when)

    def test_next_batch_constructed(self):
        opts = CodecOptions(document_class=UserDC)
        payload = encode(
            {"cursor": {"nextBatch": self.batch, "id": Int64(0), "ns": "db.coll"}, "ok": 1.0}
        )
        (envelope,) = _unpack_typed_response(payload, opts, {"cursor": {"nextBatch": 1}})
        batch = envelope["cursor"]["nextBatch"]
        self.assertEqual(len(batch), 2)
        self.assertIsInstance(batch[0], UserDC)

    def test_from_bson_class_takes_single_pass(self):
        opts = CodecOptions(document_class=ProtocolDoc)
        (envelope,) = _unpack_typed_response(self._payload(), opts, self.USER_FIELDS)
        docs = envelope["cursor"]["firstBatch"]
        self.assertEqual([doc.fields["age"] for doc in docs], [0, 1])
        self.assertIsInstance(docs[0], ProtocolDoc)

    def test_hook_called_once_per_batch_document(self):
        opts = CodecOptions(document_class=UserDC)
        adapter = opts._document_adapter
        with mock.patch.object(adapter, "from_bson", wraps=adapter.from_bson) as dict_hook:
            _unpack_typed_response(self._payload(), opts, self.USER_FIELDS)
        self.assertEqual(dict_hook.call_count, len(self.batch))

    def test_hook_receives_original_codec_options(self):
        # The hook must see the user's codec options (the ones holding the
        # typed document_class), not the dict-variant used for decoding.
        seen: list[CodecOptions[Any]] = []

        class CapturingDoc(ProtocolDoc):
            @classmethod
            def from_bson(
                cls, doc: dict[str, Any], codec_options: CodecOptions[Any]
            ) -> CapturingDoc:
                seen.append(codec_options)
                return cls(doc)

        opts = CodecOptions(document_class=CapturingDoc)
        _unpack_typed_response(self._payload(), opts, self.USER_FIELDS)
        self.assertEqual(len(seen), len(self.batch))
        for received in seen:
            self.assertIs(received, opts)


class TestCodecOptionsGate(UnitTest):
    def test_dataclass_document_class(self):
        opts = CodecOptions(document_class=UserDC)
        self.assertIs(opts.document_class, UserDC)
        self.assertIsInstance(opts._document_adapter, _DataclassAdapter)
        self.assertIs(opts._document_adapter.document_type, UserDC)

    def test_protocol_document_class(self):
        opts = CodecOptions(document_class=ProtocolDoc)
        self.assertIs(opts.document_class, ProtocolDoc)
        self.assertIs(opts._document_adapter, ProtocolDoc)

    def test_document_class_supports_mapping_introspection(self):
        # The public attribute holds the user's class, so mapping checks
        # answer False instead of raising on an adapter instance.
        opts = CodecOptions(document_class=UserDC)
        self.assertFalse(issubclass(opts.document_class, MutableMapping))

    def test_document_adapter_is_memoized(self):
        opts = CodecOptions(document_class=UserDC)
        self.assertIs(opts._document_adapter, opts._document_adapter)

    def test_document_adapter_none_for_mapping_classes(self):
        self.assertIsNone(CodecOptions()._document_adapter)
        self.assertIsNone(CodecOptions(document_class=SON)._document_adapter)
        self.assertIsNone(CodecOptions(document_class=RawBSONDocument)._document_adapter)

    def test_document_adapter_survives_replace(self):
        # _replace/_make bypass __new__, so the adapter must resolve lazily.
        opts = CodecOptions(document_class=UserDC)._replace(tz_aware=True)
        self.assertIsInstance(opts._document_adapter, _DataclassAdapter)

    def test_adapter_resolved_lazily_and_memoized(self):
        # __new__ resolves the adapter once for validation only; the first
        # _document_adapter access resolves it again and caches it, so
        # repeated access on the per-reply hot path costs no resolution.
        import bson.codec_options as codec_options_mod

        with mock.patch.object(
            codec_options_mod,
            "_resolve_document_class",
            wraps=_resolve_document_class,
        ) as resolve:
            opts = CodecOptions(document_class=UserDC)
            self.assertEqual(resolve.call_count, 1)
            self.assertIsInstance(opts._document_adapter, _DataclassAdapter)
            self.assertEqual(resolve.call_count, 2)
            self.assertIsInstance(opts._document_adapter, _DataclassAdapter)
            self.assertEqual(resolve.call_count, 2)

    def test_dict_options_memoized(self):
        opts = CodecOptions(document_class=UserDC, tz_aware=True)
        dict_options = opts._dict_options
        self.assertIs(dict_options.document_class, dict)
        self.assertEqual(dict_options, opts.with_options(document_class=dict))
        # Memoized: the typed unpack path reads this on every reply.
        self.assertIs(opts._dict_options, dict_options)

    def test_unsupported_class_still_raises(self):
        with self.assertRaisesRegex(TypeError, "document_class must be dict"):
            # The static rejection matches the runtime one.
            CodecOptions(document_class=NotADocumentClass)  # type: ignore[type-var]

    def test_marker_without_hook_rejected_at_construction(self):
        class HooklessDoc:
            _type_marker = _BSON_DESERIALIZABLE_MARKER

        with self.assertRaisesRegex(TypeError, "does not implement from_bson"):
            # The static rejection matches the runtime one.
            CodecOptions(document_class=HooklessDoc)  # type: ignore[type-var]

    def test_equality_and_repr(self):
        self.assertEqual(CodecOptions(document_class=UserDC), CodecOptions(document_class=UserDC))
        self.assertNotEqual(CodecOptions(document_class=UserDC), CodecOptions())
        self.assertIn("UserDC", repr(CodecOptions(document_class=UserDC)))
        self.assertNotIn("Adapter", repr(CodecOptions(document_class=UserDC)))

    def test_with_options_round_trip(self):
        opts = CodecOptions(document_class=UserDC)
        self.assertEqual(opts.with_options(tz_aware=True).document_class, opts.document_class)
        self.assertIs(opts.with_options(document_class=dict).document_class, dict)

    def test_existing_document_classes_unchanged(self):
        self.assertIs(CodecOptions().document_class, dict)
        self.assertIs(CodecOptions(document_class=SON).document_class, SON)
        self.assertIs(CodecOptions(document_class=RawBSONDocument).document_class, RawBSONDocument)

    def test_validate_document_class_accepts_dataclass(self):
        self.assertIs(validate_document_class("document_class", UserDC), UserDC)
        self.assertIs(validate_document_class("document_class", ProtocolDoc), ProtocolDoc)
        self.assertIs(validate_document_class("document_class", dict), dict)
        with self.assertRaisesRegex(TypeError, "document_class must be dict"):
            validate_document_class("document_class", NotADocumentClass)

    def test_client_document_class_kwarg(self):
        client = self.simple_client(connect=False, document_class=UserDC)
        self.assertIs(client.codec_options.document_class, UserDC)
        # Client repr prints the user's class, not the private adapter.
        self.assertIn("UserDC", repr(client))
        self.assertNotIn("Adapter", repr(client))


class TestTypedDocumentClassIntegration(IntegrationTest):
    def setUp(self):
        super().setUp()
        self.coll = self.db.typed_docs
        self.coll.drop()
        self.ids = [ObjectId() for _ in range(10)]
        self.coll.insert_many(
            [{"_id": oid, "name": f"user{i}", "age": i} for i, oid in enumerate(self.ids)]
        )

    def typed_coll(self, cls):
        return self.db.get_collection("typed_docs", codec_options=CodecOptions(document_class=cls))

    def test_dataclass_find_one(self):
        user = self.typed_coll(UserDC).find_one({"name": "user3"})
        self.assertIsInstance(user, UserDC)
        self.assertEqual(user._id, self.ids[3])
        self.assertEqual(user.age, 3)

    def test_dataclass_find_multi_batch(self):
        users = self.typed_coll(UserDC).find(batch_size=3).to_list()
        self.assertEqual(len(users), 10)
        for user in users:
            self.assertIsInstance(user, UserDC)
        self.assertEqual({u.name for u in users}, {f"user{i}" for i in range(10)})

    def test_dataclass_aggregate(self):
        cursor = self.typed_coll(UserDC).aggregate(
            [{"$match": {"age": {"$gte": 5}}}, {"$sort": {"age": 1}}], batchSize=2
        )
        users = cursor.to_list()
        self.assertEqual([u.age for u in users], [5, 6, 7, 8, 9])
        for user in users:
            self.assertIsInstance(user, UserDC)

    def test_from_bson_class_find_and_aggregate(self):
        coll = self.typed_coll(ProtocolDoc)
        doc = coll.find_one({"name": "user0"})
        self.assertIsInstance(doc, ProtocolDoc)
        self.assertEqual(doc.fields["name"], "user0")
        docs = coll.find(batch_size=4).to_list()
        self.assertEqual(len(docs), 10)
        agg = (coll.aggregate([{"$sort": {"age": -1}}], batchSize=3)).to_list()
        self.assertEqual(agg[0].fields["age"], 9)
        self.assertTrue(all(isinstance(d, ProtocolDoc) for d in docs + agg))

    def test_bson_scalar_fidelity(self):
        @dataclass
        class Event:
            _id: ObjectId
            when: datetime.datetime

        coll = self.db.typed_events
        coll.drop()
        when = datetime.datetime(2026, 7, 22, 12, 0, 0)
        coll.insert_one({"_id": ObjectId(), "when": when})
        event = self.db.get_collection(
            "typed_events",
            codec_options=CodecOptions(document_class=Event),
        ).find_one()
        self.assertIsInstance(event._id, ObjectId)
        self.assertIsInstance(event.when, datetime.datetime)
        self.assertEqual(event.when, when)

    def test_getmore_envelope_across_batches(self):
        listener = OvertCommandListener()
        client = self.rs_or_single_client(event_listeners=[listener])
        coll = client.pymongo_test.get_collection(
            "typed_docs",
            codec_options=CodecOptions(document_class=UserDC),
        )
        users = coll.find(batch_size=3).to_list()
        self.assertEqual(len(users), 10)
        getmores = listener.started_command_names().count("getMore")
        self.assertGreaterEqual(getmores, 3)

    def test_explicit_session(self):
        with self.client.start_session() as session:
            users = self.typed_coll(UserDC).find(session=session).to_list()
        self.assertEqual(len(users), 10)
        self.assertIsInstance(users[0], UserDC)

    def test_command_error_raises_normally(self):
        with self.assertRaises(OperationFailure):
            self.typed_coll(UserDC).find({"$badOperator": 1}).to_list()

    def test_non_cursor_commands_work(self):
        coll = self.typed_coll(UserDC)
        result = coll.insert_one({"_id": ObjectId(), "name": "extra", "age": 99})
        self.assertTrue(result.acknowledged)
        update = coll.update_one({"name": "extra"}, {"$set": {"age": 100}})
        self.assertEqual(update.modified_count, 1)
        delete = coll.delete_one({"name": "extra"})
        self.assertEqual(delete.deleted_count, 1)

    def test_distinct_falls_back_to_plain_values(self):
        names = self.typed_coll(UserDC).distinct("name")
        self.assertEqual(sorted(names), sorted(f"user{i}" for i in range(10)))

    def test_find_one_and_update_returns_dict_poc_limitation(self):
        doc = self.typed_coll(UserDC).find_one_and_update({"name": "user1"}, {"$set": {"age": 42}})
        self.assertIsInstance(doc, dict)
        self.coll.update_one({"name": "user1"}, {"$set": {"age": 1}})

    def test_dict_son_raw_paths_unchanged(self):
        self.assertIsInstance(self.coll.find_one(), dict)
        son_coll = self.db.get_collection(
            "typed_docs", codec_options=CodecOptions(document_class=SON)
        )
        self.assertIsInstance(son_coll.find_one(), SON)
        raw_coll = self.db.get_collection(
            "typed_docs", codec_options=CodecOptions(document_class=RawBSONDocument)
        )
        self.assertIsInstance(raw_coll.find_one(), RawBSONDocument)

    def test_empty_result_set(self):
        users = self.typed_coll(UserDC).find({"name": "nobody"}).to_list()
        self.assertEqual(users, [])


class TestTypedDocumentWriteIntegration(IntegrationTest):
    """Write-path support for typed document instances."""

    def setUp(self):
        super().setUp()
        self.coll = self.db.typed_writes
        self.coll.drop()

    def typed_coll(self, cls):
        return self.db.get_collection(
            "typed_writes", codec_options=CodecOptions(document_class=cls)
        )

    def test_insert_one_typed_roundtrip(self):
        oid = ObjectId()
        coll = self.typed_coll(UserDC)
        result = coll.insert_one(UserDC(oid, "Ada", 36))
        self.assertEqual(result.inserted_id, oid)
        self.assertEqual(coll.find_one({"_id": oid}), UserDC(oid, "Ada", 36))

    def test_insert_one_generates_id_for_default_none(self):
        coll = self.typed_coll(AutoIdDC)
        user = AutoIdDC("Ada", 36)
        result = coll.insert_one(user)
        self.assertIsInstance(result.inserted_id, ObjectId)
        self.assertEqual(
            coll.find_one({"_id": result.inserted_id}),
            AutoIdDC("Ada", 36, result.inserted_id),
        )
        # The caller's instance is not mutated (no _id back-propagation).
        self.assertIsNone(user._id)
        # A second default-_id insert must not collide on _id: null.
        result2 = coll.insert_one(AutoIdDC("Bea", 25))
        self.assertNotEqual(result.inserted_id, result2.inserted_id)

    def test_insert_many_typed_and_plain_mixed(self):
        coll = self.typed_coll(UserDC)
        oids = [ObjectId() for _ in range(3)]
        result = coll.insert_many(
            [
                UserDC(oids[0], "user0", 0),
                {"_id": oids[1], "name": "user1", "age": 1},
                UserDC(oids[2], "user2", 2),
            ]
        )
        self.assertEqual(result.inserted_ids, oids)
        users = coll.find(sort=[("age", 1)]).to_list()
        self.assertEqual(users, [UserDC(oid, f"user{i}", i) for i, oid in enumerate(oids)])

    def test_plain_dict_insert_unchanged_on_typed_collection(self):
        # Mappings pass through untouched: the caller's dict still acquires
        # the generated _id.
        doc: dict[str, Any] = {"name": "Ada", "age": 36}
        result = self.typed_coll(UserDC).insert_one(doc)
        self.assertEqual(doc["_id"], result.inserted_id)

    def test_wrong_instance_type_rejected(self):
        coll = self.typed_coll(UserDC)
        with self.assertRaisesRegex(TypeError, "document must be an instance of dict"):
            coll.insert_one(NotADocumentClass())
        # An instance of a different dataclass is not silently encoded.
        with self.assertRaisesRegex(TypeError, "document must be an instance of dict"):
            coll.insert_one(OtherDC(ObjectId(), "red"))

    def test_untyped_collection_rejects_typed_instance(self):
        with self.assertRaisesRegex(TypeError, "document must be an instance of dict"):
            self.coll.insert_one(UserDC(ObjectId(), "Ada", 36))

    def test_replace_one_typed(self):
        coll = self.typed_coll(AutoIdDC)
        oid = (coll.insert_one(AutoIdDC("Ada", 36))).inserted_id
        result = coll.replace_one({"_id": oid}, AutoIdDC("Ada", 37))
        self.assertEqual(result.modified_count, 1)
        self.assertEqual(coll.find_one({"_id": oid}), AutoIdDC("Ada", 37, oid))

    def test_find_one_and_replace_typed(self):
        coll = self.typed_coll(AutoIdDC)
        oid = (coll.insert_one(AutoIdDC("Ada", 36))).inserted_id
        coll.find_one_and_replace({"_id": oid}, AutoIdDC("Bea", 20))
        self.assertEqual(coll.find_one({"_id": oid}), AutoIdDC("Bea", 20, oid))

    def test_find_one_and_update_rejects_typed_instance(self):
        # A typed instance is never a valid update spec ($ operators).
        coll = self.typed_coll(UserDC)
        with self.assertRaises((TypeError, ValueError)):
            coll.find_one_and_update({"name": "Ada"}, UserDC(ObjectId(), "Ada", 36))

    def test_bulk_write_typed_insert_and_replace(self):
        coll = self.typed_coll(UserDC)
        oids = [ObjectId() for _ in range(2)]
        result = coll.bulk_write(
            [
                InsertOne(UserDC(oids[0], "user0", 0)),
                InsertOne(UserDC(oids[1], "user1", 1)),
                ReplaceOne({"_id": oids[0]}, UserDC(oids[0], "user0", 100)),
            ]
        )
        self.assertEqual(result.inserted_count, 2)
        self.assertEqual(result.modified_count, 1)
        users = coll.find(sort=[("age", 1)]).to_list()
        self.assertEqual(users, [UserDC(oids[1], "user1", 1), UserDC(oids[0], "user0", 100)])

    @client_context.require_version_min(8, 0, 0, -24)
    def test_client_bulk_write_typed_insert(self):
        # Client-level bulk writes convert against the client's options.
        client = self.rs_or_single_client(document_class=UserDC)
        oid = ObjectId()
        result = client.bulk_write(
            [InsertOne(namespace=f"{self.db.name}.typed_writes", document=UserDC(oid, "Ada", 36))]
        )
        self.assertEqual(result.inserted_count, 1)
        self.assertEqual(self.coll.find_one({"_id": oid}), {"_id": oid, "name": "Ada", "age": 36})

    @client_context.require_version_min(8, 0, 0, -24)
    def test_client_bulk_write_typed_replace(self):
        client = self.rs_or_single_client(document_class=UserDC)
        oid = ObjectId()
        self.coll.insert_one({"_id": oid, "name": "Ada", "age": 36})
        result = client.bulk_write(
            [
                ReplaceOne(
                    namespace=f"{self.db.name}.typed_writes",
                    filter={"_id": oid},
                    replacement=UserDC(oid, "Ada", 37),
                )
            ]
        )
        self.assertEqual(result.modified_count, 1)
        self.assertEqual(self.coll.find_one({"_id": oid}), {"_id": oid, "name": "Ada", "age": 37})


@unittest.skipUnless(_HAVE_PYDANTIC, "pydantic v2 is not installed")
class TestPydanticIntegration(IntegrationTest):
    def setUp(self):
        super().setUp()
        self.coll = self.db.typed_docs_pydantic
        self.coll.drop()
        self.coll.insert_many(
            [{"_id": ObjectId(), "name": f"user{i}", "age": i} for i in range(10)]
        )
        self.typed = self.db.get_collection(
            "typed_docs_pydantic", codec_options=CodecOptions(document_class=UserModel)
        )

    def test_find_one(self):
        user = self.typed.find_one({"name": "user2"})
        self.assertIsInstance(user, UserModel)
        self.assertIsInstance(user.id, ObjectId)
        self.assertEqual(user.age, 2)

    def test_find_multi_batch(self):
        users = self.typed.find(batch_size=3).to_list()
        self.assertEqual(len(users), 10)
        self.assertTrue(all(isinstance(u, UserModel) for u in users))

    def test_aggregate(self):
        cursor = self.typed.aggregate([{"$sort": {"age": 1}}], batchSize=4)
        users = cursor.to_list()
        self.assertEqual([u.age for u in users], list(range(10)))
        self.assertTrue(all(isinstance(u, UserModel) for u in users))

    def test_validation_error_propagates_mid_batch(self):
        self.coll.insert_one({"_id": ObjectId(), "name": "bad", "age": "not-an-int"})
        with self.assertRaises(ValidationError):
            self.typed.find(batch_size=3).to_list()


@unittest.skipUnless(_HAVE_PYDANTIC, "pydantic v2 is not installed")
class TestPydanticWriteIntegration(IntegrationTest):
    def setUp(self):
        super().setUp()
        self.coll = self.db.typed_writes_pydantic
        self.coll.drop()
        self.typed = self.db.get_collection(
            "typed_writes_pydantic", codec_options=CodecOptions(document_class=UserModel)
        )

    def test_insert_one_roundtrip(self):
        oid = ObjectId()
        result = self.typed.insert_one(UserModel(id=oid, name="Ada", age=36))
        # The aliased id field is dumped under its wire name.
        self.assertEqual(result.inserted_id, oid)
        self.assertEqual(self.typed.find_one({"_id": oid}), UserModel(id=oid, name="Ada", age=36))

    def test_replace_one_typed(self):
        oid = ObjectId()
        self.typed.insert_one(UserModel(id=oid, name="Ada", age=36))
        result = self.typed.replace_one({"_id": oid}, UserModel(id=oid, name="Ada", age=37))
        self.assertEqual(result.modified_count, 1)
        user = self.typed.find_one({"_id": oid})
        self.assertEqual(user.age, 37)


if __name__ == "__main__":
    unittest.main()
