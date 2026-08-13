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
import io
import sys
import types
from collections.abc import MutableMapping
from dataclasses import InitVar, dataclass, field
from typing import Any, Optional
from unittest import mock

sys.path[0:0] = [""]

import bson
from bson import encode, json_util
from bson.adapters import (
    _BSON_DESERIALIZABLE_MARKER,
    _bson_deserializable_class,
    _convert_typed_document,
    _DataclassAdapter,
    _DocumentAdapter,
    _PydanticAdapter,
    _resolve_document_class,
)
from bson.binary import Binary
from bson.codec_options import CodecOptions, TypeDecoder, TypeRegistry
from bson.decimal128 import Decimal128
from bson.int64 import Int64
from bson.objectid import ObjectId
from bson.raw_bson import RawBSONDocument
from bson.son import SON
from gridfs.grid_file_shared import _clear_entity_type_registry
from gridfs.synchronous import GridFSBucket
from pymongo.common import validate_document_class
from pymongo.errors import OperationFailure
from pymongo.message import _unpack_typed_response
from pymongo.operations import InsertOne, ReplaceOne
from pymongo.synchronous.change_stream import CollectionChangeStream
from pymongo.synchronous.command_cursor import CommandCursor
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

    class AutoIdUserModel(BaseModel):
        """A round-trip-capable model whose ``_id`` defaults to None."""

        model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
        id: Optional[ObjectId] = Field(alias="_id", default=None)
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

    def test_marker_class_with_only_to_bson_rejected(self):
        class EncodeOnlyDoc:
            _type_marker = _BSON_DESERIALIZABLE_MARKER

            @classmethod
            def to_bson(cls, doc: Any, codec_options: CodecOptions[Any]) -> dict[str, Any]:
                return {}

        with self.assertRaisesRegex(TypeError, "does not implement from_bson"):
            CodecOptions(document_class=EncodeOnlyDoc)  # type: ignore[type-var]

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

    def test_extra_forbid_without_id_mapping_rejected(self):
        # extra='forbid' with no _id alias would make every decode raise
        # ValidationError on the reply's '_id' key; rejected at construction
        # exactly like the implicit-strict (extra unset) case.
        class Strict(BaseModel):
            model_config = ConfigDict(extra="forbid")
            x: int

        with self.assertRaisesRegex(TypeError, "extra='forbid'"):
            _PydanticAdapter(Strict)
        with self.assertRaisesRegex(TypeError, "extra='forbid'"):
            CodecOptions(document_class=Strict)

    def test_extra_forbid_with_id_mapping_allowed(self):
        # extra='forbid' plus an _id alias is the intended strict setup: known
        # keys decode, unknown keys still raise.
        class StrictWithId(BaseModel):
            model_config = ConfigDict(extra="forbid")
            id: Optional[int] = Field(alias="_id", default=None)
            x: int

        adapter = _PydanticAdapter(StrictWithId)
        self.assertEqual(adapter.from_bson({"_id": 1, "x": 2}, CodecOptions()).id, 1)
        with self.assertRaises(ValidationError):
            adapter.from_bson({"_id": 1, "x": 2, "stray": 3}, CodecOptions())

    def test_implicit_strict_decode_requires_validate_extra_kwarg(self):
        # On pydantic < 2.12 model_validate has no extra= argument, so the
        # driver cannot enforce strict decoding: unknown wire keys would be
        # silently dropped, then deleted by a read-modify-replace. A model
        # relying on implicit strictness is rejected at construction instead.
        class Old(BaseModel):
            id: Optional[int] = Field(alias="_id", default=None)

            @classmethod
            def model_validate(cls, obj: Any) -> Any:  # the pydantic < 2.12 signature
                return super().model_validate(obj)

        with self.assertRaisesRegex(TypeError, "pydantic"):
            _PydanticAdapter(Old)
        with self.assertRaisesRegex(TypeError, "pydantic"):
            CodecOptions(document_class=Old)

    def test_explicit_extra_works_without_validate_extra_kwarg(self):
        # An explicit extra policy is enforced by pydantic itself on every
        # version, so it must keep working without the extra= kwarg.
        class OldLenient(BaseModel):
            model_config = ConfigDict(extra="ignore")
            name: str

        class OldLenientCompat(OldLenient):
            @classmethod
            def model_validate(cls, obj: Any) -> Any:
                return super().model_validate(obj)

        adapter = _PydanticAdapter(OldLenientCompat)
        doc = adapter.from_bson({"_id": 1, "name": "x", "junk": 2}, CodecOptions())
        self.assertEqual(doc.name, "x")

    def test_to_bson_strips_none_id(self):
        adapter = _PydanticAdapter(AutoIdUserModel)
        self.assertEqual(
            adapter.to_bson(AutoIdUserModel(name="ada", age=1), CodecOptions()),
            {"name": "ada", "age": 1},
        )
        oid = ObjectId()
        self.assertEqual(
            adapter.to_bson(AutoIdUserModel(id=oid, name="ada", age=1), CodecOptions()),
            {"_id": oid, "name": "ada", "age": 1},
        )

    def test_validation_only_id_alias_cannot_write(self):
        class ViaValidationAlias(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
            ident: ObjectId = Field(validation_alias="_id")

        opts = CodecOptions(document_class=ViaValidationAlias)
        model = ViaValidationAlias(ident=ObjectId())
        with self.assertRaisesRegex(TypeError, "validation-only alias"):
            _convert_typed_document(model, opts)

    def test_serialization_alias_id_round_trips(self):
        class Symmetric(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
            ident: ObjectId = Field(validation_alias="_id", serialization_alias="_id")

        adapter = _PydanticAdapter(Symmetric)
        oid = ObjectId()
        self.assertEqual(adapter.to_bson(Symmetric(ident=oid), CodecOptions()), {"_id": oid})

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

    def test_protocol_class_with_own_document_type_attribute(self):
        # An ODM-style protocol class may use `document_type` for its own
        # purposes (a collection name, a related class); dispatch must not
        # mistake the attribute for the adapter contract: isinstance() against
        # a string crashes, against an unrelated class silently skips to_bson.
        class Unrelated:
            pass

        for attr in ("movies", Unrelated):
            with self.subTest(document_type=attr):

                class Movie:
                    _type_marker = _BSON_DESERIALIZABLE_MARKER
                    document_type = attr

                    def __init__(self, title: str) -> None:
                        self.title = title

                    @classmethod
                    def from_bson(
                        cls, doc: dict[str, Any], codec_options: CodecOptions[Any]
                    ) -> Movie:
                        return cls(doc["title"])

                    @classmethod
                    def to_bson(
                        cls, doc: Movie, codec_options: CodecOptions[Any]
                    ) -> dict[str, Any]:
                        return {"title": doc.title}

                opts = CodecOptions(document_class=Movie)
                self.assertEqual(_convert_typed_document(Movie("Alien"), opts), {"title": "Alien"})


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

    def test_dataclass_to_bson_converts_dataclasses_in_containers(self):
        @dataclass
        class Item:
            label: str

        @dataclass
        class Box:
            _id: ObjectId
            items: list[Item]
            by_name: dict[str, Item]
            pair: tuple[Item, Item]

        oid = ObjectId()
        box = Box(oid, [Item("a")], {"b": Item("b")}, (Item("c"), Item("d")))
        dumped = _DataclassAdapter(Box).to_bson(box, CodecOptions())
        self.assertEqual(
            dumped,
            {
                "_id": oid,
                "items": [{"label": "a"}],
                "by_name": {"b": {"label": "b"}},
                # Tuples dump as lists; both encode to BSON arrays.
                "pair": [{"label": "c"}, {"label": "d"}],
            },
        )
        encode(dumped)

    def test_dataclass_to_bson_shares_leaf_values(self):
        # Unlike dataclasses.asdict, leaf values are returned by reference,
        # never deep-copied: the BSON encoder only reads the dumped document,
        # and copying values like a large Binary payload would dominate the
        # cost of a write.
        @dataclass
        class Blob:
            _id: ObjectId
            payload: Binary
            created: datetime.datetime

        blob = Blob(
            ObjectId(),
            Binary(b"x" * 1024),
            datetime.datetime.now(datetime.timezone.utc),
        )
        dumped = _DataclassAdapter(Blob).to_bson(blob, CodecOptions())
        self.assertIs(dumped["_id"], blob._id)
        self.assertIs(dumped["payload"], blob.payload)
        self.assertIs(dumped["created"], blob.created)

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

    def test_dataclass_with_init_false_field_rejected(self):
        # to_bson writes every dataclass field, but from_bson calls cls(**doc),
        # which cannot accept an init=False field: every decode would crash.
        @dataclass
        class Metered:
            _id: Optional[ObjectId] = None
            total: int = field(init=False, default=0)

        with self.assertRaisesRegex(TypeError, "init=False"):
            _DataclassAdapter(Metered)
        with self.assertRaisesRegex(TypeError, "init=False"):
            CodecOptions(document_class=Metered)

    def test_dataclass_with_required_init_only_arg_rejected(self):
        # A required InitVar is demanded by cls(**doc) but never stored in
        # documents, so every decode would crash.
        @dataclass
        class Seeded:
            seed: InitVar[int]
            _id: Optional[ObjectId] = None

        with self.assertRaisesRegex(TypeError, "init-only"):
            _DataclassAdapter(Seeded)
        with self.assertRaisesRegex(TypeError, "init-only"):
            CodecOptions(document_class=Seeded)

    def test_dataclass_with_defaulted_init_var_allowed(self):
        # An InitVar with a default round-trips: to_bson never writes it and
        # cls(**doc) falls back to the default on decode.
        @dataclass
        class Tunable:
            _id: Optional[ObjectId] = None
            scale: InitVar[int] = 1

        adapter = _DataclassAdapter(Tunable)
        oid = ObjectId()
        self.assertEqual(adapter.from_bson({"_id": oid}, CodecOptions()), Tunable(_id=oid))

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

    def test_mapping_with_marker_but_no_hook_rejected_at_construction(self):
        # A mapping class would otherwise pass the mapping check here and
        # only blow up on the first decoded reply, deep in the message layer.
        class HooklessMap(dict):
            _type_marker = _BSON_DESERIALIZABLE_MARKER

        with self.assertRaisesRegex(TypeError, "does not implement from_bson"):
            CodecOptions(document_class=HooklessMap)

    def test_mapping_with_marker_and_hook_opts_into_typed_decoding(self):
        # Documents current semantics: a mapping class carrying the marker
        # and hook takes the typed decode path, not the mapping one.
        class MarkedMap(dict):
            _type_marker = _BSON_DESERIALIZABLE_MARKER

            @classmethod
            def from_bson(cls, doc: dict[str, Any], codec_options: CodecOptions[Any]) -> MarkedMap:
                return cls(doc)

        opts = CodecOptions(document_class=MarkedMap)
        self.assertIs(opts._document_adapter, MarkedMap)

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

    def test_database_aggregate_decodes_as_dicts(self):
        # Database-level pipelines return server metadata documents, which
        # must not decode into the typed document_class.
        client = self.rs_or_single_client(document_class=UserDC)
        cursor = client.admin.aggregate([{"$currentOp": {}}])
        ops = cursor.to_list()
        self.assertTrue(ops)
        for op in ops:
            self.assertIsInstance(op, dict)

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

    def test_insert_one_generates_id_when_none(self):
        typed = self.db.get_collection(
            "typed_writes_pydantic", codec_options=CodecOptions(document_class=AutoIdUserModel)
        )
        result = typed.insert_one(AutoIdUserModel(name="Ada", age=36))
        self.assertIsInstance(result.inserted_id, ObjectId)
        user = typed.find_one({"_id": result.inserted_id})
        self.assertEqual(user.name, "Ada")


class _ToDecimalDecoder(TypeDecoder):
    bson_type = Decimal128

    def transform_bson(self, value: Decimal128) -> Any:
        return value.to_decimal()


class TestTypedChangeStreamUnit(UnitTest):
    def _change_stream(self, coll):
        # (target, pipeline, full_document, resume_after, max_await_time_ms,
        #  batch_size, collation, start_at_operation_time, session, start_after)
        return CollectionChangeStream(coll, None, None, None, None, None, None, None, None, None)

    def test_typed_document_class_is_replaced_with_dict(self):
        client = self.simple_client(connect=False)
        coll = client.db.get_collection(
            "typed_cs", codec_options=CodecOptions(document_class=UserDC)
        )
        cs = self._change_stream(coll)
        self.assertIs(cs._target.codec_options.document_class, dict)
        self.assertIs(cs._orig_codec_options.document_class, dict)
        self.assertFalse(cs._decode_custom)

    def test_typed_document_class_with_custom_type_registry(self):
        client = self.simple_client(connect=False)
        codec_options = CodecOptions(
            document_class=UserDC, type_registry=TypeRegistry([_ToDecimalDecoder()])
        )
        coll = client.db.get_collection("typed_cs", codec_options=codec_options)
        cs = self._change_stream(coll)
        # The custom-type-registry path re-decodes each raw change document
        # with _orig_codec_options, which must not be typed.
        self.assertTrue(cs._decode_custom)
        self.assertIs(cs._orig_codec_options.document_class, dict)
        self.assertIs(cs._target.codec_options.document_class, RawBSONDocument)


class TestTypedChangeStream(IntegrationTest):
    @client_context.require_change_streams
    def test_watch_with_typed_document_class_yields_dict_envelopes(self):
        coll = self.db.get_collection("typed_cs", codec_options=CodecOptions(document_class=UserDC))
        coll.drop()
        oid = ObjectId()
        with coll.watch(max_await_time_ms=250) as stream:
            coll.insert_one({"_id": oid, "name": "ada", "age": 1})
            change = stream.next()
            self.assertIsInstance(change, dict)
            self.assertEqual(change["operationType"], "insert")
            self.assertEqual(change["fullDocument"], {"_id": oid, "name": "ada", "age": 1})
            self.assertIsNotNone(stream.resume_token)


class _IdentityDecryptEncrypter:
    """Stands in for _Encrypter(bypass_auto_encryption=True); decrypt is identity."""

    _bypass_auto_encryption = True

    def decrypt(self, response: bytes) -> bytes:
        return bytes(response)

    def close(self) -> None:
        pass


class TestTypedDocumentAutoDecryption(IntegrationTest):
    def test_command_reply_decoded_typed_with_encrypter(self):
        coll = self.db.typed_decrypt
        coll.drop()
        oid = ObjectId()
        coll.insert_one({"_id": oid, "name": "ada", "age": 1})

        client = self.rs_or_single_client()
        client._encrypter = _IdentityDecryptEncrypter()
        typed = client[self.db.name].get_collection(
            "typed_decrypt", codec_options=CodecOptions(document_class=UserDC)
        )
        cursor = typed.aggregate([])
        self.assertEqual(cursor.to_list(), [UserDC(_id=oid, name="ada", age=1)])


class TestTypedGridFS(IntegrationTest):
    def test_gridfs_ignores_typed_document_class(self):
        db = self.client.get_database(
            self.db.name, codec_options=CodecOptions(document_class=UserDC)
        )
        bucket = GridFSBucket(db)
        oid = bucket.upload_from_stream("hello.txt", b"hello world")
        stream = bucket.open_download_stream(oid)
        self.assertEqual(stream.read(), b"hello world")

    def test_clear_entity_type_registry_replaces_typed_document_class(self):
        client = self.simple_client(connect=False)
        coll = client.db.get_collection(
            "fs.files", codec_options=CodecOptions(document_class=UserDC)
        )
        cleared = _clear_entity_type_registry(coll)
        self.assertIs(cleared.codec_options.document_class, dict)


class TestTypedClientBulkWrite(IntegrationTest):
    @client_context.require_version_min(8, 0, 0, -24)
    def test_results_cursor_decodes_as_dict(self):
        client = self.rs_or_single_client(document_class=UserDC)
        captured = []
        real_cursor = CommandCursor

        def capture(coll, *args, **kwargs):
            captured.append(coll)
            return real_cursor(coll, *args, **kwargs)

        with mock.patch("pymongo.client_bulk.CommandCursor", side_effect=capture):
            result = client.bulk_write(
                [InsertOne(namespace=f"{self.db.name}.typed_bulk", document={"i": 1})],
                verbose_results=True,
            )
        self.assertEqual(result.inserted_count, 1)
        self.assertEqual(len(captured), 1)
        # getMore replies on the results cursor decode with this collection's
        # codec options; a typed document_class cannot describe per-op results.
        self.assertIs(captured[0].codec_options.document_class, dict)


class TestTypedDatabaseCursors(IntegrationTest):
    def setUp(self):
        super().setUp()
        self.typed_db = self.client.get_database(
            self.db.name, codec_options=CodecOptions(document_class=UserDC)
        )

    def test_list_collections_multi_batch(self):
        self.db.typed_lc_a.insert_one({})
        self.db.typed_lc_b.insert_one({})
        names = self.typed_db.list_collection_names(cursor={"batchSize": 1})
        self.assertIn("typed_lc_a", names)
        self.assertIn("typed_lc_b", names)

    def test_cursor_command_multi_batch(self):
        coll = self.db.typed_cursor_cmd
        coll.drop()
        coll.insert_many([{"i": i} for i in range(4)])
        cursor = self.typed_db.cursor_command("find", coll.name, batchSize=2)
        docs = cursor.to_list()
        self.assertEqual(sorted(d["i"] for d in docs), [0, 1, 2, 3])

    def test_command_rejects_typed_codec_options_param(self):
        # command decodes its reply with the given options and has no
        # cursor batch a typed class could describe; the requested type
        # must fail loudly instead of silently returning dicts.
        with self.assertRaisesRegex(TypeError, "not supported by command"):
            # The static rejection matches the runtime one.
            self.db.command(
                "ping",
                codec_options=CodecOptions(document_class=UserDC),  # type: ignore[type-var]
            )

    def test_cursor_command_rejects_typed_codec_options_param(self):
        with self.assertRaisesRegex(TypeError, "not supported by cursor_command"):
            # The static rejection matches the runtime one.
            self.db.cursor_command(
                "find",
                "typed_cursor_cmd",
                codec_options=CodecOptions(document_class=UserDC),  # type: ignore[type-var]
            )


class TestStandaloneBsonRejectsTyped(UnitTest):
    def test_standalone_decode_apis_reject_typed_options(self):
        # The decode APIs statically reject typed options; these calls
        # deliberately bypass mypy to pin the runtime rejection.
        data = encode({"a": 1})
        opts = CodecOptions(document_class=UserDC)
        with self.assertRaisesRegex(TypeError, "mapping document_class"):
            bson.decode(data, opts)  # type: ignore[type-var]
        with self.assertRaisesRegex(TypeError, "mapping document_class"):
            bson.decode_all(data, opts)  # type: ignore[type-var]
        with self.assertRaisesRegex(TypeError, "mapping document_class"):
            next(bson.decode_iter(data, opts))  # type: ignore[type-var]
        with self.assertRaisesRegex(TypeError, "mapping document_class"):
            next(bson.decode_file_iter(io.BytesIO(data), opts))  # type: ignore[type-var]


class TestJSONOptionsRejectsTyped(UnitTest):
    def test_json_options_rejects_typed_document_class(self):
        with self.assertRaisesRegex(TypeError, "typed document classes"):
            json_util.JSONOptions(document_class=UserDC)


if __name__ == "__main__":
    unittest.main()
