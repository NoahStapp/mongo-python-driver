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
import sys
import types
from dataclasses import dataclass
from typing import Any
from unittest import mock

sys.path[0:0] = [""]

import bson
from bson import encode
from bson.adapters import (
    _BSON_DESERIALIZABLE_MARKER,
    _bson_deserializable_class,
    _DataclassAdapter,
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
from test.asynchronous import AsyncIntegrationTest, AsyncUnitTest, unittest
from test.utils_shared import OvertCommandListener

_IS_SYNC = False


@dataclass
class UserDC:
    _id: ObjectId
    name: str
    age: int


class ProtocolDoc:
    """A hand-rolled implementation of the from_bson protocol."""

    _type_marker = _BSON_DESERIALIZABLE_MARKER

    def __init__(self, fields: dict[str, Any]) -> None:
        self.fields = fields

    @classmethod
    def from_bson(cls, doc: dict[str, Any], codec_options: CodecOptions[Any]) -> ProtocolDoc:
        return cls(doc)


class NotADocumentClass:
    pass


try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError

    _HAVE_PYDANTIC = True

    class UserModel(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
        id: ObjectId = Field(alias="_id")
        name: str
        age: int

except ImportError:
    _HAVE_PYDANTIC = False


class TestDocumentClassResolution(AsyncUnitTest):
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


class TestAdapters(AsyncUnitTest):
    def test_adapter_eq_hash_repr(self):
        a, b = _DataclassAdapter(UserDC), _DataclassAdapter(UserDC)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, _PydanticAdapter(UserDC))
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
            @classmethod
            def model_validate(cls, obj: Any) -> Any:
                return ("validated", obj["name"])

        self.assertEqual(
            _PydanticAdapter(FakeModel).from_bson({"name": "x"}, CodecOptions()),
            ("validated", "x"),
        )


class TestFromBsonHook(AsyncUnitTest):
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


class TestUnpackTypedResponseSinglePass(AsyncUnitTest):
    """Reply unpacking for typed document classes.

    The reply is decoded to dicts in a single pass exactly like the
    plain-dict path (envelope included) and the batch documents are
    replaced with constructed instances.
    """

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
        opts = CodecOptions(document_class=UserDC)  # type: ignore[type-var]
        (envelope,) = _unpack_typed_response(self._payload(), opts, self.USER_FIELDS)
        self.assertEqual(
            envelope["cursor"]["firstBatch"],
            [UserDC(oid, f"user{i}", i) for i, oid in enumerate(self.oids)],
        )

    def test_envelope_subdocuments_are_plain_dicts(self):
        # Non-user envelope fields must decode like the plain-dict path:
        # nested documents (e.g. postBatchResumeToken) come back as dicts,
        # not RawBSONDocument.
        opts = CodecOptions(document_class=UserDC)  # type: ignore[type-var]
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
        opts = CodecOptions(document_class=UserDC)  # type: ignore[type-var]
        payload = encode(
            {"cursor": {"nextBatch": self.batch, "id": Int64(0), "ns": "db.coll"}, "ok": 1.0}
        )
        (envelope,) = _unpack_typed_response(payload, opts, {"cursor": {"nextBatch": 1}})
        batch = envelope["cursor"]["nextBatch"]
        self.assertEqual(len(batch), 2)
        self.assertIsInstance(batch[0], UserDC)

    def test_from_bson_class_takes_single_pass(self):
        opts = CodecOptions(document_class=ProtocolDoc)  # type: ignore[type-var]
        (envelope,) = _unpack_typed_response(self._payload(), opts, self.USER_FIELDS)
        docs = envelope["cursor"]["firstBatch"]
        self.assertEqual([doc.fields["age"] for doc in docs], [0, 1])
        self.assertIsInstance(docs[0], ProtocolDoc)

    def test_hook_called_once_per_batch_document(self):
        opts = CodecOptions(document_class=UserDC)  # type: ignore[type-var]
        adapter = opts.document_class
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

        opts = CodecOptions(document_class=CapturingDoc)  # type: ignore[type-var]
        _unpack_typed_response(self._payload(), opts, self.USER_FIELDS)
        self.assertEqual(len(seen), len(self.batch))
        for received in seen:
            self.assertIs(received, opts)


class TestCodecOptionsGate(AsyncUnitTest):
    def test_dataclass_document_class(self):
        opts = CodecOptions(document_class=UserDC)  # type: ignore[type-var]
        self.assertIsInstance(opts.document_class, _DataclassAdapter)
        self.assertIs(opts.document_class.document_type, UserDC)

    def test_protocol_document_class(self):
        opts = CodecOptions(document_class=ProtocolDoc)  # type: ignore[type-var]
        self.assertIs(opts.document_class, ProtocolDoc)

    def test_unsupported_class_still_raises(self):
        with self.assertRaisesRegex(TypeError, "document_class must be dict"):
            CodecOptions(document_class=NotADocumentClass)  # type: ignore[type-var]

    def test_marker_without_hook_rejected_at_construction(self):
        class HooklessDoc:
            _type_marker = _BSON_DESERIALIZABLE_MARKER

        with self.assertRaisesRegex(TypeError, "does not implement from_bson"):
            CodecOptions(document_class=HooklessDoc)  # type: ignore[type-var]

    def test_equality_and_repr(self):
        self.assertEqual(CodecOptions(document_class=UserDC), CodecOptions(document_class=UserDC))
        self.assertNotEqual(CodecOptions(document_class=UserDC), CodecOptions())
        self.assertIn("UserDC", repr(CodecOptions(document_class=UserDC)))  # type: ignore[type-var]

    def test_with_options_round_trip(self):
        opts = CodecOptions(document_class=UserDC)  # type: ignore[type-var]
        self.assertEqual(opts.with_options(tz_aware=True).document_class, opts.document_class)
        self.assertIs(opts.with_options(document_class=dict).document_class, dict)

    def test_existing_document_classes_unchanged(self):
        self.assertIs(CodecOptions().document_class, dict)
        self.assertIs(CodecOptions(document_class=SON).document_class, SON)
        self.assertIs(CodecOptions(document_class=RawBSONDocument).document_class, RawBSONDocument)

    def test_validate_document_class_accepts_dataclass(self):
        resolved = validate_document_class("document_class", UserDC)
        self.assertIsInstance(resolved, _DataclassAdapter)
        self.assertIs(validate_document_class("document_class", ProtocolDoc), ProtocolDoc)
        self.assertIs(validate_document_class("document_class", dict), dict)
        with self.assertRaisesRegex(TypeError, "document_class must be dict"):
            validate_document_class("document_class", NotADocumentClass)

    def test_client_document_class_kwarg(self):
        client = self.simple_client(connect=False, document_class=UserDC)
        self.assertIsInstance(client.codec_options.document_class, _DataclassAdapter)
        # Client repr must not crash with an adapter document_class.
        repr(client)


class TestTypedDocumentClassIntegration(AsyncIntegrationTest):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.coll = self.db.typed_docs
        await self.coll.drop()
        self.ids = [ObjectId() for _ in range(10)]
        await self.coll.insert_many(
            [{"_id": oid, "name": f"user{i}", "age": i} for i, oid in enumerate(self.ids)]
        )

    def typed_coll(self, cls):
        return self.db.get_collection("typed_docs", codec_options=CodecOptions(document_class=cls))

    async def test_dataclass_find_one(self):
        user = await self.typed_coll(UserDC).find_one({"name": "user3"})
        self.assertIsInstance(user, UserDC)
        self.assertEqual(user._id, self.ids[3])
        self.assertEqual(user.age, 3)

    async def test_dataclass_find_multi_batch(self):
        users = await self.typed_coll(UserDC).find(batch_size=3).to_list()
        self.assertEqual(len(users), 10)
        for user in users:
            self.assertIsInstance(user, UserDC)
        self.assertEqual({u.name for u in users}, {f"user{i}" for i in range(10)})

    async def test_dataclass_aggregate(self):
        cursor = await self.typed_coll(UserDC).aggregate(
            [{"$match": {"age": {"$gte": 5}}}, {"$sort": {"age": 1}}], batchSize=2
        )
        users = await cursor.to_list()
        self.assertEqual([u.age for u in users], [5, 6, 7, 8, 9])
        for user in users:
            self.assertIsInstance(user, UserDC)

    async def test_from_bson_class_find_and_aggregate(self):
        coll = self.typed_coll(ProtocolDoc)
        doc = await coll.find_one({"name": "user0"})
        self.assertIsInstance(doc, ProtocolDoc)
        self.assertEqual(doc.fields["name"], "user0")
        docs = await coll.find(batch_size=4).to_list()
        self.assertEqual(len(docs), 10)
        agg = await (await coll.aggregate([{"$sort": {"age": -1}}], batchSize=3)).to_list()
        self.assertEqual(agg[0].fields["age"], 9)
        self.assertTrue(all(isinstance(d, ProtocolDoc) for d in docs + agg))

    async def test_bson_scalar_fidelity(self):
        @dataclass
        class Event:
            _id: ObjectId
            when: datetime.datetime

        coll = self.db.typed_events
        await coll.drop()
        when = datetime.datetime(2026, 7, 22, 12, 0, 0)
        await coll.insert_one({"_id": ObjectId(), "when": when})
        event = await self.db.get_collection(  # type: ignore[type-var]
            "typed_events",
            codec_options=CodecOptions(document_class=Event),  # type: ignore[type-var]
        ).find_one()
        self.assertIsInstance(event._id, ObjectId)
        self.assertIsInstance(event.when, datetime.datetime)
        self.assertEqual(event.when, when)

    async def test_getmore_envelope_across_batches(self):
        listener = OvertCommandListener()
        client = await self.async_rs_or_single_client(event_listeners=[listener])
        coll = client.pymongo_test.get_collection(  # type: ignore[type-var]
            "typed_docs",
            codec_options=CodecOptions(document_class=UserDC),  # type: ignore[type-var]
        )
        users = await coll.find(batch_size=3).to_list()
        self.assertEqual(len(users), 10)
        getmores = listener.started_command_names().count("getMore")
        self.assertGreaterEqual(getmores, 3)

    async def test_explicit_session(self):
        async with self.client.start_session() as session:
            users = await self.typed_coll(UserDC).find(session=session).to_list()
        self.assertEqual(len(users), 10)
        self.assertIsInstance(users[0], UserDC)

    async def test_command_error_raises_normally(self):
        with self.assertRaises(OperationFailure):
            await self.typed_coll(UserDC).find({"$badOperator": 1}).to_list()

    async def test_non_cursor_commands_work(self):
        coll = self.typed_coll(UserDC)
        result = await coll.insert_one({"_id": ObjectId(), "name": "extra", "age": 99})
        self.assertTrue(result.acknowledged)
        update = await coll.update_one({"name": "extra"}, {"$set": {"age": 100}})
        self.assertEqual(update.modified_count, 1)
        delete = await coll.delete_one({"name": "extra"})
        self.assertEqual(delete.deleted_count, 1)

    async def test_distinct_falls_back_to_plain_values(self):
        names = await self.typed_coll(UserDC).distinct("name")
        self.assertEqual(sorted(names), sorted(f"user{i}" for i in range(10)))

    async def test_find_one_and_update_returns_dict_poc_limitation(self):
        doc = await self.typed_coll(UserDC).find_one_and_update(
            {"name": "user1"}, {"$set": {"age": 42}}
        )
        self.assertIsInstance(doc, dict)
        await self.coll.update_one({"name": "user1"}, {"$set": {"age": 1}})

    async def test_dict_son_raw_paths_unchanged(self):
        self.assertIsInstance(await self.coll.find_one(), dict)
        son_coll = self.db.get_collection(
            "typed_docs", codec_options=CodecOptions(document_class=SON)
        )
        self.assertIsInstance(await son_coll.find_one(), SON)
        raw_coll = self.db.get_collection(
            "typed_docs", codec_options=CodecOptions(document_class=RawBSONDocument)
        )
        self.assertIsInstance(await raw_coll.find_one(), RawBSONDocument)

    async def test_empty_result_set(self):
        users = await self.typed_coll(UserDC).find({"name": "nobody"}).to_list()
        self.assertEqual(users, [])


@unittest.skipUnless(_HAVE_PYDANTIC, "pydantic v2 is not installed")
class TestPydanticIntegration(AsyncIntegrationTest):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.coll = self.db.typed_docs_pydantic
        await self.coll.drop()
        await self.coll.insert_many(
            [{"_id": ObjectId(), "name": f"user{i}", "age": i} for i in range(10)]
        )
        self.typed = self.db.get_collection(
            "typed_docs_pydantic", codec_options=CodecOptions(document_class=UserModel)
        )

    async def test_find_one(self):
        user = await self.typed.find_one({"name": "user2"})
        self.assertIsInstance(user, UserModel)
        self.assertIsInstance(user.id, ObjectId)
        self.assertEqual(user.age, 2)

    async def test_find_multi_batch(self):
        users = await self.typed.find(batch_size=3).to_list()
        self.assertEqual(len(users), 10)
        self.assertTrue(all(isinstance(u, UserModel) for u in users))

    async def test_aggregate(self):
        cursor = await self.typed.aggregate([{"$sort": {"age": 1}}], batchSize=4)
        users = await cursor.to_list()
        self.assertEqual([u.age for u in users], list(range(10)))
        self.assertTrue(all(isinstance(u, UserModel) for u in users))

    async def test_validation_error_propagates_mid_batch(self):
        await self.coll.insert_one({"_id": ObjectId(), "name": "bad", "age": "not-an-int"})
        with self.assertRaises(ValidationError):
            await self.typed.find(batch_size=3).to_list()


if __name__ == "__main__":
    unittest.main()
