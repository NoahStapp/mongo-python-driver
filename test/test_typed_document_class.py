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
    BsonConstructPlan,
    _bson_deserializable_class,
    _DataclassAdapter,
    _decode_typed_batch,
    _PydanticAdapter,
    _resolve_document_class,
)
from bson.codec_options import CodecOptions
from bson.objectid import ObjectId
from bson.raw_bson import RawBSONDocument
from bson.son import SON
from pymongo.common import validate_document_class
from pymongo.errors import OperationFailure
from test import IntegrationTest, UnitTest, unittest
from test.utils_shared import OvertCommandListener

_IS_SYNC = True


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
    def from_bson(cls, data: Any, codec_options: CodecOptions[Any]) -> ProtocolDoc:
        return cls(bson.decode(data, codec_options.with_options(document_class=dict)))


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


class TestDocumentClassResolution(UnitTest):
    def test_protocol_class_used_as_is(self):
        self.assertIs(_resolve_document_class(ProtocolDoc), ProtocolDoc)

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


class TestAdapters(UnitTest):
    def test_dataclass_adapter_from_bson(self):
        oid = ObjectId()
        data = encode({"_id": oid, "name": "Ada", "age": 36})
        adapter = _DataclassAdapter(UserDC)
        user = adapter.from_bson(data, CodecOptions())
        self.assertIsInstance(user, UserDC)
        self.assertEqual(user._id, oid)
        self.assertEqual(user.name, "Ada")
        self.assertEqual(user.age, 36)

    def test_dataclass_adapter_accepts_memoryview(self):
        data = memoryview(encode({"_id": ObjectId(), "name": "Ada", "age": 36}))
        user = _DataclassAdapter(UserDC).from_bson(data, CodecOptions())
        self.assertIsInstance(user, UserDC)

    def test_adapter_eq_hash_repr(self):
        a, b = _DataclassAdapter(UserDC), _DataclassAdapter(UserDC)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, _PydanticAdapter(UserDC))
        self.assertIn("UserDC", repr(a))

    def test_dataclass_adapter_from_bson_batch(self):
        oids = [ObjectId() for _ in range(3)]
        buffer = b"".join(
            encode({"_id": oid, "name": f"user{i}", "age": i}) for i, oid in enumerate(oids)
        )
        users = _DataclassAdapter(UserDC).from_bson_batch(buffer, CodecOptions())
        self.assertEqual(len(users), 3)
        for i, user in enumerate(users):
            self.assertIsInstance(user, UserDC)
            self.assertEqual(user._id, oids[i])
            self.assertEqual(user.age, i)

    def test_from_bson_batch_empty_buffer(self):
        self.assertEqual(_DataclassAdapter(UserDC).from_bson_batch(b"", CodecOptions()), [])


class TestDecodeTypedBatch(UnitTest):
    def setUp(self):
        super().setUp()
        self.buffer = b"".join(
            encode({"_id": ObjectId(), "name": f"user{i}", "age": i}) for i in range(3)
        )

    def test_uses_from_bson_batch_when_available(self):
        # _PydanticAdapter advertises no construct plan, so batch decoding
        # goes through its from_bson_batch hook.
        class FakeModel:
            @classmethod
            def model_validate(cls, obj: Any) -> Any:
                return obj

        adapter = _PydanticAdapter(FakeModel)
        with mock.patch.object(
            adapter, "from_bson_batch", wraps=adapter.from_bson_batch
        ) as batch_hook:
            users = _decode_typed_batch(adapter, self.buffer, CodecOptions())
        batch_hook.assert_called_once()
        self.assertEqual([u["age"] for u in users], [0, 1, 2])

    def test_falls_back_to_per_document_from_bson(self):
        # ProtocolDoc implements only from_bson: each document must arrive
        # as its own raw BSON slice.
        docs = _decode_typed_batch(ProtocolDoc, self.buffer, CodecOptions())
        self.assertEqual(len(docs), 3)
        for i, doc in enumerate(docs):
            self.assertIsInstance(doc, ProtocolDoc)
            self.assertEqual(doc.fields["age"], i)

    def test_protocol_class_may_implement_from_bson_batch(self):
        class BatchingProtocolDoc(ProtocolDoc):
            batch_calls = 0

            @classmethod
            def from_bson_batch(cls, data: Any, codec_options: CodecOptions[Any]) -> Any:
                cls.batch_calls += 1
                return bson.decode_all(data, codec_options.with_options(document_class=dict))

        out = _decode_typed_batch(BatchingProtocolDoc, self.buffer, CodecOptions())
        self.assertEqual(BatchingProtocolDoc.batch_calls, 1)
        self.assertEqual([d["age"] for d in out], [0, 1, 2])

    def test_empty_buffer(self):
        self.assertEqual(_decode_typed_batch(ProtocolDoc, b"", CodecOptions()), [])


class PlannedDoc:
    """A hand-rolled class advertising a static construction plan."""

    _type_marker = _BSON_DESERIALIZABLE_MARKER

    def __init__(self, _id: ObjectId, name: str, age: int = -1) -> None:
        self._id = _id
        self.name = name
        self.age = age

    @classmethod
    def __bson_construct_plan__(cls) -> BsonConstructPlan:
        return BsonConstructPlan(fields=("_id", "name", "age"))

    @classmethod
    def from_bson(cls, data: Any, codec_options: CodecOptions[Any]) -> PlannedDoc:
        raise AssertionError("plan-based decoding must not fall back to from_bson")


class TestConstructPlan(UnitTest):
    def setUp(self):
        super().setUp()
        self.oids = [ObjectId() for _ in range(3)]
        self.buffer = b"".join(
            encode({"_id": oid, "name": f"user{i}", "age": i}) for i, oid in enumerate(self.oids)
        )

    def test_dataclass_adapter_advertises_plan(self):
        plan = _DataclassAdapter(UserDC).__bson_construct_plan__()
        self.assertEqual(plan.fields, ("_id", "name", "age"))
        self.assertIs(plan.factory, UserDC)
        self.assertEqual(plan.strategy, "call")

    def test_plan_preferred_over_from_bson_batch(self):
        adapter = _DataclassAdapter(UserDC)
        with mock.patch.object(adapter, "from_bson_batch") as batch_hook:
            users = _decode_typed_batch(adapter, self.buffer, CodecOptions())
        batch_hook.assert_not_called()
        self.assertEqual(len(users), 3)
        for i, user in enumerate(users):
            self.assertIsInstance(user, UserDC)
            self.assertEqual(user._id, self.oids[i])
            self.assertEqual(user.age, i)

    def test_hand_rolled_plan_call_strategy(self):
        docs = _decode_typed_batch(PlannedDoc, self.buffer, CodecOptions())
        self.assertEqual(len(docs), 3)
        for i, doc in enumerate(docs):
            self.assertIsInstance(doc, PlannedDoc)
            self.assertEqual(doc._id, self.oids[i])
            self.assertEqual(doc.name, f"user{i}")
            self.assertEqual(doc.age, i)

    def test_plan_param_renaming(self):
        class RenamedDoc:
            _type_marker = _BSON_DESERIALIZABLE_MARKER

            def __init__(self, id: ObjectId, name: str, age: int) -> None:
                self.id = id
                self.name = name
                self.age = age

            @classmethod
            def __bson_construct_plan__(cls) -> BsonConstructPlan:
                return BsonConstructPlan(
                    fields=("_id", "name", "age"), params=("id", "name", "age")
                )

        docs = _decode_typed_batch(RenamedDoc, self.buffer, CodecOptions())
        self.assertEqual([d.id for d in docs], self.oids)

    def test_plan_extra_keys_raise_type_error(self):
        # Same semantics as cls(**decoded): unexpected wire keys are passed
        # through and the constructor rejects them.
        buffer = encode({"_id": ObjectId(), "name": "x", "age": 1, "extra": True})
        with self.assertRaises(TypeError):
            _decode_typed_batch(PlannedDoc, buffer, CodecOptions())

    def test_plan_missing_keys_use_constructor_defaults(self):
        buffer = encode({"_id": ObjectId(), "name": "x"})
        (doc,) = _decode_typed_batch(PlannedDoc, buffer, CodecOptions())
        self.assertEqual(doc.age, -1)

    def test_setattr_strategy_skips_init(self):
        class SetattrDoc:
            _type_marker = _BSON_DESERIALIZABLE_MARKER

            def __init__(self) -> None:
                raise AssertionError("__init__ must not run under the setattr strategy")

            @classmethod
            def __bson_construct_plan__(cls) -> BsonConstructPlan:
                return BsonConstructPlan(fields=("_id", "name", "age"), strategy="setattr")

        docs = _decode_typed_batch(SetattrDoc, self.buffer, CodecOptions())
        self.assertEqual(len(docs), 3)
        for i, doc in enumerate(docs):
            self.assertIsInstance(doc, SetattrDoc)
            self.assertEqual(doc.name, f"user{i}")
            self.assertEqual(doc.age, i)

    def test_unknown_strategy_raises(self):
        class BadPlanDoc:
            _type_marker = _BSON_DESERIALIZABLE_MARKER

            @classmethod
            def __bson_construct_plan__(cls) -> BsonConstructPlan:
                return BsonConstructPlan(fields=("_id",), strategy="bogus")

        with self.assertRaisesRegex(ValueError, "bogus"):
            _decode_typed_batch(BadPlanDoc, self.buffer, CodecOptions())

    def test_empty_buffer(self):
        self.assertEqual(_decode_typed_batch(PlannedDoc, b"", CodecOptions()), [])

    def test_bson_scalars_survive_plan_path(self):
        oid = ObjectId()
        when = datetime.datetime(2026, 7, 23, 12, 0, 0)

        @dataclass
        class Event:
            _id: ObjectId
            when: datetime.datetime

        (event,) = _decode_typed_batch(
            _DataclassAdapter(Event), encode({"_id": oid, "when": when}), CodecOptions()
        )
        self.assertEqual(event._id, oid)
        self.assertEqual(event.when, when)


class TestCodecOptionsGate(UnitTest):
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

    def test_protocol_class_find_and_aggregate(self):
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
        event = self.db.get_collection(  # type: ignore[type-var]
            "typed_events",
            codec_options=CodecOptions(document_class=Event),  # type: ignore[type-var]
        ).find_one()
        self.assertIsInstance(event._id, ObjectId)
        self.assertIsInstance(event.when, datetime.datetime)
        self.assertEqual(event.when, when)

    def test_getmore_envelope_across_batches(self):
        listener = OvertCommandListener()
        client = self.rs_or_single_client(event_listeners=[listener])
        coll = client.pymongo_test.get_collection(  # type: ignore[type-var]
            "typed_docs",
            codec_options=CodecOptions(document_class=UserDC),  # type: ignore[type-var]
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


if __name__ == "__main__":
    unittest.main()
