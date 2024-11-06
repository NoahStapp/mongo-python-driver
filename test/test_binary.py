# Copyright 2009-present MongoDB, Inc.
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

"""Tests for the Binary wrapper."""
from __future__ import annotations

import array
import base64
import copy
import mmap
import pickle
import sys
import uuid
from typing import Any

sys.path[0:0] = [""]

from test import IntegrationTest, client_context, unittest

import bson
from bson import decode, encode
from bson.binary import *
from bson.codec_options import CodecOptions
from bson.son import SON
from pymongo.common import validate_uuid_representation
from pymongo.write_concern import WriteConcern


class BinaryData:
    # Generated by the Java driver
    from_java = (
        b"bAAAAAdfaWQAUCBQxkVm+XdxJ9tOBW5ld2d1aWQAEAAAAAMIQkfACFu"
        b"Z/0RustLOU/G6Am5ld2d1aWRzdHJpbmcAJQAAAGZmOTk1YjA4LWMwND"
        b"ctNDIwOC1iYWYxLTUzY2VkMmIyNmU0NAAAbAAAAAdfaWQAUCBQxkVm+"
        b"XdxJ9tPBW5ld2d1aWQAEAAAAANgS/xhRXXv8kfIec+dYdyCAm5ld2d1"
        b"aWRzdHJpbmcAJQAAAGYyZWY3NTQ1LTYxZmMtNGI2MC04MmRjLTYxOWR"
        b"jZjc5Yzg0NwAAbAAAAAdfaWQAUCBQxkVm+XdxJ9tQBW5ld2d1aWQAEA"
        b"AAAAPqREIbhZPUJOSdHCJIgaqNAm5ld2d1aWRzdHJpbmcAJQAAADI0Z"
        b"DQ5Mzg1LTFiNDItNDRlYS04ZGFhLTgxNDgyMjFjOWRlNAAAbAAAAAdf"
        b"aWQAUCBQxkVm+XdxJ9tRBW5ld2d1aWQAEAAAAANjQBn/aQuNfRyfNyx"
        b"29COkAm5ld2d1aWRzdHJpbmcAJQAAADdkOGQwYjY5LWZmMTktNDA2My"
        b"1hNDIzLWY0NzYyYzM3OWYxYwAAbAAAAAdfaWQAUCBQxkVm+XdxJ9tSB"
        b"W5ld2d1aWQAEAAAAAMtSv/Et1cAQUFHUYevqxaLAm5ld2d1aWRzdHJp"
        b"bmcAJQAAADQxMDA1N2I3LWM0ZmYtNGEyZC04YjE2LWFiYWY4NzUxNDc"
        b"0MQAA"
    )
    java_data = base64.b64decode(from_java)

    # Generated by the .net driver
    from_csharp = (
        b"ZAAAABBfaWQAAAAAAAVuZXdndWlkABAAAAAD+MkoCd/Jy0iYJ7Vhl"
        b"iF3BAJuZXdndWlkc3RyaW5nACUAAAAwOTI4YzlmOC1jOWRmLTQ4Y2"
        b"ItOTgyNy1iNTYxOTYyMTc3MDQAAGQAAAAQX2lkAAEAAAAFbmV3Z3V"
        b"pZAAQAAAAA9MD0oXQe6VOp7mK4jkttWUCbmV3Z3VpZHN0cmluZwAl"
        b"AAAAODVkMjAzZDMtN2JkMC00ZWE1LWE3YjktOGFlMjM5MmRiNTY1A"
        b"ABkAAAAEF9pZAACAAAABW5ld2d1aWQAEAAAAAPRmIO2auc/Tprq1Z"
        b"oQ1oNYAm5ld2d1aWRzdHJpbmcAJQAAAGI2ODM5OGQxLWU3NmEtNGU"
        b"zZi05YWVhLWQ1OWExMGQ2ODM1OAAAZAAAABBfaWQAAwAAAAVuZXdn"
        b"dWlkABAAAAADISpriopuTEaXIa7arYOCFAJuZXdndWlkc3RyaW5nA"
        b"CUAAAA4YTZiMmEyMS02ZThhLTQ2NGMtOTcyMS1hZWRhYWQ4MzgyMT"
        b"QAAGQAAAAQX2lkAAQAAAAFbmV3Z3VpZAAQAAAAA98eg0CFpGlPihP"
        b"MwOmYGOMCbmV3Z3VpZHN0cmluZwAlAAAANDA4MzFlZGYtYTQ4NS00"
        b"ZjY5LThhMTMtY2NjMGU5OTgxOGUzAAA="
    )
    csharp_data = base64.b64decode(from_csharp)


class TestBinary(unittest.TestCase):
    def test_binary(self):
        a_string = "hello world"
        a_binary = Binary(b"hello world")
        self.assertTrue(a_binary.startswith(b"hello"))
        self.assertTrue(a_binary.endswith(b"world"))
        self.assertTrue(isinstance(a_binary, Binary))
        self.assertFalse(isinstance(a_string, Binary))

    def test_exceptions(self):
        self.assertRaises(TypeError, Binary, None)
        self.assertRaises(TypeError, Binary, 5)
        self.assertRaises(TypeError, Binary, 10.2)
        self.assertRaises(TypeError, Binary, b"hello", None)
        self.assertRaises(TypeError, Binary, b"hello", "100")
        self.assertRaises(ValueError, Binary, b"hello", -1)
        self.assertRaises(ValueError, Binary, b"hello", 256)
        self.assertTrue(Binary(b"hello", 0))
        self.assertTrue(Binary(b"hello", 255))
        self.assertRaises(TypeError, Binary, "hello")

    def test_subtype(self):
        one = Binary(b"hello")
        self.assertEqual(one.subtype, 0)
        two = Binary(b"hello", 2)
        self.assertEqual(two.subtype, 2)
        three = Binary(b"hello", 100)
        self.assertEqual(three.subtype, 100)

    def test_equality(self):
        two = Binary(b"hello")
        three = Binary(b"hello", 100)
        self.assertNotEqual(two, three)
        self.assertEqual(three, Binary(b"hello", 100))
        self.assertEqual(two, Binary(b"hello"))
        self.assertNotEqual(two, Binary(b"hello "))
        self.assertNotEqual(b"hello", Binary(b"hello"))

        # Explicitly test inequality
        self.assertFalse(three != Binary(b"hello", 100))
        self.assertFalse(two != Binary(b"hello"))

    def test_repr(self):
        one = Binary(b"hello world")
        self.assertEqual(repr(one), "Binary({}, 0)".format(repr(b"hello world")))
        two = Binary(b"hello world", 2)
        self.assertEqual(repr(two), "Binary({}, 2)".format(repr(b"hello world")))
        three = Binary(b"\x08\xFF")
        self.assertEqual(repr(three), "Binary({}, 0)".format(repr(b"\x08\xFF")))
        four = Binary(b"\x08\xFF", 2)
        self.assertEqual(repr(four), "Binary({}, 2)".format(repr(b"\x08\xFF")))
        five = Binary(b"test", 100)
        self.assertEqual(repr(five), "Binary({}, 100)".format(repr(b"test")))

    def test_hash(self):
        one = Binary(b"hello world")
        two = Binary(b"hello world", 42)
        self.assertEqual(hash(Binary(b"hello world")), hash(one))
        self.assertNotEqual(hash(one), hash(two))
        self.assertEqual(hash(Binary(b"hello world", 42)), hash(two))

    def test_uuid_subtype_4(self):
        """Only STANDARD should decode subtype 4 as native uuid."""
        expected_uuid = uuid.uuid4()
        expected_bin = Binary(expected_uuid.bytes, 4)
        doc = {"uuid": expected_bin}
        encoded = encode(doc)
        for uuid_rep in (
            UuidRepresentation.PYTHON_LEGACY,
            UuidRepresentation.JAVA_LEGACY,
            UuidRepresentation.CSHARP_LEGACY,
        ):
            opts = CodecOptions(uuid_representation=uuid_rep)
            self.assertEqual(expected_bin, decode(encoded, opts)["uuid"])
        opts = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
        self.assertEqual(expected_uuid, decode(encoded, opts)["uuid"])

    def test_legacy_java_uuid(self):
        # Test decoding
        data = BinaryData.java_data
        docs = bson.decode_all(data, CodecOptions(SON[str, Any], False, PYTHON_LEGACY))
        for d in docs:
            self.assertNotEqual(d["newguid"], uuid.UUID(d["newguidstring"]))

        docs = bson.decode_all(data, CodecOptions(SON[str, Any], False, STANDARD))
        for d in docs:
            self.assertNotEqual(d["newguid"], uuid.UUID(d["newguidstring"]))

        docs = bson.decode_all(data, CodecOptions(SON[str, Any], False, CSHARP_LEGACY))
        for d in docs:
            self.assertNotEqual(d["newguid"], uuid.UUID(d["newguidstring"]))

        docs = bson.decode_all(data, CodecOptions(SON[str, Any], False, JAVA_LEGACY))
        for d in docs:
            self.assertEqual(d["newguid"], uuid.UUID(d["newguidstring"]))

        # Test encoding
        encoded = b"".join(
            [encode(doc, False, CodecOptions(uuid_representation=PYTHON_LEGACY)) for doc in docs]
        )
        self.assertNotEqual(data, encoded)

        encoded = b"".join(
            [encode(doc, False, CodecOptions(uuid_representation=STANDARD)) for doc in docs]
        )
        self.assertNotEqual(data, encoded)

        encoded = b"".join(
            [encode(doc, False, CodecOptions(uuid_representation=CSHARP_LEGACY)) for doc in docs]
        )
        self.assertNotEqual(data, encoded)

        encoded = b"".join(
            [encode(doc, False, CodecOptions(uuid_representation=JAVA_LEGACY)) for doc in docs]
        )
        self.assertEqual(data, encoded)

    def test_legacy_csharp_uuid(self):
        data = BinaryData.csharp_data

        # Test decoding
        docs = bson.decode_all(data, CodecOptions(SON[str, Any], False, PYTHON_LEGACY))
        for d in docs:
            self.assertNotEqual(d["newguid"], uuid.UUID(d["newguidstring"]))

        docs = bson.decode_all(data, CodecOptions(SON[str, Any], False, STANDARD))
        for d in docs:
            self.assertNotEqual(d["newguid"], uuid.UUID(d["newguidstring"]))

        docs = bson.decode_all(data, CodecOptions(SON[str, Any], False, JAVA_LEGACY))
        for d in docs:
            self.assertNotEqual(d["newguid"], uuid.UUID(d["newguidstring"]))

        docs = bson.decode_all(data, CodecOptions(SON[str, Any], False, CSHARP_LEGACY))
        for d in docs:
            self.assertEqual(d["newguid"], uuid.UUID(d["newguidstring"]))

        # Test encoding
        encoded = b"".join(
            [encode(doc, False, CodecOptions(uuid_representation=PYTHON_LEGACY)) for doc in docs]
        )
        self.assertNotEqual(data, encoded)

        encoded = b"".join(
            [encode(doc, False, CodecOptions(uuid_representation=STANDARD)) for doc in docs]
        )
        self.assertNotEqual(data, encoded)

        encoded = b"".join(
            [encode(doc, False, CodecOptions(uuid_representation=JAVA_LEGACY)) for doc in docs]
        )
        self.assertNotEqual(data, encoded)

        encoded = b"".join(
            [encode(doc, False, CodecOptions(uuid_representation=CSHARP_LEGACY)) for doc in docs]
        )
        self.assertEqual(data, encoded)

    def test_pickle(self):
        b1 = Binary(b"123", 2)

        # For testing backwards compatibility with pre-2.4 pymongo
        p = (
            b"\x80\x03cbson.binary\nBinary\nq\x00C\x03123q\x01\x85q"
            b"\x02\x81q\x03}q\x04X\x10\x00\x00\x00_Binary__subtypeq"
            b"\x05K\x02sb."
        )

        if not sys.version.startswith("3.0"):
            self.assertEqual(b1, pickle.loads(p))

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertEqual(b1, pickle.loads(pickle.dumps(b1, proto)))

        uu = uuid.uuid4()
        uul = Binary.from_uuid(uu, UuidRepresentation.PYTHON_LEGACY)

        self.assertEqual(uul, copy.copy(uul))
        self.assertEqual(uul, copy.deepcopy(uul))

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertEqual(uul, pickle.loads(pickle.dumps(uul, proto)))

    def test_buffer_protocol(self):
        b0 = Binary(b"123", 2)

        self.assertEqual(b0, Binary(memoryview(b"123"), 2))
        self.assertEqual(b0, Binary(bytearray(b"123"), 2))
        with mmap.mmap(-1, len(b"123")) as mm:
            mm.write(b"123")
            mm.seek(0)
            self.assertEqual(b0, Binary(mm, 2))
        self.assertEqual(b0, Binary(array.array("B", b"123"), 2))


class TestUuidSpecExplicitCoding(unittest.TestCase):
    uuid: uuid.UUID

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.uuid = uuid.UUID("00112233445566778899AABBCCDDEEFF")

    @staticmethod
    def _hex_to_bytes(hexstring):
        return bytes.fromhex(hexstring)

    # Explicit encoding prose test #1
    def test_encoding_1(self):
        obj = Binary.from_uuid(self.uuid)
        expected_obj = Binary(self._hex_to_bytes("00112233445566778899AABBCCDDEEFF"), 4)
        self.assertEqual(obj, expected_obj)

    def _test_encoding_w_uuid_rep(self, uuid_rep, expected_hexstring, expected_subtype):
        obj = Binary.from_uuid(self.uuid, uuid_rep)
        expected_obj = Binary(self._hex_to_bytes(expected_hexstring), expected_subtype)
        self.assertEqual(obj, expected_obj)

    # Explicit encoding prose test #2
    def test_encoding_2(self):
        self._test_encoding_w_uuid_rep(
            UuidRepresentation.STANDARD, "00112233445566778899AABBCCDDEEFF", 4
        )

    # Explicit encoding prose test #3
    def test_encoding_3(self):
        self._test_encoding_w_uuid_rep(
            UuidRepresentation.JAVA_LEGACY, "7766554433221100FFEEDDCCBBAA9988", 3
        )

    # Explicit encoding prose test #4
    def test_encoding_4(self):
        self._test_encoding_w_uuid_rep(
            UuidRepresentation.CSHARP_LEGACY, "33221100554477668899AABBCCDDEEFF", 3
        )

    # Explicit encoding prose test #5
    def test_encoding_5(self):
        self._test_encoding_w_uuid_rep(
            UuidRepresentation.PYTHON_LEGACY, "00112233445566778899AABBCCDDEEFF", 3
        )

    # Explicit encoding prose test #6
    def test_encoding_6(self):
        with self.assertRaises(ValueError):
            Binary.from_uuid(self.uuid, UuidRepresentation.UNSPECIFIED)

    # Explicit decoding prose test #1
    def test_decoding_1(self):
        obj = Binary(self._hex_to_bytes("00112233445566778899AABBCCDDEEFF"), 4)

        # Case i:
        self.assertEqual(obj.as_uuid(), self.uuid)
        # Case ii:
        self.assertEqual(obj.as_uuid(UuidRepresentation.STANDARD), self.uuid)
        # Cases iii-vi:
        for uuid_rep in (
            UuidRepresentation.JAVA_LEGACY,
            UuidRepresentation.CSHARP_LEGACY,
            UuidRepresentation.PYTHON_LEGACY,
        ):
            with self.assertRaises(ValueError):
                obj.as_uuid(uuid_rep)

    def _test_decoding_legacy(self, hexstring, uuid_rep):
        obj = Binary(self._hex_to_bytes(hexstring), 3)

        # Case i:
        with self.assertRaises(ValueError):
            obj.as_uuid()
        # Cases ii-iii:
        for rep in (UuidRepresentation.STANDARD, UuidRepresentation.UNSPECIFIED):
            with self.assertRaises(ValueError):
                obj.as_uuid(rep)
        # Case iv:
        self.assertEqual(obj.as_uuid(uuid_rep), self.uuid)

    # Explicit decoding prose test #2
    def test_decoding_2(self):
        self._test_decoding_legacy(
            "7766554433221100FFEEDDCCBBAA9988", UuidRepresentation.JAVA_LEGACY
        )

    # Explicit decoding prose test #3
    def test_decoding_3(self):
        self._test_decoding_legacy(
            "33221100554477668899AABBCCDDEEFF", UuidRepresentation.CSHARP_LEGACY
        )

    # Explicit decoding prose test #4
    def test_decoding_4(self):
        self._test_decoding_legacy(
            "00112233445566778899AABBCCDDEEFF", UuidRepresentation.PYTHON_LEGACY
        )


class TestUuidSpecImplicitCoding(IntegrationTest):
    uuid: uuid.UUID

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.uuid = uuid.UUID("00112233445566778899AABBCCDDEEFF")

    @staticmethod
    def _hex_to_bytes(hexstring):
        return bytes.fromhex(hexstring)

    def _get_coll_w_uuid_rep(self, uuid_rep):
        codec_options = self.client.codec_options.with_options(
            uuid_representation=validate_uuid_representation(None, uuid_rep)
        )
        coll = self.db.get_collection(
            "pymongo_test", codec_options=codec_options, write_concern=WriteConcern("majority")
        )
        return coll

    def _test_encoding(self, uuid_rep, expected_hexstring, expected_subtype):
        coll = self._get_coll_w_uuid_rep(uuid_rep)
        coll.delete_many({})
        coll.insert_one({"_id": self.uuid})
        self.assertTrue(
            coll.find_one({"_id": Binary(self._hex_to_bytes(expected_hexstring), expected_subtype)})
        )

    # Implicit encoding prose test #1
    def test_encoding_1(self):
        self._test_encoding("javaLegacy", "7766554433221100FFEEDDCCBBAA9988", 3)

    # Implicit encoding prose test #2
    def test_encoding_2(self):
        self._test_encoding("csharpLegacy", "33221100554477668899AABBCCDDEEFF", 3)

    # Implicit encoding prose test #3
    def test_encoding_3(self):
        self._test_encoding("pythonLegacy", "00112233445566778899AABBCCDDEEFF", 3)

    # Implicit encoding prose test #4
    def test_encoding_4(self):
        self._test_encoding("standard", "00112233445566778899AABBCCDDEEFF", 4)

    # Implicit encoding prose test #5
    def test_encoding_5(self):
        with self.assertRaises(ValueError):
            self._test_encoding("unspecified", "dummy", -1)

    def _test_decoding(
        self,
        client_uuid_representation_string,
        legacy_field_uuid_representation,
        expected_standard_field_value,
        expected_legacy_field_value,
    ):
        coll = self._get_coll_w_uuid_rep(client_uuid_representation_string)
        coll.drop()

        standard_val = Binary.from_uuid(self.uuid, UuidRepresentation.STANDARD)
        legacy_val = Binary.from_uuid(self.uuid, legacy_field_uuid_representation)
        coll.insert_one({"standard": standard_val, "legacy": legacy_val})

        doc = coll.find_one()
        self.assertEqual(doc["standard"], expected_standard_field_value)
        self.assertEqual(doc["legacy"], expected_legacy_field_value)

    # Implicit decoding prose test #1
    def test_decoding_1(self):
        standard_binary = Binary.from_uuid(self.uuid, UuidRepresentation.STANDARD)
        self._test_decoding(
            "javaLegacy", UuidRepresentation.JAVA_LEGACY, standard_binary, self.uuid
        )
        self._test_decoding(
            "csharpLegacy", UuidRepresentation.CSHARP_LEGACY, standard_binary, self.uuid
        )
        self._test_decoding(
            "pythonLegacy", UuidRepresentation.PYTHON_LEGACY, standard_binary, self.uuid
        )

    # Implicit decoding pose test #2
    def test_decoding_2(self):
        legacy_binary = Binary.from_uuid(self.uuid, UuidRepresentation.PYTHON_LEGACY)
        self._test_decoding("standard", UuidRepresentation.PYTHON_LEGACY, self.uuid, legacy_binary)

    # Implicit decoding pose test #3
    def test_decoding_3(self):
        expected_standard_value = Binary.from_uuid(self.uuid, UuidRepresentation.STANDARD)
        for legacy_uuid_rep in (
            UuidRepresentation.PYTHON_LEGACY,
            UuidRepresentation.CSHARP_LEGACY,
            UuidRepresentation.JAVA_LEGACY,
        ):
            expected_legacy_value = Binary.from_uuid(self.uuid, legacy_uuid_rep)
            self._test_decoding(
                "unspecified", legacy_uuid_rep, expected_standard_value, expected_legacy_value
            )


if __name__ == "__main__":
    unittest.main()
