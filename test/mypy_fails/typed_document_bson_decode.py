from __future__ import annotations

from dataclasses import dataclass

import bson
from bson.codec_options import CodecOptions


@dataclass
class Movie:
    name: str
    year: int


options = CodecOptions(document_class=Movie)
# bson-level decode has no typed-document support, so typed codec options
# must be rejected statically (the runtime instantiates document_class as a
# mapping and would raise).
bson.decode(
    b"", codec_options=options
)  # error: Value of type variable "_MappingDocumentType" of "decode" cannot be "Movie"
bson.decode_all(
    b"", options
)  # error: Value of type variable "_MappingDocumentType" of "decode_all" cannot be "Movie"
