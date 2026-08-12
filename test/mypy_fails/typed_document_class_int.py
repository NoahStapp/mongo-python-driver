from __future__ import annotations

from bson.codec_options import CodecOptions

CodecOptions(
    document_class=int
)  # error: Value of type variable "_DocumentType" of "CodecOptions" cannot be "int"
