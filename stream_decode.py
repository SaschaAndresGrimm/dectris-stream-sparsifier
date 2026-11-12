# stream_decode.py
"""CBOR decoding helpers for the DECTRIS Stream v2 protocol."""
import cbor2
import numpy as np
from dectris.compression import decompress


def decode_multi_dim_array(tag, column_major):
    dimensions, contents = tag.value
    if isinstance(contents, list):
        array = np.empty((len(contents),), dtype=object)
        array[:] = contents
    elif isinstance(contents, (np.ndarray, np.generic)):
        array = contents
    else:
        raise cbor2.CBORDecodeValueError("expected array or typed array")
    return array.reshape(dimensions, order="F" if column_major else "C")


def decode_typed_array(tag, dtype):
    if not isinstance(tag.value, bytes):
        raise cbor2.CBORDecodeValueError("expected byte string in typed array")
    return np.frombuffer(tag.value, dtype=dtype)


def decode_dectris_compression(tag):
    algorithm, elem_size, encoded = tag.value
    return decompress(encoded, algorithm, elem_size=elem_size)


_TYPED_ARRAY_DTYPES = {
    64: "u1", 65: ">u2", 66: ">u4", 67: ">u8",
    68: "u1", 69: "<u2", 70: "<u4", 71: "<u8",
    72: "i1", 73: ">i2", 74: ">i4", 75: ">i8",
    77: "<i2", 78: "<i4", 79: "<i8",
    80: ">f2", 81: ">f4", 82: ">f8", 83: ">f16",
    84: "<f2", 85: "<f4", 86: "<f8", 87: "<f16",
}


def tag_hook(arg1, arg2):
    # cbor2 5.x calls tag_hook(decoder, tag); cbor2 6.x calls tag_hook(tag, immutable).
    tag = arg1 if isinstance(arg1, cbor2.CBORTag) else arg2
    if tag.tag in _TYPED_ARRAY_DTYPES:
        return decode_typed_array(tag, _TYPED_ARRAY_DTYPES[tag.tag])
    if tag.tag == 40:
        return decode_multi_dim_array(tag, column_major=False)
    if tag.tag == 1040:
        return decode_multi_dim_array(tag, column_major=True)
    if tag.tag == 56500:
        return decode_dectris_compression(tag)
    return tag


def decode_message(buf):
    """Fully decode a Stream v2 message (including decompression)."""
    return cbor2.loads(buf, tag_hook=tag_hook)


_TYPE_KEY = b"\x64type"  # CBOR text string of length 4: "type"


def peek_type(buf):
    """Return the message 'type' without decoding/decompressing the payload.

    Fast path: look for the encoded key "type" near the start of the message
    and read the short text string that follows. Falls back to a full CBOR
    parse (without tag decoding / decompression) if the fast path fails.
    """
    head = bytes(buf[:64])
    i = head.find(_TYPE_KEY)
    if i >= 0:
        j = i + len(_TYPE_KEY)
        if j < len(head) and 0x60 <= head[j] <= 0x77:  # short text string
            n = head[j] - 0x60
            value = head[j + 1:j + 1 + n]
            if len(value) == n:
                return value.decode("ascii", "replace")
    msg = cbor2.loads(buf)
    return msg.get("type") if isinstance(msg, dict) else None


def rational_to_seconds(value):
    """Stream v2 times are [numerator, denominator] rationals; convert to float seconds."""
    if value is None:
        return float("nan")
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2 and value[1]:
            return float(value[0]) / float(value[1])
        if len(value) == 1:
            return float(value[0])
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
