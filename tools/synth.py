# tools/synth.py
"""Synthetic DECTRIS Stream v2 messages for benchmarks (bslz4-compressed images)."""
import struct

import cbor2
import lz4.block
import numpy as np

# Detector presets: (height, width) in pixels. Anything else: pass HxW.
DETECTORS = {
    "SELUN": (95, 95),    # SELUN: 200 um pixels, up to 120 kHz at 16 bit
    "1M": (1062, 1028),   # EIGER2 X 1M
    "4M": (2162, 2068),   # EIGER2 X 4M
    "16M": (4362, 4148),  # EIGER2 X 16M
}

_TYPED_ARRAY_TAG = {1: 68, 2: 69, 4: 70}  # little-endian uint8/16/32


def _bitshuffle_block(raw, elem_size):
    n = len(raw) // elem_size
    bits = np.unpackbits(raw.reshape(n, elem_size), axis=1, bitorder="little")
    return np.packbits(bits.reshape(n, elem_size * 8).T, axis=1, bitorder="little").tobytes()


def bslz4_compress(arr):
    """Bitshuffle + LZ4 in the HDF5-filter layout that dectris.compression decodes."""
    es = arr.dtype.itemsize
    raw = np.frombuffer(np.ascontiguousarray(arr).tobytes(), np.uint8)
    block = 8192 // es
    out = [struct.pack(">QI", raw.size, block * es)]
    n_full, rem = divmod(arr.size, block)
    rem8 = rem - rem % 8
    for start, count in [(i * block, block) for i in range(n_full)] + [(n_full * block, rem8)]:
        if count:
            c = lz4.block.compress(_bitshuffle_block(raw[start * es:(start + count) * es], es),
                                   store_size=False)
            out.append(struct.pack(">I", len(c)) + c)
    out.append(raw[(n_full * block + rem8) * es:].tobytes())  # leftover stored raw
    return b"".join(out)


def make_frame(shape, hit_fraction, dtype=np.uint32, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros(shape[0] * shape[1], dtype)
    k = int(a.size * hit_fraction)
    if k:
        a[rng.choice(a.size, k, replace=False)] = rng.integers(1, 20, k)
    return a.reshape(shape)


def image_message(series_id, image_id, frame):
    es = frame.dtype.itemsize
    data = cbor2.CBORTag(40, [list(frame.shape), cbor2.CBORTag(_TYPED_ARRAY_TAG[es],
                         cbor2.CBORTag(56500, ["bslz4", es, bslz4_compress(frame)]))])
    return cbor2.dumps({"type": "image", "series_id": series_id, "image_id": image_id,
                        "start_time": [image_id, 1000], "stop_time": [image_id + 1, 1000],
                        "real_time": [1, 1000], "data": {"threshold_1": data}})


def start_message(series_id):
    return cbor2.dumps({"type": "start", "series_id": series_id})


def end_message(series_id):
    return cbor2.dumps({"type": "end", "series_id": series_id})


def parse_shape(text):
    if text.upper() in DETECTORS:
        return DETECTORS[text.upper()]
    h, w = text.lower().split("x")
    return int(h), int(w)
