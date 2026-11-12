# output.py
"""Writing series metadata and merging per-worker shards into final .npy files."""
import datetime
import glob
import json
import os
import shutil
from collections.abc import Mapping

import cbor2
import numpy as np

from processing import FRAME_DTYPE, HIT_DTYPE


def _to_jsonable(obj, path, arrays):
    """Convert a decoded CBOR message to JSON; numpy arrays are moved to `arrays`."""
    if isinstance(obj, np.ndarray):
        key = path or "value"
        arrays[key] = obj
        return f"<array '{key}' shape={list(obj.shape)} dtype={obj.dtype}>"
    if isinstance(obj, Mapping):  # dict or cbor2 frozendict
        return {str(k): _to_jsonable(v, f"{path}.{k}" if path else str(k), arrays)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v, f"{path}.{i}", arrays) for i, v in enumerate(obj)]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return f"<{len(obj)} bytes>"
    if isinstance(obj, cbor2.CBORTag):
        return {"cbor_tag": obj.tag, "value": _to_jsonable(obj.value, path, arrays)}
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


def save_message(msg, base_path):
    """Save scalar fields as <base>.json and array fields (pixel_mask, flatfield, ...) as <base>_arrays.npz."""
    arrays = {}
    doc = _to_jsonable(msg, "", arrays)
    with open(f"{base_path}.json", "w") as f:
        json.dump(doc, f, indent=2)
    if arrays:
        np.savez_compressed(f"{base_path}_arrays.npz", **arrays)


def _concat_shards_to_npy(shard_files, dtype, out_path):
    """Stream raw shard files into one .npy without loading them into memory."""
    sizes = [os.path.getsize(p) for p in shard_files]
    total = sum(sizes)
    if total % dtype.itemsize:
        raise ValueError(f"shard sizes not a multiple of record size for {out_path}")
    n = total // dtype.itemsize
    with open(out_path, "wb") as out:
        np.lib.format.write_array_header_1_0(out, {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": (n,),
        })
        for p in shard_files:
            with open(p, "rb") as src:
                shutil.copyfileobj(src, out, 16 * 1024 * 1024)
    return n


def merge_series(shard_dir, out_base, keep_shards=False):
    """Merge worker shards of one series.

    Produces:
        <out_base>_hits_<threshold>.npy  (HIT_DTYPE, unsorted; sort by image_id if needed)
        <out_base>_frames.npy            (FRAME_DTYPE, sorted by image_id)

    Returns (n_frames, {threshold: n_hits}).
    """
    hit_counts = {}
    n_frames = 0
    if not os.path.isdir(shard_dir):
        return n_frames, hit_counts

    by_threshold = {}
    for path in sorted(glob.glob(os.path.join(shard_dir, "w*_hits_*.bin"))):
        name = os.path.basename(path).split("_hits_", 1)[1][:-len(".bin")]
        by_threshold.setdefault(name, []).append(path)
    for name, files in sorted(by_threshold.items()):
        hit_counts[name] = _concat_shards_to_npy(files, HIT_DTYPE, f"{out_base}_hits_{name}.npy")

    frame_files = sorted(glob.glob(os.path.join(shard_dir, "w*_frames.bin")))
    if frame_files:
        frames = np.concatenate([np.fromfile(p, dtype=FRAME_DTYPE) for p in frame_files])
        frames = frames[np.argsort(frames["image_id"], kind="stable")]
        np.save(f"{out_base}_frames.npy", frames)
        n_frames = len(frames)

    if not keep_shards:
        shutil.rmtree(shard_dir, ignore_errors=True)
    return n_frames, hit_counts
