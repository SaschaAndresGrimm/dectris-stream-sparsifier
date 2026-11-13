# processing.py
from collections.abc import Mapping

import numpy as np

# One row per hit pixel: 12 bytes (vs. 32 bytes for a float64 (N, 4) array).
HIT_DTYPE = np.dtype([
    ("image_id", "<u4"),
    ("x", "<u2"),
    ("y", "<u2"),
    ("value", "<u4"),
])

# One row per frame. Times are in seconds (converted from Stream v2 rationals).
FRAME_DTYPE = np.dtype([
    ("image_id", "<u4"),
    ("start_time", "<f8"),
    ("stop_time", "<f8"),
    ("real_time", "<f8"),
])


def build_bad_pixel_masks(start_message):
    """Return {threshold_name: bool array (True = bad pixel)} from a start message.

    Stream v2 'pixel_mask' is a map of channel -> uint32 bitmask (non-zero = bad).
    """
    masks = {}
    if not start_message:
        return masks
    pixel_mask = start_message.get("pixel_mask")
    if isinstance(pixel_mask, Mapping):  # cbor2 6.x may return frozendict, not dict
        for name, mask in pixel_mask.items():
            if isinstance(mask, np.ndarray) and mask.ndim == 2:
                masks[name] = mask != 0
    elif isinstance(pixel_mask, np.ndarray) and pixel_mask.ndim == 2:
        masks[None] = pixel_mask != 0  # applies to all thresholds
    return masks


def process_frame(data, image_id, bad_mask=None):
    """Sparsify one 2D frame. See process_frames()."""
    return process_frames(data[np.newaxis], [image_id], bad_mask)


def process_frames(stack, image_ids, bad_mask=None):
    """Find all pixels with 0 < value < dtype max (and not masked) in a stack of frames.

    Frames are mostly zeros, so we first collect the nonzero pixels (one full
    pass over the data) and apply the remaining tests only to that small set.
    Processing many small frames in one call avoids per-frame numpy overhead.

    Parameters:
        stack (np.ndarray): 3D integer array (n_frames, height, width).
        image_ids (sequence of int): image_id for each frame.
        bad_mask (np.ndarray[bool] | None): 2D, True for pixels to ignore.

    Returns:
        np.ndarray with dtype HIT_DTYPE (possibly empty), ordered by (frame, y, x).
    """
    if stack.dtype.kind not in "ui":
        raise TypeError(f"expected integer image data, got {stack.dtype}")
    n, height, width = stack.shape
    frame_px = height * width

    flat = stack.reshape(-1)
    idx = np.flatnonzero(flat)
    values = flat[idx]
    keep = values < np.iinfo(stack.dtype).max  # max = masked / gap / saturated pixels
    if stack.dtype.kind == "i":
        keep &= values > 0
    if bad_mask is not None and bad_mask.shape == (height, width):
        keep &= ~bad_mask.ravel()[idx % frame_px if n > 1 else idx]
    idx, values = idx[keep], values[keep]

    frame, pixel = np.divmod(idx, frame_px)
    y, x = np.divmod(pixel, width)
    hits = np.empty(len(idx), dtype=HIT_DTYPE)
    hits["image_id"] = np.asarray(image_ids, dtype=np.uint32)[frame] if n > 1 else image_ids[0]
    hits["x"] = x
    hits["y"] = y
    hits["value"] = values
    return hits
