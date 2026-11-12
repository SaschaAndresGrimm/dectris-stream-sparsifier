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
    """Find all pixels with 0 < value < dtype max (and not masked).

    Parameters:
        data (np.ndarray): 2D integer image.
        image_id (int): Frame index within the series.
        bad_mask (np.ndarray[bool] | None): True for pixels to ignore.

    Returns:
        np.ndarray with dtype HIT_DTYPE (possibly empty).
    """
    if data.dtype.kind not in "ui":
        raise TypeError(f"expected integer image data, got {data.dtype}")

    max_value = np.iinfo(data.dtype).max  # max = masked / gap / saturated pixels
    valid = (data > 0) & (data < max_value)
    if bad_mask is not None and bad_mask.shape == data.shape:
        valid &= ~bad_mask

    y, x = np.nonzero(valid)
    hits = np.empty(len(x), dtype=HIT_DTYPE)
    hits["image_id"] = image_id
    hits["x"] = x
    hits["y"] = y
    hits["value"] = data[y, x]
    return hits
