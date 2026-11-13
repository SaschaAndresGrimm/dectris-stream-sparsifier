# worker.py
"""Worker process: decodes images, sparsifies them, and appends results to
per-worker binary shard files. Pixel data never travels back to the main
process; only small per-series summaries do."""
import logging
import os
import re
import signal
import time

import numpy as np
import zmq

from logging_setup import setup_logging
from processing import FRAME_DTYPE, build_bad_pixel_masks, process_frames
from receiver import KIND_END, KIND_IMAGE, KIND_QUIT, KIND_START, WORKER_QUEUE_DEPTH
from stream_decode import decode_message, rational_to_seconds

FLUSH_BYTES = 16 * 1024 * 1024  # flush buffered hits to disk every ~16 MB
STACK_MAX_BYTES = 1 << 20  # frames smaller than this are sparsified as one stack
QUIT_GRACE_S = 5.0


def series_shard_dir(shard_root, series_id):
    return os.path.join(shard_root, f"series_{series_id}")


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))


def _stacks(ids, frames):
    """Yield (image_ids, 3D stack): runs of same-shape small frames are stacked,
    large frames are passed on one at a time (as a view, no copy)."""
    i = 0
    while i < len(frames):
        f = frames[i]
        j = i + 1
        if f.nbytes < STACK_MAX_BYTES:
            while (j < len(frames) and frames[j].shape == f.shape and frames[j].dtype == f.dtype
                   and (j - i + 1) * f.nbytes <= 64 * STACK_MAX_BYTES):
                j += 1
        if j - i == 1:
            yield ids[i:j], f[np.newaxis]
        else:
            yield ids[i:j], np.stack(frames[i:j])
        i = j


class SeriesWriter:
    """Buffers hits for one series and appends them to this worker's shard files."""

    def __init__(self, shard_root, worker_idx, series_id, start_message=None):
        self.series_id = series_id
        self.worker_idx = worker_idx
        self.dir = series_shard_dir(shard_root, series_id)
        os.makedirs(self.dir, exist_ok=True)
        self.prefix = os.path.join(self.dir, f"w{worker_idx:03d}")
        self.masks = build_bad_pixel_masks(start_message)

        self.hit_files = {}
        self.hit_buffers = {}
        self.frame_rows = []
        self.frames_file = open(f"{self.prefix}_frames.bin", "wb")
        self.buffered_bytes = 0
        self.n_frames = 0
        self.n_hits = {}

    def add_images(self, msgs, logger=None):
        """Sparsify a batch of decoded image messages of this series."""
        t0 = time.perf_counter()
        ids = [m["image_id"] for m in msgs]
        for m in msgs:
            self.frame_rows.append((
                m["image_id"],
                rational_to_seconds(m.get("start_time")),
                rational_to_seconds(m.get("stop_time")),
                rational_to_seconds(m.get("real_time")),
            ))
        self.n_frames += len(msgs)

        for threshold in msgs[0]["data"]:
            name = safe_name(threshold)
            mask = self.masks.get(threshold, self.masks.get(None))
            frames = [m["data"][threshold] for m in msgs]
            for group_ids, stack in _stacks(ids, frames):
                hits = process_frames(stack, group_ids, mask)
                if len(hits):
                    self.hit_buffers.setdefault(name, []).append(hits)
                    self.buffered_bytes += hits.nbytes
                self.n_hits[name] = self.n_hits.get(name, 0) + len(hits)
                if logger is not None:
                    logger.debug("Worker %d: series %s images %s..%s %s: %d hits",
                                 self.worker_idx, self.series_id, group_ids[0], group_ids[-1],
                                 name, len(hits))

        if logger is not None:
            logger.debug("Worker %d: series %s, %d image(s) processed in %.3f ms",
                         self.worker_idx, self.series_id, len(msgs),
                         (time.perf_counter() - t0) * 1e3)

        if self.buffered_bytes >= FLUSH_BYTES:
            if logger is not None:
                logger.debug("Worker %d: flushing %.1f MB of hits to shard files",
                             self.worker_idx, self.buffered_bytes / 1e6)
            self.flush()

    def flush(self):
        for name, chunks in self.hit_buffers.items():
            f = self.hit_files.get(name)
            if f is None:
                f = self.hit_files[name] = open(f"{self.prefix}_hits_{name}.bin", "wb")
            np.concatenate(chunks).tofile(f)
        self.hit_buffers.clear()
        self.buffered_bytes = 0
        if self.frame_rows:
            np.array(self.frame_rows, dtype=FRAME_DTYPE).tofile(self.frames_file)
            self.frame_rows.clear()

    def close(self):
        self.flush()
        for f in self.hit_files.values():
            f.close()
        self.frames_file.close()
        return {"frames": self.n_frames, "hits": dict(self.n_hits)}


def worker(worker_idx, worker_endpoint, shard_root, control_queue, stop_event,
           log_queue, log_level=logging.INFO):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # main handles Ctrl+C
    setup_logging(log_queue, level=log_level)
    logger = logging.getLogger(f"worker{worker_idx}")

    context = zmq.Context()
    sock = context.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVHWM, WORKER_QUEUE_DEPTH)
    sock.connect(worker_endpoint)
    logger.info("Worker %d connected to %s", worker_idx, worker_endpoint)

    writer = None
    debug = logger.isEnabledFor(logging.DEBUG)

    def finish_series(partial):
        nonlocal writer
        if writer is None:
            return
        stats = writer.close()
        control_queue.put({"type": "worker_done", "series_id": writer.series_id,
                           "worker": worker_idx, "partial": partial, **stats})
        logger.info("Worker %d finished series %s: %d frames%s", worker_idx,
                    writer.series_id, stats["frames"], " (partial)" if partial else "")
        writer = None

    stop_seen_at = None
    try:
        while True:
            if not sock.poll(1000):
                # Normally the receiver sends KIND_QUIT after the last data. This is only
                # a fallback in case the receiver died: give it a grace period first.
                if stop_event.is_set():
                    stop_seen_at = stop_seen_at or time.monotonic()
                    if time.monotonic() - stop_seen_at > QUIT_GRACE_S:
                        logger.warning("Worker %d: no quit message from receiver; exiting", worker_idx)
                        break
                continue

            parts = sock.recv_multipart(copy=False)
            kind = parts[0].bytes
            payload = parts[1] if len(parts) > 1 else None

            if kind == KIND_IMAGE:  # a batch: one image message per part
                msgs = []
                for part in parts[1:]:
                    try:
                        msgs.append(decode_message(part.buffer))
                    except Exception:
                        logger.exception("Worker %d: could not decode image", worker_idx)
                i = 0
                while i < len(msgs):  # normally one series per batch; split if not
                    sid = msgs[i].get("series_id")
                    j = i + 1
                    while j < len(msgs) and msgs[j].get("series_id") == sid:
                        j += 1
                    try:
                        if writer is None or writer.series_id != sid:
                            # Image without a preceding start (e.g. we joined mid-series)
                            finish_series(partial=True)
                            logger.debug("Worker %d: image for series %s without start message",
                                         worker_idx, sid)
                            writer = SeriesWriter(shard_root, worker_idx, sid)
                        writer.add_images(msgs[i:j], logger if debug else None)
                    except Exception:
                        logger.exception("Worker %d: error processing %d image(s)", worker_idx, j - i)
                    i = j

            elif kind == KIND_START:
                finish_series(partial=True)  # no-op unless the previous series never ended
                msg = decode_message(payload.buffer)
                writer = SeriesWriter(shard_root, worker_idx, msg.get("series_id"), msg)
                logger.debug("Worker %d: start of series %s, bad-pixel masks for %s",
                             worker_idx, writer.series_id,
                             sorted(str(k) for k in writer.masks) or "none")

            elif kind == KIND_END:
                if writer is None:
                    # This worker received no images for the series; still report in.
                    msg = decode_message(payload.buffer)
                    control_queue.put({"type": "worker_done", "series_id": msg.get("series_id"),
                                       "worker": worker_idx, "partial": False,
                                       "frames": 0, "hits": {}})
                else:
                    finish_series(partial=False)

            elif kind == KIND_QUIT:
                break
    except Exception:
        logger.exception("Worker %d crashed", worker_idx)
        raise
    finally:
        finish_series(partial=True)
        sock.close(linger=0)
        context.term()
        logger.info("Worker %d exiting", worker_idx)
