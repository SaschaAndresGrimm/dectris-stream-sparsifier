# receiver.py
"""Single receiver process: the only process connected to the detector.

The detector's PUSH socket round-robins messages over all connected PULL
sockets, so if every worker connected directly, 'start' and 'end' would each
reach only ONE worker and the others would never know a series ended.

Instead, this process receives everything and
  * forwards 'image' messages round-robin to the workers (raw bytes, no decode),
    in batches: everything already waiting on the socket (up to BATCH_MAX_IMAGES /
    BATCH_MAX_BYTES) goes to one worker in one multipart send. This keeps the
    per-message Python overhead low at high frame rates (e.g. 100+ kHz),
  * broadcasts 'start' / 'end' to EVERY worker (in-order on each worker's
    socket, so a worker sees 'end' only after all images routed to it), and
  * reports 'start' / 'end' to the main process.
"""
import logging
import signal
import time

import cbor2
import zmq

from logging_setup import setup_logging
from stream_decode import peek_type

# Wire protocol between receiver and workers: multipart [kind, payload]
KIND_START = b"S"
KIND_IMAGE = b"I"
KIND_END = b"E"
KIND_QUIT = b"Q"

WORKER_QUEUE_DEPTH = 64  # max in-flight batches per worker (bounds memory)
BATCH_MAX_IMAGES = 256
BATCH_MAX_BYTES = 1 << 20  # large frames are sent one per batch


def _send_blocking(sock, parts, stop_event=None, timeout=None):
    """Send, retrying while the peer is full. Returns False on timeout/stop."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        try:
            sock.send_multipart(parts, flags=zmq.NOBLOCK, copy=False)
            return True
        except zmq.Again:
            if deadline is not None and time.monotonic() > deadline:
                return False
            if stop_event is not None and stop_event.is_set() and deadline is None:
                deadline = time.monotonic() + 5.0
            sock.poll(100, zmq.POLLOUT)


def receiver(endpoint, worker_endpoints, control_queue, stop_event, log_queue, log_level):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # main handles Ctrl+C
    setup_logging(log_queue, level=log_level)
    logger = logging.getLogger("receiver")

    context = zmq.Context()
    pull = context.socket(zmq.PULL)
    pull.connect(endpoint)

    outs = []
    for ep in worker_endpoints:
        s = context.socket(zmq.PUSH)
        s.setsockopt(zmq.SNDHWM, WORKER_QUEUE_DEPTH)
        s.setsockopt(zmq.LINGER, 2000)
        s.bind(ep)
        outs.append(s)
    out_poller = zmq.Poller()
    for s in outs:
        out_poller.register(s, zmq.POLLOUT)

    logger.info("Receiver connected to %s, serving %d workers", endpoint, len(outs))

    next_worker = 0
    images_in_series = 0
    current_series = None
    debug = logger.isEnabledFor(logging.DEBUG)
    stats_t0 = time.monotonic()
    stats_images = stats_bytes = stats_batches = 0
    batch = []
    batch_bytes = 0

    def broadcast(kind, payload):
        for s in outs:
            _send_blocking(s, [kind, payload], stop_event)

    def send_batch(parts):
        """Send [KIND_IMAGE, img, img, ...] to the next worker with free queue space."""
        nonlocal next_worker
        n = len(outs)
        give_up_at = None
        while True:
            if stop_event.is_set():
                give_up_at = give_up_at or time.monotonic() + 5.0
                if time.monotonic() > give_up_at:
                    logger.warning("Workers not accepting data during shutdown; dropping %d image(s)",
                                   len(parts) - 1)
                    return False
            # Round-robin, skipping workers whose queue is full (simple load balancing)
            for k in range(n):
                idx = (next_worker + k) % n
                try:
                    outs[idx].send_multipart(parts, flags=zmq.NOBLOCK, copy=False)
                    next_worker = (idx + 1) % n
                    return True
                except zmq.Again:
                    continue
            out_poller.poll(100)  # all workers busy: wait for one to free up

    def flush_batch():
        nonlocal batch, batch_bytes, images_in_series, stats_t0, stats_images, stats_bytes, stats_batches
        if not batch:
            return
        if send_batch([KIND_IMAGE] + batch):
            images_in_series += len(batch)
            if debug:
                stats_images += len(batch)
                stats_bytes += batch_bytes
                stats_batches += 1
                dt = time.monotonic() - stats_t0
                if dt >= 1.0:
                    logger.debug("Receiver: series %s, %d images so far, %.0f images/s, %.1f MB/s, "
                                 "%.1f images/batch", current_series, images_in_series,
                                 stats_images / dt, stats_bytes / dt / 1e6, stats_images / stats_batches)
                    stats_t0 = time.monotonic()
                    stats_images = stats_bytes = stats_batches = 0
        batch = []
        batch_bytes = 0

    def handle(frame):
        nonlocal batch_bytes, current_series, images_in_series
        try:
            msg_type = peek_type(frame.buffer)
        except Exception:
            logger.exception("Could not parse message header; dropping message")
            return

        if msg_type == "image":
            batch.append(frame)
            batch_bytes += len(frame.buffer)
            if len(batch) >= BATCH_MAX_IMAGES or batch_bytes >= BATCH_MAX_BYTES:
                flush_batch()

        elif msg_type in ("start", "end"):
            flush_batch()  # images before start/end must reach the workers first
            raw = frame.bytes
            header = cbor2.loads(raw)  # no tag decoding / decompression
            series_id = header.get("series_id")
            if msg_type == "start":
                if current_series is not None:
                    logger.warning("Start of series %s before end of series %s",
                                   series_id, current_series)
                current_series = series_id
                images_in_series = 0
                broadcast(KIND_START, raw)
                control_queue.put({"type": "start", "series_id": series_id, "raw": raw})
                logger.info("Start of series %s", series_id)
            else:
                broadcast(KIND_END, raw)
                control_queue.put({"type": "end", "series_id": series_id, "raw": raw,
                                   "images_forwarded": images_in_series})
                logger.info("End of series %s (%d images forwarded)", series_id, images_in_series)
                current_series = None
                images_in_series = 0
        else:
            logger.warning("Unknown message type %r; dropping", msg_type)

    try:
        while not stop_event.is_set():
            if not pull.poll(500):
                continue
            # Drain everything already waiting, then send what is left in the batch.
            # When data arrives slowly, batches are small and latency stays low.
            for _ in range(10 * BATCH_MAX_IMAGES):
                try:
                    frame = pull.recv(flags=zmq.NOBLOCK, copy=False)
                except zmq.Again:
                    break
                handle(frame)
            flush_batch()
    except Exception:
        logger.exception("Receiver crashed")
        raise
    finally:
        # Tell every worker to flush and exit (ordered after all data already sent)
        for s in outs:
            if not _send_blocking(s, [KIND_QUIT, b""], timeout=5.0):
                logger.warning("Could not deliver quit message to a worker")
        pull.close(linger=0)
        for s in outs:
            s.close()
        context.term()
        logger.info("Receiver exiting")
