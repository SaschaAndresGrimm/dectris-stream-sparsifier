# main.py
"""Orchestrates one receiver process and N worker processes.

    detector --PUSH/PULL--> receiver --(per-worker ipc PUSH)--> workers --> shard files
                               |                                   |
                               +------ control_queue (small) ------+--> main

A series is finalized (shards merged into .npy files) only once the receiver
has seen 'end' AND every worker has confirmed it processed all of its images.
"""
import argparse
import json
import logging
import os
import shutil
import signal
import sys
import tempfile
from datetime import datetime
from logging.handlers import QueueListener
from multiprocessing import Event, Process, Queue
from queue import Empty

from logging_setup import setup_logging
from output import merge_series, save_message
from receiver import receiver
from stream_decode import decode_message
from worker import series_shard_dir, worker


class Series:
    def __init__(self, series_id):
        self.series_id = series_id
        self.prefix = f"{datetime.now():%Y%m%d_%H%M%S}_series{series_id}"
        self.end_received = False
        self.images_forwarded = None
        self.done = {}  # worker_idx -> list of reports


class Coordinator:
    def __init__(self, num_workers, output_dir, shard_root, keep_shards, logger):
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.shard_root = shard_root
        self.keep_shards = keep_shards
        self.logger = logger
        self.series = {}

    def _get(self, series_id):
        if series_id not in self.series:
            self.series[series_id] = Series(series_id)
        return self.series[series_id]

    def _base(self, s):
        return os.path.join(self.output_dir, s.prefix)

    def handle(self, msg):
        kind = msg["type"]
        s = self._get(msg["series_id"])
        if kind == "start":
            save_message(decode_message(msg["raw"]), f"{self._base(s)}_start")
            self.logger.info("Series %s started; saved start metadata (%s_start.json)",
                             s.series_id, s.prefix)
        elif kind == "end":
            s.end_received = True
            s.images_forwarded = msg["images_forwarded"]
            save_message(decode_message(msg["raw"]), f"{self._base(s)}_end")
            self.logger.info("Series %s: end received, waiting for workers to finish", s.series_id)
        elif kind == "worker_done":
            s.done.setdefault(msg["worker"], []).append(msg)
        else:
            self.logger.warning("Unknown control message %r", kind)
            return

        if s.end_received and len(s.done) == self.num_workers:
            self.finalize(s.series_id, partial=False)

    def finalize(self, series_id, partial):
        s = self.series.pop(series_id)
        base = self._base(s)
        self.logger.info("Series %s: merging worker shards...", series_id)
        n_frames, hits = merge_series(series_shard_dir(self.shard_root, series_id), base,
                                      keep_shards=self.keep_shards)

        if s.images_forwarded is not None and n_frames != s.images_forwarded:
            self.logger.warning("Series %s: %d images forwarded but %d frames written",
                                series_id, s.images_forwarded, n_frames)
        summary = {
            "series_id": series_id,
            "partial": partial,
            "images_forwarded": s.images_forwarded,
            "frames_written": n_frames,
            "hits": hits,
            "frames_per_worker": {str(w): sum(r["frames"] for r in reports)
                                  for w, reports in sorted(s.done.items())},
        }
        with open(f"{base}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        level = logging.WARNING if partial else logging.INFO
        self.logger.log(level, "Series %s %s: %d frames, hits per threshold %s -> %s_*",
                        series_id, "INCOMPLETE (saved partial data)" if partial else "complete",
                        n_frames, hits, base)

    def finalize_all_partial(self):
        for series_id in list(self.series):
            self.finalize(series_id, partial=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Receive a DECTRIS Stream v2 and save sparse hit lists.")
    parser.add_argument("--hostname", type=str, default="localhost",
                        help="Hostname or IP address to connect to (default: localhost)")
    parser.add_argument("--port", type=int, default=31001,
                        help="Port number to connect to (default: 31001)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of worker processes to spawn (default: 8)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for output files (default: current directory)")
    parser.add_argument("--keep-shards", action="store_true",
                        help="Keep per-worker shard files after merging (for debugging)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging (debug mode)")
    return parser.parse_args()


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    args = parse_args()
    # Ctrl+C and `kill` (SIGTERM) both trigger a clean shutdown that saves data.
    # (SIGINT is set explicitly because it is inherited as ignored under nohup / `&`.)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    os.makedirs(args.output_dir, exist_ok=True)

    log_queue = Queue()
    setup_logging(log_queue=None, level=log_level,
                  log_file=os.path.join(args.output_dir, "pilatus4_stream.log"))
    logger = logging.getLogger("main")
    listener = QueueListener(log_queue, *logging.getLogger().handlers, respect_handler_level=True)
    listener.start()

    shard_root = os.path.join(args.output_dir, ".shards")
    if os.path.isdir(shard_root) and os.listdir(shard_root):
        stale = f"{shard_root}_stale_{datetime.now():%Y%m%d_%H%M%S}"
        os.rename(shard_root, stale)
        logger.warning("Found leftover shards from a previous run; moved to %s", stale)
    os.makedirs(shard_root, exist_ok=True)

    ipc_dir = tempfile.mkdtemp(prefix="sparsificator_")
    worker_endpoints = [f"ipc://{ipc_dir}/w{i}" for i in range(args.num_workers)]
    endpoint = f"tcp://{args.hostname}:{args.port}"

    stop_event = Event()
    control_queue = Queue()
    coord = Coordinator(args.num_workers, args.output_dir, shard_root, args.keep_shards, logger)

    procs = []
    for i, ep in enumerate(worker_endpoints):
        p = Process(target=worker, name=f"Worker-{i}",
                    args=(i, ep, shard_root, control_queue, stop_event, log_queue, log_level))
        p.start()
        procs.append(p)
    rp = Process(target=receiver, name="Receiver",
                 args=(endpoint, worker_endpoints, control_queue, stop_event, log_queue, log_level))
    rp.start()
    procs.append(rp)

    exit_code = 0
    try:
        logger.info("Running with %d workers, connected to %s. Press Ctrl+C to stop.",
                    args.num_workers, endpoint)
        while True:
            try:
                coord.handle(control_queue.get(timeout=0.5))
            except Empty:
                dead = [p.name for p in procs if not p.is_alive()]
                if dead:
                    logger.error("Process(es) died unexpectedly: %s. Shutting down.", ", ".join(dead))
                    exit_code = 1
                    break
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: stopping receiver and workers...")
    except Exception:
        logger.exception("Main loop crashed; shutting down")
        exit_code = 1

    # Shutdown: keep draining the control queue while children flush and exit
    # (joining a process that still has unflushed queue items can deadlock).
    stop_event.set()
    try:
        while any(p.is_alive() for p in procs):
            try:
                coord.handle(control_queue.get(timeout=0.2))
            except Empty:
                pass
        while True:
            try:
                coord.handle(control_queue.get(timeout=0.2))
            except Empty:
                break
        for p in procs:
            p.join()
        coord.finalize_all_partial()  # series interrupted before their end message
    except KeyboardInterrupt:
        logger.warning("Second interrupt: terminating children without saving open series")
        for p in procs:
            p.terminate()
        exit_code = 1
    finally:
        shutil.rmtree(ipc_dir, ignore_errors=True)
        logger.info("All processes stopped. Exiting.")
        listener.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
