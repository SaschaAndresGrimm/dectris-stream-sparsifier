# tools/fake_detector.py
"""Fake detector: serves a synthetic Stream v2 series on a ZMQ PUSH socket.

    python tools/fake_detector.py -d 4M -n 2000           # terminal 1
    python main.py localhost -o /tmp/bench                # terminal 2

Sends start, N images (a few pre-built frames, cycled) as fast as the
receiver accepts them (or at --rate Hz), then end. The printed send rate
includes ZMQ/worker buffering (~100s of frames), so use enough images; the
real end-to-end rate is in the sparsifier log ("... fps ...") and in
<prefix>_summary.json ("frames_per_second").
"""
import argparse
import os
import sys
import time

import numpy as np
import zmq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import end_message, image_message, make_frame, parse_shape, start_message  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--detector", default="4M", help="preset (SELUN, 1M, 4M, 16M) or HxW (default: 4M)")
    p.add_argument("-t", "--dtype", default="uint32")
    p.add_argument("-f", "--hit-fraction", type=float, default=0.01)
    p.add_argument("-n", "--images", type=int, default=1000)
    p.add_argument("-r", "--rate", type=float, default=0, help="frame rate in Hz (0 = max)")
    p.add_argument("-p", "--port", type=int, default=31001)
    p.add_argument("-s", "--series", type=int, default=1, help="number of series to send")
    args = p.parse_args()

    shape = parse_shape(args.detector)
    print(f"Building frames {shape} {args.dtype}, {args.hit_fraction * 100:g}% hits...")
    frames = [make_frame(shape, args.hit_fraction, np.dtype(args.dtype), seed=i) for i in range(4)]

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 100)
    sock.bind(f"tcp://*:{args.port}")
    print(f"Listening on tcp://*:{args.port}; waiting 2 s for the sparsifier to connect")
    time.sleep(2)

    for series_id in range(1, args.series + 1):
        msgs = [image_message(series_id, 0, f) for f in frames]  # image_id is not unique; fine here
        size = np.mean([len(m) for m in msgs])
        sock.send(start_message(series_id))
        t0 = time.perf_counter()
        for i in range(args.images):
            if args.rate:
                delay = t0 + i / args.rate - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            sock.send(msgs[i % len(msgs)], copy=False)
        dt = time.perf_counter() - t0
        sock.send(end_message(series_id))
        print(f"Series {series_id}: sent {args.images} images ({size / 1e6:.2f} MB each) in {dt:.1f} s "
              f"= {args.images / dt:.0f} fps, {args.images * size * 8 / dt / 1e9:.2f} Gbit/s, "
              f"{args.images * frames[0].size / dt / 1e9:.2f} Gpx/s")
    # Stay connected like a real detector: closing right away can drop the
    # last queued messages (and 'end') while they are still in flight.
    print("Done; keeping the connection open for 10 s")
    time.sleep(10)
    sock.close(linger=-1)
    ctx.term()


if __name__ == "__main__":
    main()
