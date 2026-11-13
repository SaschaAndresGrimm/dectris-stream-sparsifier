# tools/bench_worker.py
"""Per-core cost of one worker: decode (CBOR + bslz4) and sparsify, per frame.

    python tools/bench_worker.py                      # SELUN, 1M, 4M, 16M
    python tools/bench_worker.py -d 4M 512x512 -t uint16 -f 0.01

Multiply fps/core by the number of workers for a rough upper bound on the
pipeline (until you run out of physical cores or memory bandwidth).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from processing import process_frames  # noqa: E402
from receiver import BATCH_MAX_IMAGES  # noqa: E402
from worker import STACK_MAX_BYTES  # noqa: E402
from stream_decode import decode_message  # noqa: E402
from synth import image_message, make_frame, parse_shape  # noqa: E402


def timeit(fn, n):
    fn()
    if not n:  # calibrate: ~0.3 s per measurement
        t0 = time.perf_counter()
        fn()
        n = max(3, int(0.3 / max(time.perf_counter() - t0, 1e-6)))
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--detectors", nargs="+", default=["SELUN", "1M", "4M", "16M"],
                   help="presets (SELUN, 1M, 4M, 16M) or HxW (default: all presets)")
    p.add_argument("-t", "--dtypes", nargs="+", default=["uint16", "uint32"])
    p.add_argument("-f", "--hit-fractions", nargs="+", type=float, default=[0.001, 0.01, 0.1])
    p.add_argument("-n", "--repeat", type=int, default=0, help="frames per timing (default: ~0.3 s worth)")
    args = p.parse_args()

    print(f"{'detector':>10} {'dtype':>6} {'hits':>6} {'msg kB':>8} {'ratio':>6} "
          f"{'decode us':>10} {'sparse us':>10} {'fps/core':>8} {'Mpx/s/core':>10} {'Gbit/s/core':>11}")
    for det in args.detectors:
        shape = parse_shape(det)
        for dt in args.dtypes:
            for frac in args.hit_fractions:
                frame = make_frame(shape, frac, np.dtype(dt))
                msg = image_message(1, 0, frame)
                data = decode_message(msg)["data"]["threshold_1"]
                t_dec = timeit(lambda: decode_message(msg), args.repeat)
                # Small frames are sparsified in batches (as in the worker): time per frame.
                n = BATCH_MAX_IMAGES if data.nbytes < STACK_MAX_BYTES else 1
                stack, ids = np.stack([data] * n), list(range(n))
                t_sp = timeit(lambda: process_frames(stack, ids), args.repeat) / n
                t = t_dec + t_sp
                print(f"{det:>10} {dt:>6} {frac * 100:5.1f}% {len(msg) / 1e3:8.1f} "
                      f"{frame.nbytes / len(msg):5.0f}x {t_dec * 1e6:10.0f} {t_sp * 1e6:10.0f} "
                      f"{1 / t:8.0f} {frame.size / t / 1e6:10.0f} {len(msg) * 8 / t / 1e9:11.2f}",
                      flush=True)


if __name__ == "__main__":
    main()
