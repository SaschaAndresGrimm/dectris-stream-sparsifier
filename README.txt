dectris-stream-sparsifier — DECTRIS Stream v2 sparsifier by Sascha Grimm

This application is a teaching example that receives image data from a DECTRIS
detector over ZeroMQ (Stream v2 / CBOR), converts every frame into a sparse
list of "hit" pixels using multiple worker processes, and saves one set of
files per data series. It keeps running and processes series after series
until stopped with Ctrl+C (or SIGTERM).

Prerequisites

	•	Python 3.8 or higher
	•	numpy, pyzmq, cbor2 (5.x or 6.x), dectris-compression

pip install numpy pyzmq cbor2 dectris-compression

Usage

python main.py HOST [-p PORT] [-n NUM_WORKERS] [-o OUTPUT_DIR] [-k] [-v]

Examples:

python main.py 192.168.1.100
python main.py http://192.168.1.100/ -n 16 -o /data/run1
python main.py 192.168.1.100:31001 -v

Command-Line Arguments

	•	HOST (required): Hostname or IP of the detector. A scheme (http://, tcp://),
	    trailing path and ":port" are accepted and stripped, so you can paste the
	    detector URL directly.
	•	-p / --port: Stream port (default: 31001, or the port given in HOST)
	•	-n / --num-workers: Number of worker processes (default: 8)
	•	-o / --output-dir: Where output and the log file are written (default: current directory)
	•	-k / --keep-shards: Keep per-worker intermediate files after merging (debugging)
	•	-v / --verbose: Debug logging: per-image hit counts and timings from the workers, receiver
	    throughput once per second, worker completion reports (slow; not for production rates)

Architecture

    detector --PUSH/PULL--> receiver --(one ipc socket per worker)--> workers --> shard files on disk
                               |                                         |
                               +---------- control queue (small) --------+--> main

Why a single receiver? The detector's PUSH socket distributes messages
round-robin over all connected clients. If each worker connected directly,
the 'start' and 'end' messages would each reach only ONE worker, and the
others would not know when a series ends. Data from the end of a series would
then be lost or end up in the next series.

	•	receiver.py: The only process connected to the detector. It reads the message
	    type without decompressing the payload. It forwards 'image' messages round-robin
	    to the workers, skipping busy ones, and broadcasts 'start' and 'end' to EVERY
	    worker. Each worker therefore sees 'end' only after all images routed to it.
	    Images already waiting on the socket are sent as one batch (up to 256 images
	    or 1 MB), which keeps the per-message cost low at high frame rates.
	•	worker.py: Decodes and decompresses images, finds hit pixels (processing.py), and
	    appends them to its own binary shard files. Pixel data never goes back to the
	    main process. On 'end' it flushes and reports a short summary.
	•	main.py: Saves start/end metadata. When the receiver has seen 'end' AND every
	    worker has confirmed the series, it merges the shards into the final files.
	•	processing.py: process_frame() keeps pixels with 0 < value < dtype max that are
	    not set in the detector pixel_mask from the start message.
	•	stream_decode.py: CBOR tag decoders (typed arrays, multi-dim arrays, DECTRIS compression).
	•	output.py: Metadata export and shard merging.

Output Files (per series; <prefix> = <YYYYmmdd_HHMMSS>_series<series_id>)

	•	<prefix>_hits_<threshold>.npy : one row per hit pixel
	        dtype: image_id (u4), x (u2), y (u2), value (u4)   (12 bytes/hit)
	        Rows are grouped by worker, not sorted. Sort by image_id if you need order.
	•	<prefix>_frames.npy : one row per frame, sorted by image_id
	        dtype: image_id (u4), start_time, stop_time, real_time (f8, seconds)
	•	<prefix>_start.json / <prefix>_end.json : message metadata
	•	<prefix>_start_arrays.npz : arrays from the start message (pixel_mask, flatfield, ...)
	•	<prefix>_summary.json : frame/hit counts, frames per worker, processing rate
	        (frames_per_second), and whether the series is "partial" (interrupted
	        before its 'end' message)

Loading the data:

import numpy as np
hits   = np.load("20261112_102423_series1_hits_threshold_1.npy", mmap_mode="r")
frames = np.load("20261112_102423_series1_frames.npy")
hits = hits[np.argsort(hits["image_id"], kind="stable")]       # time order
t = frames["start_time"][np.searchsorted(frames["image_id"], hits["image_id"])]  # timestamp per hit
# Export to CSV if needed:
np.savetxt("hits.csv", np.column_stack([hits[n] for n in hits.dtype.names]),
           fmt="%d", delimiter=",", header="image_id,x,y,value", comments="")

Stopping

	•	Press Ctrl+C (or send SIGTERM). The receiver stops, the workers process everything
	    already queued and flush their data, and any series in progress is saved
	    and marked "partial" in its summary. A second Ctrl+C aborts immediately
	    without saving. Ctrl+C during a merge waits until the merge is finished.
	•	Intermediate files live in <output-dir>/.shards. If a previous run crashed,
	    they are moved to .shards_stale_<timestamp> on the next start.

Performance

Rule of thumb: cost scales with PIXELS per second (frame size x frame rate),
not with network Gbit/s. Sparse frames compress 20-100x, so one worker core
handles only ~0.2-0.6 Gbit/s of wire data but ~250 Mpx/s of decompressed pixels.

Measured on an Apple M2 laptop (8 cores), synthetic frames, 1% hit pixels
(uint32; SELUN uint16):

	                 per worker core     end-to-end, 6 workers (whole laptop)
	    SELUN (95x95)   ~43,000 fps          ~90,000 fps   (laptop CPU full, see below)
	    1M  (1062x1028)    245 fps              ~800 fps
	    4M  (2162x2068)     61 fps              ~200 fps
	    16M (4362x4148)     13 fps               ~46 fps

	•	Per frame, about 2/3 of the time is decompression (bslz4) and 1/3 is sparsifying.
	•	The receiver passes ~8 Gbit/s for large frames. For small frames it needs ~2 us of
	    Python per message plus ZMQ network time: about one full core at 90 kHz.
	•	Expectation: a ~2.5 Gpx/s detector (e.g. 4M at ~560 Hz) needs ~10-15 server cores.
	    A 160 Gbit/s-class system (~4x that) needs 40+ cores and a different input design.
	•	SELUN at 120 kHz is only ~1 Gpx/s, but 120,000 messages/s. Small, fast detectors
	    are limited by per-message overhead, not by pixels; hence the batching. On the
	    laptop, ~90 kHz uses all 8 cores (fake detector + receiver + 6 workers), so
	    120 kHz should fit a server with ~8 free cores (not yet verified).

Bottlenecks, in the order you hit them:
	1.	Worker CPU (decompression + sparsify): add workers, up to the physical core count.
	2.	The single receiver: ~8-15 Gbit/s for large frames, ~100+ kHz for small frames.
	3.	Memory bandwidth / NUMA: decompressed frames are read several times.
	4.	Disk: 12 bytes per hit; only matters for dense data (>~1 GB/s at 10% hits).

How it could be improved:
	•	Remove the receiver hop: workers connect to the detector directly (the detector
	    already spreads images over all clients) and main broadcasts start/end. A series
	    is then complete once all workers are drained, not when 'end' arrives.
	•	High frame rates: per-message decode in the worker (CBOR + decompress, ~13 us for
	    a SELUN frame) is now the largest cost; a compiled decoder would help most.
	•	Faster decompression: a SIMD bitshuffle build, decompressing into a reused buffer,
	    or skipping blocks of zeros without decompressing them.
	•	Fuse decompression and sparsify in compiled code (numba/C), so a frame is
	    read once from cache rather than several times from RAM.
	•	Pin workers to cores and to the NIC's NUMA node; use uint16 mode when possible.

Benchmark tools (need: pip install lz4)

	python tools/bench_worker.py              # per-core cost for SELUN/1M/4M/16M, uint16/32, 0.1/1/10% hits
	python tools/bench_worker.py -d 512x512 -t uint16 -f 0.05   # custom geometry

	python tools/fake_detector.py -d 4M -n 2000     # terminal 1: synthetic detector on port 31001
	python tools/fake_detector.py -d SELUN -t uint16 -n 300000      # high frame rate case
	python main.py localhost -o /tmp/bench          # terminal 2: end-to-end; see "fps" in the log

	Use enough images (thousands): the fake detector's own send rate includes buffering.
	It stays connected for 10 s after 'end' (exiting at once can drop the last messages).
	The real rate is logged at the end of each series and stored in <prefix>_summary.json.

Troubleshooting

	•	Check sparsifier.log in the output directory.
	•	"N images forwarded but M frames written" means images failed to decode/process.
	    Look for worker errors in the log.
	•	If a worker or the receiver dies, main logs it, saves what it has, and exits with code 1.
