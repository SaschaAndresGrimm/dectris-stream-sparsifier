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

python main.py --hostname <hostname> --port <port> --num-workers <n> [--output-dir DIR] [--keep-shards] [--verbose]

Example:

python main.py --hostname 192.168.1.100 --port 31001 --num-workers 8 --output-dir /data/run1

Command-Line Arguments

	•	--hostname: Hostname or IP address of the detector (default: localhost)
	•	--port: Stream port (default: 31001)
	•	--num-workers: Number of worker processes (default: 8)
	•	--output-dir: Where output and the log file are written (default: current directory)
	•	--keep-shards: Keep per-worker intermediate files after merging (debugging)
	•	--verbose / -v: Debug logging (slow; not for production rates)

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
	•	<prefix>_summary.json : frame/hit counts, frames per worker, and whether
	        the series is "partial" (interrupted before its 'end' message)

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
	    without saving.
	•	Intermediate files live in <output-dir>/.shards. If a previous run crashed,
	    they are moved to .shards_stale_<timestamp> on the next start.

Troubleshooting

	•	Check pilatus4_stream.log in the output directory.
	•	"N images forwarded but M frames written" means images failed to decode/process.
	    Look for worker errors in the log.
	•	If a worker or the receiver dies, main logs it, saves what it has, and exits with code 1.
