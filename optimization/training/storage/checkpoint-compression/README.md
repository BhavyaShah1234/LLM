# Checkpoint Compression (training, storage)

Status: **done (Wave 6)**.

`checkpoint_compression_benchmark.py` saves the same real checkpoint in
three ways -- fp32 safetensors, bf16 safetensors, and a gzip-compressed
archive of the bf16 weights file -- and measures actual on-disk bytes and
load time for each, via real files (not estimates). Uses this project's
own from-scratch CLM checkpoint (`./output/pretraining/clm`).

## Real result

| Format | Size | vs. fp32 | Load time |
|---|---|---|---|
| fp32 | 207.9 MB | -- | 0.024s |
| bf16 | 105.7 MB | -49.1% | 0.020s |
| gzip(bf16) | 81.2 MB | -60.9% | 0.437s decompress + 0.020s load = 0.457s |

**bf16 is a strictly better default than fp32 here**: half the disk size,
and (if anything) marginally faster to load, since there's simply less
data to read off disk -- no downside for a checkpoint that was trained in
bf16 to begin with (this project's convention throughout).
**gzip adds a real but secondary win** (23.2% smaller than bf16 alone) at
a real, substantial cost: decompression took **~22x longer than the load
itself** (0.437s vs. 0.020s) for this small 51M-parameter model, and that
cost scales with checkpoint size -- for a multi-GB checkpoint, gzip
decompression before every load would very plausibly dominate total
load time. Compression is worth it for archival/transfer (where the
checkpoint is written once and read rarely) but a poor default for a
checkpoint loaded repeatedly (e.g. every experiment run in this project's
own `experiments/` scripts) -- read time then adds up every single time.

## Usage

```bash
python checkpoint_compression_benchmark.py --debug_first_batch
python checkpoint_compression_benchmark.py
```

No dataset needed -- this benchmark measures real file sizes and load
times of an already-trained checkpoint, not model quality.
