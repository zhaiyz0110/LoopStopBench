# Frozen trajectory data

## Download and verification

The trajectory JSONL files are distributed separately from the Git repository
because the complete set is approximately 281.8 MiB and several individual
files exceed GitHub's browser-upload limit. Download
`loopstopbench-trajectories-v0.1.0.zip` from the repository's
[GitHub Releases](https://github.com/zhaiyz0110/LoopStopBench/releases), then
verify the archive itself against `RELEASE_ASSET.sha256`. After extraction,
copy the nine files from `LoopStopBench-data-v0.1.0/data/` into this directory.

On Linux or macOS, verify the extracted files from this directory with:

```bash
sha256sum --check DATA_MANIFEST.sha256
```

On Windows PowerShell, compare `Get-FileHash -Algorithm SHA256 *.jsonl` with
`DATA_MANIFEST.sha256`. Do not use a file whose checksum differs.

The writing variants remain self-contained rather than sharing deduplicated
base records. This preserves the frozen schema, per-file checksums, and direct
compatibility with the analysis commands. ZIP compression removes most of the
repeated text, so a custom merged storage format would add complexity without
meaningfully reducing the release asset.

This directory contains the frozen per-round trajectories used by the paper.
Each JSONL row records one round; rows are grouped into trajectories by
`traj_id`.

The release does not include `data/raw/` task caches. It therefore supports
offline replay of the frozen trajectories but not exact reconstruction of every
collection prompt from the original benchmark downloads.

## Schema

The Python schema is defined in `src/loopstop/schema.py`. Version 1.1 records:

```json
{
  "traj_id": "...",
  "task_id": "...",
  "loop_type": "code_repair",
  "model_cfg": "code_small",
  "seed": 0,
  "t": 1,
  "output": "...",
  "diff_vs_prev": "...",
  "visible_signals": {
    "score": 0.6,
    "feedback_text": "...",
    "edit_dist_prev": null,
    "emb_dist_prev": null,
    "output_len": 512,
    "mean_logprob": -0.42
  },
  "hidden_truth": {"r": 0.55},
  "tokens_in": 3012,
  "tokens_out": 980,
  "latency_ms": 4210.0,
  "timestamp": 1780000000.0,
  "schema_version": "1.1"
}
```

Loop-specific visible and hidden fields are documented in the manuscript and
implemented in `src/loopstop/loops/`.

## Evaluation boundary

Policies may observe only `VisibleHistory`. They must never read
`hidden_truth`, which is retained solely for offline evaluation. Collection
runs to the full configured horizon; a replayed stopping policy selects a
prefix without changing later states.

## Code validity exclusion

The raw code files include `Mbpp_793`, whose cached hidden-test set is empty.
All final code analyses exclude it with:

```text
--exclude-tasks Mbpp_793
```

This leaves 350 code-8B trajectories and 351 code-32B trajectories over 117
valid tasks. See `result/AUTHORITATIVE_RESULTS.md` for the frozen estimates.

## Provenance and licensing

These files contain model-generated text, benchmark-derived identifiers and
feedback, and author-generated annotations. They are not covered wholesale by
the repository's Apache-2.0 code license. Read `DATA_LICENSES.md` and
`THIRD_PARTY_NOTICES.md` before reuse or redistribution.
