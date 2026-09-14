# LoopStopBench

Code, frozen trajectories, result tables, and figure-generation scripts for:

> **Adaptive Stopping in Iterative LLM Loops: Measuring Regret and Recoverable Policy Gain**

LoopStopBench measures how much stopping utility a policy using output-side
visible history can recover beyond a tuned simple rule. It covers code repair,
critic-driven writing, and retrieval question answering. Policies are evaluated
by offline replay on the same full-horizon trajectories.

Repository: https://github.com/zhaiyz0110/LoopStopBench

## Primary finding

Under the primary setting (`lambda=0.005`, generation-only token cost,
`stop_at_last`), four of five family-specific targets exclude a material
effect. Code-8B total regret remains inconclusive. See
[`result/AUTHORITATIVE_RESULTS.md`](result/AUTHORITATIVE_RESULTS.md) for the
frozen numbers and reporting constraints.

## Repository contents

```text
LoopStopBench/
|-- configs/                 Experiment, model, and replay configurations
|-- data/                    Schema, checksums, and downloaded trajectories
|-- docker/                  Public reference container configuration
|-- result/
|   |-- AUTHORITATIVE_RESULTS.md
|   |-- RESULT_MANIFEST.md
|   |-- *.csv                Frozen numerical results
|   `-- figure/              Paper figures
|-- scripts/                 Collection, replay, analysis, and figure entry points
|-- src/loopstop/            Installable Python package
|-- tests/                   Unit tests over synthetic trajectories
|-- DATA_LICENSES.md         Data-specific rights and upstream restrictions
|-- LICENSE                  Apache-2.0 license for original software
|-- NOTICE                   Scope of the repository-level license
|-- REPRODUCIBILITY.md       Detailed reproduction commands and scope
|-- THIRD_PARTY_NOTICES.md   Benchmark, model, and dependency attribution
`-- CITATION.cff            Machine-readable citation metadata
```

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[full,dev]"
python -m pytest -q
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

The Docker deployment is optional. It is documented in
[`docker/README.md`](docker/README.md). GPU-backed services are required only
for new trajectory collection or critic rescoring; frozen replay, bootstrap
analysis, and figure generation are CPU workloads.

Embedding features default to the public model ID
`sentence-transformers/all-MiniLM-L6-v2`. Collection accepts
`--embedding-model MODEL_ID_OR_PATH`; feature augmentation accepts the same
option and retains `--model` as a compatibility alias. The
`LOOPSTOP_EMBEDDING_MODEL` environment variable can supply a site-specific
cache path without placing private paths in source control.

## Frozen data

Each JSONL row is one round. The schema is defined in
[`src/loopstop/schema.py`](src/loopstop/schema.py) and summarized in
[`data/README.md`](data/README.md).

The nine JSONL files total approximately 281.8 MiB and are distributed outside
the Git repository. Download the versioned trajectory archive from
the repository's [GitHub Releases](https://github.com/zhaiyz0110/LoopStopBench/releases),
extract the files directly into `data/`, and verify them against
[`data/DATA_MANIFEST.sha256`](data/DATA_MANIFEST.sha256). The release asset for
version 0.1.0 is named `loopstopbench-trajectories-v0.1.0.zip`; its archive
checksum is recorded in
[`data/RELEASE_ASSET.sha256`](data/RELEASE_ASSET.sha256). The archive contains
`LoopStopBench-data-v0.1.0/data/`; copy the nine JSONL files from that directory
into the repository's `data/` directory after extraction.

| File | Purpose | Raw trajectories/tasks | Final analysis scope |
|---|---|---:|---|
| `m2_code_8b.jsonl` | Code repair, 8B generator | 353/118 | 350/117 after exclusion |
| `m2_code_32b.jsonl` | Code repair, 32B generator | 354/118 | 351/117 after exclusion |
| `m2_writing_small_judged.jsonl` | Writing, 8B author and 32B critic | 200/100 | all |
| `m2_writing_mid_judged.jsonl` | Writing, 32B author and 8B critic | 200/100 | all |
| `qa_m2.jsonl` | Retrieval QA | 200/200 | all |
| `m2_writing_small_rep8_1.jsonl` | Writing-small repeated-query signals | 200/100 | robustness only |
| `m2_writing_mid_rep8.jsonl` | Writing-mid repeated-query signals | 200/100 | robustness only |
| `ws_r3.jsonl` | Writing-small critic-scale signals | 200/100 | robustness only |
| `ws_r4.jsonl` | Writing-small pairwise-query signals | 200/100 | robustness only |

### Required code validity exclusion

All final code analyses exclude `Mbpp_793` because its cached hidden-test set
is empty. The raw archives retain its three trajectories per model arm for
provenance. Every command that analyzes code must therefore include:

```text
--exclude-tasks Mbpp_793
```

The corrected code result files begin with `result/p0_excl_`. Some mixed CSV
files retain pre-correction code rows; use the file mapping in
[`result/RESULT_MANIFEST.md`](result/RESULT_MANIFEST.md).

## Reproduce the primary analysis

The following command refits the tested visible policy class inside the
task-clustered bootstrap and evaluates all five primary cells:

```bash
mkdir -p reproduced
python scripts/oracle_visible.py \
  --traj data/m2_code_8b.jsonl \
         data/m2_code_32b.jsonl \
         data/m2_writing_small_judged.jsonl \
         data/m2_writing_mid_judged.jsonl \
         data/qa_m2.jsonl \
  --lambda 0.005 \
  --mode stop_at_last \
  --method all \
  --bootstrap 1000 \
  --force-recoverable \
  --exclude-tasks Mbpp_793 \
  --seed 0 \
  --out reproduced/primary_l005.csv
```

Expected primary values are:

| Cell | Formal target | Point | One-sided 95% endpoints | Margin | Verdict |
|---|---|---:|---|---:|---|
| code-8B | Total regret | 0.0099 | [0.0043, 0.0262] | 0.02 | Inconclusive |
| code-32B | Total regret | 0.0029 | [0.0011, 0.0173] | 0.02 | Equivalent |
| writing-small | Recoverable | 0.0111 | upper 0.0314 | 0.05 | Equivalent |
| writing-mid | Recoverable | 0.0025 | upper 0.0220 | 0.05 | Equivalent |
| QA | Recoverable | 0.0000 (raw -0.0019) | upper 0.0172 | 0.02 | Equivalent |

To regenerate the paper figures from the frozen CSV files:

```bash
python scripts/make_figures.py \
  --results-dir result \
  --p0-results-dir result \
  --out-dir reproduced/figures
```

Additional model-class, transfer, signal, cost, and return-rule commands are
listed in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Reproduction scope

The release supports numerical replay and regeneration of the reported tables
and figures from frozen trajectories. It does not promise bitwise regeneration
of the trajectories from raw source tasks: original task caches are not
redistributed, some API model versions are externally hosted, and complete
package/GPU timing snapshots were not retained. New collection runs may
therefore differ from the frozen release. This is a frozen-replay release, not
an end-to-end archival snapshot of trajectory collection.

Stopping policies must access trajectories through `VisibleHistory`; hidden
quality is present for evaluation but is unavailable to policies at inference
time. Collection always runs to the full horizon, and stopping policies only
truncate trajectories during offline replay.

## Security

The code-repair harness evaluates generated Python. Direct subprocess execution
is disabled by default and is not a security sandbox. Read
[`docker/README.md`](docker/README.md) before enabling code execution. Do not
run generated code on a workstation or in the networked collection container.

## License and attribution

Original source code and documentation are licensed under Apache-2.0. Dataset
and artifact terms differ by file and upstream source; read
[`DATA_LICENSES.md`](DATA_LICENSES.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistribution.

## Citation

Use [`CITATION.cff`](CITATION.cff) or cite the associated manuscript. Citation
metadata should be updated with the article DOI and final bibliographic details
after publication.
