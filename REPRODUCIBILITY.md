# Reproducibility Guide

This guide distinguishes analysis reproducibility from trajectory collection.
The repository supports replaying policies, recomputing the reported
statistics, and regenerating figures from the frozen JSONL trajectories. It
does not contain every upstream task cache, model weight, API credential, or
exact service snapshot required to recollect all trajectories byte for byte.

## 1. Environment

Use Python 3.10 or newer. A clean local installation is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[full,dev]"
python -m pytest -q
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

### Recorded experiment environment

The surviving launch configuration and execution logs establish the following
facts about the formal runs:

| Component | Recorded version or status |
|---|---|
| Collection/serving image | `vllm/vllm-openai:v0.9.1` |
| Analysis Python | Python 3.11 |
| PyTorch | Required by the recurrent model ladder; exact formal-run version not retained |
| Matplotlib | Required for released figure generation; exact formal-run version not retained |
| NumPy/SciPy/scikit-learn/LightGBM | Exact formal-run versions not retained |
| CUDA, driver, GPU kernels | Exact snapshot not retained |
| External API models | Provider-side revision snapshots not available |

`pyproject.toml` now declares every direct package needed for frozen analysis,
the recurrent model ladder, and figure generation. Its version constraints are
supported release requirements, not reconstructed claims about unrecorded
formal-run versions. Do not replace the unknown entries above with current
local versions. Small numerical differences remain possible across software
versions and hardware kernels.

Create a separate output directory so the released result files remain
unchanged:

```bash
mkdir -p reproduced
```

## 2. Frozen data and P0 exclusion

The primary analysis uses these five trajectory files:

```text
data/m2_code_8b.jsonl
data/m2_code_32b.jsonl
data/m2_writing_small_judged.jsonl
data/m2_writing_mid_judged.jsonl
data/qa_m2.jsonl
```

Large JSONL files are distributed as a separate release artifact. After
downloading the trajectory bundle, verify it
against `data/RELEASE_ASSET.sha256`, extract it, and then verify its nine JSONL
files against `data/DATA_MANIFEST.sha256`. Place the JSONL files directly under
`data/` before running the commands below.

All final code analyses exclude task `Mbpp_793` because its cached hidden-test
set is empty. The raw archives retain the affected trajectories. The corrected
samples are 350 trajectories/117 tasks for code-8B and 351 trajectories/117
tasks for code-32B. Pass the exclusion explicitly to analysis commands.

## 3. Primary analysis

Run the primary decomposition and task-clustered bootstrap with:

```bash
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

The primary formal targets should agree, up to ordinary floating-point and
library-version variation, with the following values:

| Cell | Formal target | Point | BCa endpoint(s) | Verdict |
|---|---|---:|---|---|
| code-8B | Total regret | 0.0099 | [0.0043, 0.0262] | Inconclusive |
| code-32B | Total regret | 0.0029 | [0.0011, 0.0173] | Equivalent |
| writing-small | Recoverable | 0.0111 | upper 0.0314 | Equivalent |
| writing-mid | Recoverable | 0.0025 | upper 0.0220 | Equivalent |
| QA | Recoverable | 0.0000 (raw -0.0019) | upper 0.0172 | Equivalent |

The materiality margins are 0.02 for code and QA and 0.05 for writing. Code
uses total regret as its formal target; writing and QA use recoverable gain.

## 4. Cost-weight sensitivity

Repeat the command above with `--lambda 0`, `--lambda 0.005`, and
`--lambda 0.02`, changing `--out` for each run. Keep `--exclude-tasks
Mbpp_793` in every run. The verdict matrix and exact endpoints are in
`result/AUTHORITATIVE_RESULTS.md`.

## 5. Model-class ladder

Run one trajectory cell at a time. For example:

```bash
python scripts/oracle_visible.py \
  --traj data/m2_code_8b.jsonl \
  --lambda 0.005 \
  --mode stop_at_last \
  --model-class \
  --mc-label utility \
  --mc-bootstrap 1000 \
  --exclude-tasks Mbpp_793 \
  --seed 0 \
  --out reproduced/model_class_code8.csv
```

Use the same structure for the other four primary files. The truth-fed
bidirectional GRU is a positive control and is not part of the deployable
output-side policy class.

## 6. Leave-one-family-out VOI

```bash
python scripts/train_voi.py \
  --traj data/m2_code_8b.jsonl \
         data/m2_code_32b.jsonl \
         data/m2_writing_small_judged.jsonl \
         data/m2_writing_mid_judged.jsonl \
         data/qa_m2.jsonl \
  --held-out-loop all \
  --utility-eval \
  --lambda 0.005 \
  --bootstrap 1000 \
  --exclude-tasks Mbpp_793 \
  --seed 0 \
  --util-out reproduced/xloop_l005.csv
```

Repeat at `--lambda 0` and `--lambda 0.02` for the full sensitivity analysis.
The deployable result is the transfer-threshold policy. The oracle-threshold
sweep is an optimistic, non-deployable diagnostic.

## 7. Signal and protocol analyses

The critic-capacity runs use `data/ws_r3.jsonl` and one of these score fields:

```text
score_critic_gen_8b
score_critic_gen_32b
score_critic_72b
```

Pass the selected field with `--score-field`, together with `--method all`,
`--bootstrap 1000`, `--lambda 0.005`, and a unique output name. The pairwise
protocol comparison uses:

```bash
python scripts/oracle_visible.py \
  --traj data/ws_r4.jsonl \
  --lambda 0.005 \
  --method all \
  --bootstrap 1000 \
  --compare-score-fields score_critic_gen_32b score_pw_gen_32b \
  --seed 0 \
  --out reproduced/pairwise_protocol.csv
```

Repeated-query comparisons use `--compare-score-fields score score_rep8` on
`data/m2_writing_small_rep8_1.jsonl` and
`data/m2_writing_mid_rep8.jsonl`. These paired comparisons are exploratory.

## 8. Alternative cost and return rules

For the full writing cost, analyze the two primary writing files with
`--include-critic-cost`. The repeated-query critic calls are not included in
the utility cost because the stored cost fields do not attribute those calls
per stopping decision.

For `stop_and_select`, repeat the primary command with `--mode
stop_and_select`; retain `--exclude-tasks Mbpp_793` for code.

## 9. Figures

Generate the released figure set with explicit public-release paths:

```bash
python scripts/make_figures.py \
  --results-dir result \
  --p0-results-dir result \
  --out-dir reproduced/figures
```

The explicit arguments above match the packaged repository defaults and make
the input/output locations visible in execution logs.

## 10. Descriptive capability analysis

`result/capability_difficulty_p0.csv` contains the P0-corrected capability and
difficulty summary consumed by optional Figure 5. It excludes `Mbpp_793` and
reports a paired 8B-minus-32B regret difference over 117 tasks. Reconstructing
the model-specific difficulty tiers requires the original
`code_tasks_m2.json` task-pool cache, which is not redistributed because it
contains upstream benchmark material. The released CSV allows inspection and
figure regeneration but not reconstruction of those tier assignments from
the public files alone.

## 11. Result provenance and limitations

- Use `result/AUTHORITATIVE_RESULTS.md` for manuscript numbers.
- `result/RESULT_MANIFEST.md` maps claims to released files and identifies
  superseded rows.
- Bootstrap resampling is clustered by task, while reported utility is the
  mean over trajectories.
- Different experiment families refit different parts of the analysis
  pipeline; consult the manuscript and source code before comparing intervals.
- Frozen replay is deterministic conditional on data, seed, dependencies, and
  hardware kernels, but exact bitwise equality is not guaranteed.
- Trajectory recollection may require separately licensed benchmark data,
  model access, and service credentials. See `DATA_LICENSES.md` and
  `THIRD_PARTY_NOTICES.md`.

## Security

Code-repair trajectories may contain model-generated Python. Do not execute
untrusted generated code on a host system. Use an isolated, disposable
environment with networking disabled and only a temporary working directory
mounted writable.
