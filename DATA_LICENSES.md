# Data Licensing and Provenance

The repository-level Apache-2.0 license applies to original software,
configuration, and documentation. It does **not** automatically relicense every
field in the frozen JSONL trajectories.

## Rights granted by the LoopStopBench contributors

To the extent that the contributors hold the necessary rights, the original
selection metadata, schema annotations, derived scores, result tables, and
figures in `data/` and `result/` are made available under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).

This grant excludes third-party benchmark content and model-provider material.
Those components retain the terms described below and in
`THIRD_PARTY_NOTICES.md`.

## File-level provenance

### Code trajectories

Files:

- `data/m2_code_8b.jsonl`
- `data/m2_code_32b.jsonl`

Task identifiers, visible tests, and hidden-test construction are derived from
HumanEval+ and MBPP+ as distributed by EvalPlus. The EvalPlus release
repositories use Apache-2.0; HumanEval+ additionally retains the MIT terms of
OpenAI HumanEval. The raw EvalPlus task files and canonical solutions are not
included in this repository.

The trajectory files contain generated candidate programs and derived
pass/failure information. Reusers remain responsible for the upstream
EvalPlus/HumanEval/MBPP terms and for evaluating generated code safely.

### Retrieval QA trajectories

File:

- `data/qa_m2.jsonl`

The evaluated task identifiers and source questions come from the HotpotQA
distractor validation data. HotpotQA data are distributed under CC BY-SA 4.0.
This repository does not include the original HotpotQA task cache or the full
source paragraph collection. To the extent that released records reproduce or
adapt HotpotQA material, the CC BY-SA 4.0 attribution and share-alike conditions
continue to apply.

### Writing trajectories

Files:

- `data/m2_writing_small_judged.jsonl`
- `data/m2_writing_mid_judged.jsonl`
- `data/m2_writing_small_rep8_1.jsonl`
- `data/m2_writing_mid_rep8.jsonl`
- `data/ws_r3.jsonl`
- `data/ws_r4.jsonl`

The writing set combines contributor-authored essay and technical-explanation
prompts with assignments based on arXiv abstracts. The original task cache,
including source abstract text, is not included. arXiv identifiers are
descriptive metadata, but each e-print and abstract remains subject to its
submitter's license and applicable copyright. LoopStopBench does not relicense
those source abstracts.

The JSONL files contain model-generated drafts, critic feedback, judge scores,
and derived annotations. Rights in model outputs can depend on the model license,
provider terms, input material, and applicable law. The CC BY 4.0 grant above
therefore applies only to rights held by the LoopStopBench contributors and is
not a warranty of exclusive rights in every generated passage.

## Model and service terms

No model weights are redistributed. The frozen records were produced with Qwen
and Llama model families and, for limited calibration, externally hosted API
models. Use of the models and any outputs remains subject to the corresponding
model licenses, acceptable-use policies, and service terms listed in
`THIRD_PARTY_NOTICES.md`.

## Attribution

Reuse should cite the LoopStopBench manuscript and the source benchmark(s)
relevant to the reused records. HotpotQA-derived material must retain the
HotpotQA attribution and CC BY-SA 4.0 terms. This document is a provenance
statement, not legal advice; reusers must verify that their intended use
complies with all applicable upstream terms.
