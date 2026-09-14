# Result Manifest

This file maps each paper analysis to its released result artifacts. Use
`AUTHORITATIVE_RESULTS.md` for reported values and the CSV files for the
machine-readable records.

## Primary and cost-weight analyses

| Analysis | Authoritative files |
|---|---|
| Primary code, lambda 0.005 | `p0_excl_code_l005_b1000.csv`, `p0_excl_code_l005_b1000_bootstrap.csv` |
| Code, lambda 0 | `p0_excl_code_l000_b1000.csv`, `p0_excl_code_l000_b1000_bootstrap.csv` |
| Code, lambda 0.02 | `p0_excl_code_l020_b1000.csv`, `p0_excl_code_l020_b1000_bootstrap.csv` |
| Writing and QA, lambda 0/0.005/0.02 | `oof_b1000_l000*`, `oof_b1000_l005*`, `oof_b1000_l020*`, supplemented by `oof_tri_l000_bootstrap.csv`, `oof_tri_l005_code8_bootstrap.csv`, and `oof_tri_l020_bootstrap.csv` only where the authoritative table says an endpoint came from the tri-lambda run |

Code rows inside the mixed `oof_b1000_l*.csv` and `oof_tri_l*` files predate the
P0 correction. Final code results use files beginning with `p0_excl_`.

## Model-class ladder

| Cells | Authoritative files |
|---|---|
| code-8B and code-32B | `p0_excl_mc3_m2_code_8b.csv`, `p0_excl_mc3_m2_code_32b.csv` |
| writing-small and writing-mid | `mc3_m2_writing_small_judged.csv`, `mc3_m2_writing_mid_judged.csv` |
| QA | `mc3_qa_m2.csv` |

Any code rows in older `mc3_*` files are superseded by the `p0_excl_mc3_*`
files.

## Cross-loop VOI

| Lambda | Authoritative file |
|---:|---|
| 0 | `xloop_voi_oof_b1000_l000.csv` |
| 0.005 | `xloop_voi_oof_b1000_l005.csv` |
| 0.02 | `xloop_voi_oof_b1000_l020.csv` |

The transfer-threshold result is deployable. The oracle-threshold result is an
optimistic diagnostic selected on the held-out family.

## Signal, critic, and query protocol

| Analysis | Authoritative files |
|---|---|
| 8B critic | `oof_signal_critic8_b1000.csv`, `oof_signal_critic8_b1000_bootstrap.csv` |
| 32B critic | `oof_signal_critic32_b1000.csv`, `oof_signal_critic32_b1000_bootstrap.csv` |
| 72B critic | `oof_signal_critic72_b1000.csv`, `oof_signal_critic72_b1000_bootstrap.csv` |
| 32B pairwise critic | `oof_signal_pairwise32_b1000.csv`, `oof_signal_pairwise32_b1000_bootstrap.csv` |
| Paired absolute-versus-pairwise contrast | `oof_pair_protocol_b1000.csv` |
| Writing-small repeated query | `oof_repeat_ws_pair_b1000.csv` |
| Writing-mid repeated query | `oof_repeat_wm_pair_b1000.csv` |

Paired score-field comparisons are post-hoc exploratory analyses.

## Cost and return-rule sensitivity

| Analysis | Authoritative files |
|---|---|
| Author plus critic token cost | `oof_full_cost_b1000.csv`, `oof_full_cost_b1000_bootstrap.csv` |
| `stop_and_select`, code | `p0_excl_code_stopselect_b1000.csv`, `p0_excl_code_stopselect_b1000_bootstrap.csv` |
| `stop_and_select`, writing and QA | `oof_stop_and_select_b1000.csv`, `oof_stop_and_select_b1000_bootstrap.csv` |

Code rows in `oof_stop_and_select_b1000*` are pre-P0 and superseded.

## Descriptive diagnostics

| Diagnostic | File |
|---|---|
| Corrected code-8B heuristic regret table | `regret_m2_code_8b_stop_at_last.csv` |
| Corrected code-32B heuristic regret table | `regret_m2_code_32b_stop_at_last.csv` |
| Corrected code label prevalence | `p0_excl_code_label_prevalence.csv` |
| P0-corrected capability/difficulty summary | `capability_difficulty_p0.csv` |

`m2_report.md` replaces a superseded internal milestone report and contains no
current numerical claims.

The capability/difficulty CSV is descriptive. It was rebuilt with
`Mbpp_793` excluded, but reconstruction of its model-specific difficulty tiers
requires an original task-pool cache that is not redistributed. Tier cells are
unbalanced; the 117-task paired comparison is the more stable summary.

## Figures

Released paper figures are under `figure/`. They were produced from the result
families above. To avoid overwriting them, direct regenerated figures to a new
directory as shown in `REPRODUCIBILITY.md`.

## Superseded files

- Smoke-test outputs and files containing `smoke`.
- Pre-OOF files named `mc_*`, `mc2_*`, `ovis_*`, `oracle_visible*`, or
  `qa_ovis*` if added from an internal workspace.
- Legacy cross-loop files without `oof_b1000` in the filename.
- Pre-P0 code rows in mixed OOF files.

When a CSV and `AUTHORITATIVE_RESULTS.md` appear to disagree, first check
whether the CSV row is pre-P0 or whether a verdict stored in an old CSV used an
incorrect loop-family margin. Report the discrepancy rather than silently
choosing a value.
