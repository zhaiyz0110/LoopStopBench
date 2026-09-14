# LoopStopBench Authoritative Results

**Frozen on:** 2026-09-12 (P0 code-sample correction)  
**Primary setting:** `lambda=0.005`, `stop_at_last`, generation-only token cost  
**Intervals:** task-clustered BCa, `B=1000`. Brackets report the two one-sided 95% endpoints (equivalently a 90% two-sided interval).  
**Materiality:** code/QA `Delta=0.02`; writing `Delta=0.05`.

Use this file for manuscript numbers. Files named `mc_*`, `mc2_*`, `ovis_*`, `oracle_visible*`, `qa_ovis*`, or containing `smoke` either predate the OOF baseline correction or are diagnostic runs.

**P0 sample correction:** all final code analyses exclude `Mbpp_793`, whose cached hidden-test set is empty. The corrected code samples contain 350 trajectories/117 tasks for code-8B and 351 trajectories/117 tasks for code-32B. Some mixed historical CSV files retain pre-P0 code rows. `RESULT_MANIFEST.md` identifies their replacements.

## 1. Decision Rules

- **Equivalent:** upper endpoint `< Delta`.
- **Falsified:** lower endpoint `> Delta`.
- **Inconclusive:** interval crosses `Delta`.
- The formal target for **code** is total regret. The formal target for **writing/QA** is recoverable policy-suboptimality.
- `raw recoverable` may be negative as a probe. Reported recoverable is floored at zero because the visible policy class contains the OOF heuristic baseline.
- `--force-recoverable` for code is supplementary. A code total-regret equivalence result is the stronger formal result.
- `--compare-score-fields` analyses are post-hoc paired robustness analyses; their `+/-Delta` equivalence statements are explicitly exploratory.
- A truth-fed bidirectional GRU is a **positive control**, not a member of the output-side visible policy class.

## 2. Primary Regret Decomposition

Sources: code rows from `p0_excl_code_l005_b1000.csv` and `p0_excl_code_l005_b1000_bootstrap.csv`; writing/QA rows from `oof_b1000_l005.csv` and `oof_b1000_l005_bootstrap.csv`.

| Cell | n/tasks | OOF best heuristic (U) | U(oracle) | U(o-vis used) | Information gap | Recoverable | Total regret | Formal target endpoints | Delta | Verdict |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| code-8B | 350/117 | S2 (0.9177) | 0.9275 | 0.9177 | 0.0099 | 0.0000 (raw -0.0015) | 0.0099 | total `[0.0043, 0.0262]` | 0.02 | **Inconclusive** |
| code-32B | 351/117 | S2 (0.9401) | 0.9430 | 0.9401 | 0.0029 | 0.0000 (raw -0.0000) | 0.0029 | total `[0.0011, 0.0173]` | 0.02 | **Equivalent** |
| writing-small | 200/100 | OOF[f0:S6,f1:S1-n3] (0.5704) | 0.7454 | 0.5816 | 0.1639 | 0.0111 | 0.1750 | rec. `[--, 0.0314]` | 0.05 | **Equivalent** |
| writing-mid | 200/100 | OOF[f0:S1-n8,f1:S1-n5] (0.6456) | 0.7400 | 0.6481 | 0.0919 | 0.0025 | 0.0944 | rec. `[--, 0.0220]` | 0.05 | **Equivalent** |
| QA | 200/200 | S2 (0.3035) | 0.3709 | 0.3035 | 0.0675 | 0.0000 (raw -0.0019) | 0.0675 | rec. `[--, 0.0172]` | 0.02 | **Equivalent** |

`--` means the lower endpoint was not emitted by the original full run; it is unnecessary for an equivalence verdict because the upper endpoint is already below `Delta`.

**Code supplementary probes:** at the primary objective, the heuristic-floored recoverable estimate is 0 for both code cells, with stored endpoints `[0.0000, 0.0000]`. The unfloored diagnostics are -0.0015 for code-8B and -0.0000 for code-32B. These boundary values do not imply that the population increment is exactly zero; the formal code target remains total regret.

## 3. Lambda Sensitivity

Sources: code rows from `p0_excl_code_l000_b1000*`, `p0_excl_code_l005_b1000*`, and `p0_excl_code_l020_b1000*`; writing/QA rows from `oof_b1000_l000/l005/l020*.csv` plus `oof_tri_l000/l005/l020_bootstrap.csv` where needed.

| Lambda | Cell | Formal target | Point | Endpoints | Delta | Verdict |
|---:|---|---|---:|---|---:|---|
| 0 | code-8B | total regret | 0.0073 | `[0.0019, 0.0788]` | 0.02 | **Inconclusive** |
| 0 | code-32B | total regret | 0.0050 | `[0.0011, 0.0279]` | 0.02 | **Inconclusive** |
| 0 | writing-small | recoverable | 0.0222 | `[0.0000, 0.0664]` | 0.05 | **Inconclusive** |
| 0 | writing-mid | recoverable | 0.0068 | `[--, 0.0260]` | 0.05 | **Equivalent** |
| 0 | QA | recoverable | 0.0000 (raw -0.0062) | `[--, 0.0034]` | 0.02 | **Equivalent** |
| 0.005 | code-8B | total regret | 0.0099 | `[0.0043, 0.0262]` | 0.02 | **Inconclusive** |
| 0.005 | code-32B | total regret | 0.0029 | `[0.0011, 0.0173]` | 0.02 | **Equivalent** |
| 0.005 | writing-small | recoverable | 0.0111 | `[--, 0.0314]` | 0.05 | **Equivalent** |
| 0.005 | writing-mid | recoverable | 0.0025 | `[--, 0.0220]` | 0.05 | **Equivalent** |
| 0.005 | QA | recoverable | 0.0000 (raw -0.0019) | `[--, 0.0172]` | 0.02 | **Equivalent** |
| 0.02 | code-8B | total regret | 0.0200 | `[0.0101, 0.0376]` | 0.02 | **Inconclusive** |
| 0.02 | code-32B | total regret | 0.0036 | `[0.0015, 0.0083]` | 0.02 | **Equivalent** |
| 0.02 | writing-small | recoverable | 0.0000 (raw -0.0007) | `[--, 0.0066]` | 0.05 | **Equivalent** |
| 0.02 | writing-mid | recoverable | 0.0000 (raw -0.0024) | `[--, 0.0045]` | 0.05 | **Equivalent** |
| 0.02 | QA | recoverable | 0.0282 | `[0.0018, 0.0666]` | 0.02 | **Inconclusive** |

No lambda-sensitivity cell is formally falsified.

## 4. Model-Class Ladder

Sources: code rows from `p0_excl_mc3_m2_code_8b.csv` and `p0_excl_mc3_m2_code_32b.csv`; writing/QA rows from `mc3_*.csv`. All rows use utility labels at `lambda=0.005`.

### 4.1 Signal-Only Models

| Cell | Model | Recoverable | Endpoints | Delta | Verdict |
|---|---|---:|---|---:|---|
| code-8B | GBT | -0.0065 | `[-0.0170, -0.0023]` | 0.02 | Equivalent |
| code-8B | GRU-h8 | -0.0007 | `[-0.0031, 0.0013]` | 0.02 | Equivalent |
| code-8B | GRU-h16 | -0.0004 | `[-0.0028, 0.0018]` | 0.02 | Equivalent |
| code-8B | GRU-h32 | -0.0037 | `[-0.0128, 0.0016]` | 0.02 | Equivalent |
| code-8B | GRU-h64 | -0.0055 | `[-0.0131, -0.0019]` | 0.02 | Equivalent |
| code-8B | GRU-h128 | 0.0004 | `[-0.0021, 0.0030]` | 0.02 | Equivalent |
| code-8B | LSTM-h64 | -0.0047 | `[-0.0133, 0.0001]` | 0.02 | Equivalent |
| code-32B | GBT | -0.0159 | `[-0.0300, -0.0065]` | 0.02 | Equivalent |
| code-32B | GRU-h8 | -0.0063 | `[-0.0175, -0.0028]` | 0.02 | Equivalent |
| code-32B | GRU-h16 | -0.0038 | `[-0.0064, 0.0002]` | 0.02 | Equivalent |
| code-32B | GRU-h32 | -0.0072 | `[-0.0159, -0.0037]` | 0.02 | Equivalent |
| code-32B | GRU-h64 | -0.0009 | `[-0.0014, 0.0079]` | 0.02 | Equivalent |
| code-32B | GRU-h128 | -0.0133 | `[-0.0455, -0.0038]` | 0.02 | Equivalent |
| code-32B | LSTM-h64 | -0.0050 | `[-0.0146, -0.0014]` | 0.02 | Equivalent |
| writing-small | GBT | 0.0132 | `[-0.0137, 0.0312]` | 0.05 | Equivalent |
| writing-small | GRU-h8 | 0.0145 | `[-0.0146, 0.0329]` | 0.05 | Equivalent |
| writing-small | GRU-h16 | 0.0194 | `[-0.0071, 0.0357]` | 0.05 | Equivalent |
| writing-small | **GRU-h32** | **0.0328** | **`[0.0065, 0.0462]`** | 0.05 | **Significant >0; sub-material** |
| writing-small | **GRU-h64** | **0.0299** | **`[0.0028, 0.0477]`** | 0.05 | **Significant >0; sub-material** |
| writing-small | GRU-h128 | 0.0214 | `[-0.0062, 0.0364]` | 0.05 | Equivalent |
| writing-small | LSTM-h64 | 0.0229 | `[-0.0025, 0.0399]` | 0.05 | Equivalent |
| writing-mid | GBT | 0.0073 | `[-0.0030, 0.0255]` | 0.05 | Equivalent |
| writing-mid | GRU-h8 | 0.0015 | `[-0.0086, 0.0164]` | 0.05 | Equivalent |
| writing-mid | GRU-h16 | -0.0068 | `[-0.0187, 0.0101]` | 0.05 | Equivalent |
| writing-mid | GRU-h32 | 0.0033 | `[-0.0081, 0.0179]` | 0.05 | Equivalent |
| writing-mid | GRU-h64 | 0.0032 | `[-0.0079, 0.0203]` | 0.05 | Equivalent |
| writing-mid | GRU-h128 | 0.0083 | `[-0.0017, 0.0238]` | 0.05 | Equivalent |
| writing-mid | LSTM-h64 | 0.0114 | `[-0.0012, 0.0242]` | 0.05 | Equivalent |
| QA | GBT | -0.0251 | `[-0.0447, -0.0101]` | 0.02 | Equivalent |
| QA | GRU-h8 | -0.0207 | `[-0.0415, 0.0021]` | 0.02 | Equivalent |
| QA | GRU-h16 | -0.0325 | `[-0.0576, -0.0102]` | 0.02 | Equivalent |
| QA | GRU-h32 | -0.0332 | `[-0.0600, -0.0088]` | 0.02 | Equivalent |
| QA | **GRU-h64** | **-0.0047** | **`[-0.0223, 0.0291]`** | 0.02 | **Inconclusive** |
| QA | **GRU-h128** | **-0.0099** | **`[-0.0289, 0.0200]`** | 0.02 | **Inconclusive** |
| QA | LSTM-h64 | -0.0271 | `[-0.0548, -0.0041]` | 0.02 | Equivalent |

### 4.2 Positive Controls

| Cell | Truth-fed model recoverable | Endpoints | Gate (`lower>0`) | Interpretation |
|---|---:|---|---|---|
| code-8B | 0.0011 | `[-0.0025, 0.0056]` | Fail/no resolution | Total decision space is only about 0.01 |
| code-32B | -0.0038 | `[-0.0070, -0.0001]` | Fail/no resolution | Total decision space is only about 0.003 |
| writing-small | 0.1613 | `[0.1375, 0.1766]` | **Pass** | Pipeline can recover a large available gain |
| writing-mid | 0.0868 | `[0.0750, 0.1003]` | **Pass** | Pipeline can recover a large available gain |
| QA | 0.0191 | `[0.0058, 0.0570]` | **Pass** | Positive but only partial recovery |

**Model-class conclusion:** no signal-only model is proven to produce material gain. However, QA GRU-h64/h128 are inconclusive, so do not claim that every model-class upper endpoint is below the margin.

## 5. Cross-Loop VOI

Sources: `xloop_voi_oof_b1000_l000/l005/l020.csv`.

| Lambda | Held-out family | AUROC | OOF best heuristic | Oracle-tau recoverable [endpoints] | Verdict | Transfer-tau recoverable [endpoints] | Verdict |
|---:|---|---:|---|---|---|---|---|
| 0 | code | 0.8061 | S2 | -0.0102 `[-0.0194, 0.0002]` | Equivalent | -0.0175 `[-0.0313, -0.0018]` | Equivalent |
| 0 | writing | 0.6234 | S2 | -0.0010 `[-0.0230, 0.0118]` | Equivalent | -0.0010 `[-0.0232, 0.0120]` | Equivalent |
| 0 | QA | 0.8160 | S2 | -0.0071 `[-0.0358, 0.0065]` | Equivalent | -0.0328 `[-0.0649, -0.0035]` | Equivalent |
| 0.005 | code | 0.8061 | S2 | -0.0331 `[-0.0453, -0.0255]` | Equivalent | -0.0352 `[-0.0465, -0.0254]` | Equivalent |
| 0.005 | writing | 0.6234 | OOF[f0:S6,f1:S1-n3] | -0.0097 `[-0.0238, -0.0006]` | Equivalent | -0.0327 `[-0.0467, -0.0206]` | Equivalent |
| 0.005 | QA | 0.8160 | S2 | 0.0002 `[-0.0247, 0.0263]` | **Inconclusive** | -0.0456 `[-0.0812, -0.0152]` | **Equivalent** |
| 0.02 | code | 0.8061 | S2 | -0.0394 `[-0.0569, -0.0296]` | Equivalent | -0.1856 `[-0.2122, -0.1593]` | Equivalent |
| 0.02 | writing | 0.6234 | S8 | -0.0378 `[-0.0523, -0.0298]` | Equivalent | -0.2017 `[-0.2227, -0.1900]` | Equivalent |
| 0.02 | QA | 0.8160 | OOF[f0:S1-n3,f1:S2] | 0.0361 `[0.0022, 0.0648]` | **Inconclusive** | -0.0795 `[-0.1270, -0.0338]` | **Equivalent** |

All nine deployable transfer-tau tests are equivalent. The optimistic, non-deployable QA oracle-tau probe is inconclusive at `lambda=0.005` and `0.02`.

## 6. Signal Manipulations

### 6.1 Critic Capacity and Query Protocol

Sources: `oof_signal_critic*_b1000*.csv`, `oof_signal_pairwise32_b1000*.csv`.

| Visible signal | Corr(quality) | MI (bits) | Recoverable | Endpoints | Delta | Verdict |
|---|---:|---:|---:|---|---:|---|
| 8B critic, absolute | -0.026 | 0.0775 | 0.0241 | `[0.0102, 0.0548]` | 0.05 | **Inconclusive** |
| 32B critic, absolute | 0.057 | 0.0819 | 0.0317 | `[0.0266, 0.0559]` | 0.05 | **Inconclusive** |
| 72B critic, absolute | 0.043 | 0.0848 | 0.0209 | `[0.0000, 0.0452]` | 0.05 | **Equivalent** |
| 32B critic, pairwise | 0.172 | 0.0879 | 0.0318 | `[0.0077, 0.0535]` | 0.05 | **Inconclusive** |

Recoverable is non-monotonic across 8B/32B/72B (`0.0241 -> 0.0317 -> 0.0209`) even as MI rises slightly. No critic condition is proven to produce material gain.

The released critic bootstrap CSVs use the writing-family margin `Delta=0.05`; the verdicts above match their corrected metadata. The numerical estimates were not changed by that metadata correction.

### 6.2 Paired Query-Protocol Contrast

Source: `oof_pair_protocol_b1000.csv`.

| Contrast | Raw rec. A | Raw rec. B | Paired change | Endpoints | Exploratory margin | Verdict |
|---|---:|---:|---:|---|---:|---|
| pairwise-32B - absolute-32B | 0.0317 | 0.0318 | 0.0001 | `[-0.0350, 0.0110]` | +/-0.05 | **Exploratory equivalence** |

Quality correlation triples (`0.057 -> 0.172`) while paired recoverable changes by only 0.0001. Bootstrap mean change is 0.0157, reflecting non-smooth policy/baseline reselection; report the point, BCa endpoints, and exploratory status together.

### 6.3 Repeated-Query Contrasts

Sources: `oof_repeat_ws_pair_b1000.csv`, `oof_repeat_wm_pair_b1000.csv`.

| Cell | M=1 raw rec. | M=8 raw rec. | Paired change | Endpoints | Exploratory margin | Verdict |
|---|---:|---:|---:|---|---:|---|
| writing-small | 0.0111 | 0.0246 | 0.0135 | `[-0.0005, 0.0467]` | +/-0.05 | **Exploratory equivalence** |
| writing-mid | 0.0025 | -0.0017 | -0.0042 | `[-0.0210, 0.0053]` | +/-0.05 | **Exploratory equivalence** |

Eight-query averaging does not produce a significant or material paired improvement in either writing cell.

## 7. Cost and Return-Rule Sensitivity

### 7.1 Author + Critic Token Cost

Sources: `oof_full_cost_b1000.csv`, `oof_full_cost_b1000_bootstrap.csv`.

| Cell | OOF best heuristic (U) | U(oracle) | Recoverable | Endpoints | Total regret | Delta | Verdict |
|---|---|---:|---:|---|---:|---:|---|
| writing-small | OOF[f0:S8,f1:S1-n3] (0.5558) | 0.7043 | 0.0155 | `[0.0000, 0.0510]` | 0.1486 | 0.05 | **Inconclusive** |
| writing-mid | OOF[f0:S1-n3,f1:S1-n5] (0.6118) | 0.6970 | 0.0129 | `[0.0057, 0.0317]` | 0.0852 | 0.05 | **Equivalent** |

Primary results use generation-only cost. When critic tokens are added, writing-small becomes narrowly inconclusive (upper endpoint 0.0510); writing-mid remains equivalent.

### 7.2 `stop_and_select`

Sources: code rows from `p0_excl_code_stopselect_b1000.csv` and `p0_excl_code_stopselect_b1000_bootstrap.csv`; writing/QA rows from `oof_stop_and_select_b1000.csv` and `oof_stop_and_select_b1000_bootstrap.csv`.

| Cell | OOF best heuristic (U) | U(oracle) | Formal target | Point | Endpoints | Delta | Verdict |
|---|---|---:|---|---:|---|---:|---|
| code-8B | S2 (0.9177) | 0.9210 | total regret | 0.0034 | `[0.0016, 0.0073]` | 0.02 | **Equivalent** |
| code-32B | S2 (0.9401) | 0.9404 | total regret | 0.0003 | `[0.0000, 0.0007]` | 0.02 | **Equivalent** |
| writing-small | OOF[f0:S1-n1,f1:S1-n3] (0.4938) | 0.5329 | recoverable | 0.0112 | `[0.0046, 0.0233]` | 0.05 | **Equivalent** |
| writing-mid | OOF[f0:S1-n5,f1:S1-n8] (0.5394) | 0.5988 | recoverable | 0.0221 | `[0.0160, 0.0447]` | 0.05 | **Equivalent** |
| QA | S2 (0.3153) | 0.3405 | recoverable | 0.0024 | `[0.0000, 0.0313]` | 0.02 | **Inconclusive** |

The absolute best-heuristic utility drops sharply under visible-score selection in writing (`0.5704 -> 0.4938` small; `0.6456 -> 0.5394` mid), consistent with critic-score/hidden-truth misalignment. QA improves slightly (`0.3035 -> 0.3153`) but its recoverable equivalence test becomes inconclusive.

### 7.3 Capability and Difficulty (Descriptive)

Source: `capability_difficulty_p0.csv`, recomputed after excluding `Mbpp_793`.
Regret uses the fixed S1-n5 reference at `lambda=0.005`.

| Difficulty tier | code-32B mean regret (n) | code-8B mean regret (n) | 8B - 32B |
|---|---:|---:|---:|
| Easy | 0.0096 (24) | 0.1652 (117) | +0.1556 |
| Medium | 0.0071 (24) | 0.1040 (123) | +0.0969 |
| Hard | 0.0073 (3) | 0.1252 (110) | +0.1179 |

Across 117 same-task pairs, the mean 8B-minus-32B regret difference is
`+0.0992` (paired t analogue `3.999`). In the descriptive regression, the
standardized model-size and difficulty coefficients are `+0.0755` and
`-0.0216`, respectively.

This analysis supports an association between generator capability and
stopping regret, but it is not a randomized causal comparison. The
model-specific difficulty tiers are highly unbalanced, especially the
code-32B hard cell (`n=3` trajectories); the same-task contrast is therefore
more informative than individual tier means. The public release contains the
derived CSV but not the original task-pool cache needed to reconstruct its
difficulty assignments.

## 8. Claims supported by the frozen results

### Supported claims

1. Under the primary `lambda=0.005`, generation-only, `stop_at_last` setting on the corrected sample, writing-small, writing-mid, QA, and code-32B pass their formal equivalence tests; code-8B total regret is inconclusive, while its supplementary recoverable is equivalent.
2. No signal-only model in the tested ladder is proven to produce material gain. Writing-small GRU-h32/h64 recover statistically significant but sub-material gains.
3. Every deployable cross-loop transfer-tau policy is equivalent across all three lambda values, despite held-out AUROC up to 0.816.
4. Critic capacity does not monotonically raise recoverable utility. No critic or query-protocol condition is proven material.
5. Paired query-protocol and repeated-query changes are exploratory-equivalent within +/-0.05.
6. Most estimated stopping regret remains in the information-insufficiency segment in the primary setting.
7. In the corrected descriptive capability analysis, code-8B has higher S1-n5 regret than code-32B on the same 117 tasks; tier-specific comparisons require caution because their sample sizes are unbalanced.

### Unsupported formulations

1. Do **not** write that every upper endpoint in all families/model classes/ablations is below its margin.
2. Do **not** describe QA GRU-h64/h128 as equivalent; both are inconclusive.
3. Do **not** describe QA oracle-tau as equivalent at `lambda=0.005` or `0.02`; both are inconclusive.
4. Do **not** describe 8B/32B critic-scale conditions as equivalent at the writing margin; both are inconclusive.
5. Do **not** claim full-cost robustness for writing-small; its upper endpoint is 0.0510.
6. Do **not** call any inconclusive result a falsification. No formal target in the frozen results is falsified.
7. Do **not** describe code-32B as equivalent at `lambda=0`; its corrected total-regret interval is inconclusive.

## 9. Superseded and Excluded Files

Use these source families for the rewrite:

- Primary/lambda code rows: `p0_excl_code_l000/l005/l020_b1000*.csv`; writing/QA rows: `oof_b1000_l*.csv`, supplemented by `oof_tri_l*_bootstrap.csv` where needed.
- Model class: corrected code rows in `p0_excl_mc3_m2_code_*.csv`; writing/QA rows in `mc3_*.csv`.
- Cross-loop: `xloop_voi_oof_b1000_l*.csv`.
- Signal: `oof_signal_*_b1000*.csv`, plus paired `oof_pair_protocol_b1000.csv`.
- Repeated query: `oof_repeat_*_pair_b1000.csv`.
- Cost/return: `oof_full_cost_b1000*.csv`; corrected code return-rule rows in `p0_excl_code_stopselect_b1000*.csv`, with writing/QA rows in `oof_stop_and_select_b1000*.csv`.
- Descriptive code-policy table: `regret_m2_code_*_stop_at_last.csv`.
- Code label prevalence: `p0_excl_code_label_prevalence.csv`.
- Descriptive capability/difficulty analysis: `capability_difficulty_p0.csv`.

Do not use for final paper numbers:

- `mc_*`, `mc2_*`, `mc_smoke*` (old/in-sample heuristic baseline or smoke).
- `ovis_*`, `oracle_visible*`, `qa_ovis*` (pre-OOF historical results).
- `oof_*smoke*.csv` (execution checks only).
- `xloop_voi.csv`, `xloop_voi_lam*.csv` (pre-OOF baseline).
- Pre-P0 code rows in `oof_b1000_l*.csv`, `oof_tri_l*_bootstrap.csv`, `mc3_m2_code_*.csv`, and `oof_stop_and_select_b1000*.csv` (superseded by the corrected P0 files above).
