"""Build the cached task file used by the code-repair loop.

The script loads EvalPlus tasks, evaluates canonical solutions in subprocesses,
and creates per-case assertions. Plus inputs are selected by deterministic hash
up to ``--max-plus`` per task. Floating-point outputs use ``math.isclose``.
Optional difficulty filtering estimates first-round pass rates and retains
tasks within ``--lo`` and ``--hi``.

Example:
  python scripts/build_code_tasks.py --dataset humaneval_plus \\
      --out data/raw/code_tasks_filtered.json
To skip difficulty filtering:
  python scripts/build_code_tasks.py --dataset humaneval_plus --skip-filter \\
      --out data/raw/code_tasks_pilot.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
os.environ["HUMANEVAL_OVERRIDE_PATH"] = str(project_root / "data/raw/HumanEvalPlus.jsonl.gz")
os.environ["MBPP_OVERRIDE_PATH"] = str(project_root / "data/raw/MbppPlus.jsonl.gz")

_COMPUTE_RUNNER = r"""
import json, signal, sys

payload = json.loads(sys.stdin.read())
ns = {}
exec(payload["code"], ns)
candidate = ns[payload["entry_point"]]

def _on_alarm(signum, frame):
    raise TimeoutError()
signal.signal(signal.SIGALRM, _on_alarm)

out = []
for case in payload["inputs"]:
    try:
        signal.alarm(payload.get("per_case_timeout", 5))
        result = candidate(*case)
        out.append({"ok": True, "repr": repr(result), "float": isinstance(result, float)})
    except BaseException as e:
        out.append({"ok": False, "err": repr(e)[:200]})
    finally:
        signal.alarm(0)
print(json.dumps(out))
"""


def compute_expected(code: str, entry_point: str, inputs: list, per_case_timeout: int = 5) -> list:
    """Evaluate the dataset's canonical solution in a timeout-limited subprocess."""
    payload = json.dumps(
        {"code": code, "entry_point": entry_point, "inputs": inputs, "per_case_timeout": per_case_timeout}
    )
    proc = subprocess.run(
        [sys.executable, "-c", _COMPUTE_RUNNER],
        input=payload,
        capture_output=True,
        text=True,
        timeout=per_case_timeout * len(inputs) + 15,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def make_assertion(args: list, expected: dict, atol: float) -> str:
    args_repr = repr(tuple(args))
    if expected["float"] or atol:
        tol = atol or 1e-6
        return (
            f"import math; assert math.isclose(candidate(*{args_repr}), "
            f"{expected['repr']}, rel_tol=1e-6, abs_tol={tol})"
        )
    return f"assert candidate(*{args_repr}) == {expected['repr']}"


def sample_plus(inputs: list, task_id: str, max_plus: int) -> list[tuple[int, list]]:
    """Return deterministically sampled ``(original index, input)`` pairs."""
    indexed = list(enumerate(inputs))
    if len(indexed) <= max_plus:
        return indexed
    keyed = sorted(
        indexed, key=lambda p: hashlib.md5(f"{task_id}:{p[0]}".encode()).hexdigest()
    )
    return sorted(keyed[:max_plus], key=lambda p: p[0])


def build_tasks(dataset: str, max_plus: int) -> list[dict]:
    if dataset == "humaneval_plus":
        from evalplus.data import get_human_eval_plus

        problems = get_human_eval_plus()
    elif dataset == "mbpp_plus":
        from evalplus.data import get_mbpp_plus

        problems = get_mbpp_plus()
    else:
        raise ValueError(dataset)

    tasks, skipped = [], []
    for task_id, p in problems.items():
        code = p["prompt"] + p["canonical_solution"]
        atol = float(p.get("atol") or 0)
        base_inputs = list(p["base_input"])
        plus_pairs = sample_plus(list(p["plus_input"]), task_id, max_plus)
        try:
            base_exp = compute_expected(code, p["entry_point"], base_inputs)
            plus_exp = compute_expected(code, p["entry_point"], [a for _, a in plus_pairs])
        except Exception as e:
            skipped.append((task_id, f"canonical failed: {e}"))
            continue
        if any(not e["ok"] for e in base_exp + plus_exp):
            skipped.append((task_id, "canonical raised on some inputs"))
            continue
        tasks.append(
            {
                "task_id": task_id.replace("/", "_"),
                "payload": {
                    "prompt": p["prompt"],
                    "entry_point": p["entry_point"],
                    "base_tests": [
                        {"id": f"base{i}", "assertion": make_assertion(a, e, atol)}
                        for i, (a, e) in enumerate(zip(base_inputs, base_exp))
                    ],
                    "plus_tests": [
                        {"id": f"plus{orig_i}", "assertion": make_assertion(a, e, atol)}
                        for (orig_i, a), e in zip(plus_pairs, plus_exp)
                    ],
                },
            }
        )
    if skipped:
        print(f"[warn] skipped {len(skipped)} problems: {skipped[:5]}...")
    return tasks


def assign_tier(rate: float, lo: float, hi: float) -> str:
    """Assign an in-range task to an easy, medium, or hard tier."""
    if rate < lo or rate > hi:
        return "out"
    if rate >= 0.60:
        return "easy"
    if rate >= 0.30:
        return "medium"
    return "hard"


def calibrate_difficulty(tasks: list[dict], gens: list[str], args) -> list[dict]:
    """Estimate first-round pass rates and difficulty tiers for each model.

    A task is retained when at least one model's pass rate falls within the
    configured range. Each retained task stores model-specific rates and tiers.
    """
    from loopstop.llm import client_from_config, load_models_config
    from loopstop.loops.code_repair import GEN_TMPL, SYSTEM, CodeRepairLoop, run_test_cases

    cfg = load_models_config(args.models_config)
    clients = {g: client_from_config(cfg, g) for g in gens}
    parse = CodeRepairLoop.parse_output

    kept = []
    for n, t in enumerate(tasks, 1):
        p = t["payload"]
        rates, tiers = {}, {}
        for g, client in clients.items():
            passes = 0
            for k in range(args.k):
                res = client.chat(
                    [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": GEN_TMPL.format(prompt=p["prompt"])},
                    ],
                    temperature=0.7,
                    max_tokens=2048,
                    seed=9000 + k,
                )
                code = parse(None, res.text)
                results = run_test_cases(code, p["base_tests"], p["entry_point"])
                passes += all(ok for _, ok, _ in results)
            rates[g] = passes / args.k
            tiers[g] = assign_tier(rates[g], args.lo, args.hi)
        in_band = any(v != "out" for v in tiers.values())
        rate_str = " ".join(f"{g}={rates[g]:.2f}({tiers[g]})" for g in gens)
        print(f"[{n}/{len(tasks)}] {t['task_id']}  {rate_str}  "
              f"{'KEEP' if in_band else 'drop'}", flush=True)
        if in_band:
            t["first_round_pass_rate"] = rates
            t["difficulty_tier"] = tiers
            kept.append(t)
    return kept


def stratified_sample(tasks: list[dict], ref: str, targets: dict, n_final: int) -> list[dict]:
    """Sample deterministic easy, medium, and hard proportions for a reference model."""
    buckets: dict[str, list] = {"easy": [], "medium": [], "hard": []}
    for t in tasks:
        tier = t.get("difficulty_tier", {}).get(ref, "out")
        if tier in buckets:
            buckets[tier].append(t)
    out = []
    for tier, frac in targets.items():
        want = round(n_final * frac)
        pool = sorted(buckets[tier], key=lambda x: hashlib.md5(x["task_id"].encode()).hexdigest())
        if len(pool) < want:
            print(f"[warn] tier {tier!r}: want {want}, only {len(pool)} available (按 {ref} 分层)")
        out.extend(pool[:want])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["humaneval_plus"],
                    help="One or more datasets: humaneval_plus or mbpp_plus")
    ap.add_argument("--out", default="data/raw/code_tasks_filtered.json")
    ap.add_argument("--max-plus", type=int, default=50, help="每题 plus 测试采样上限")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 题(调试)")
    ap.add_argument("--skip-filter", action="store_true", help="Skip difficulty filtering")
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--gen", default="gen_8b", help="单模型标定端点(向后兼容)")
    ap.add_argument("--gen-multi", nargs="+", default=None,
                    help="Model endpoints for joint difficulty calibration; overrides --gen")
    ap.add_argument("--k", type=int, default=5, help="每题首轮样本数")
    ap.add_argument("--lo", type=float, default=0.05, help="带外下界(<lo 记 hard 之外, drop)")
    ap.add_argument("--hi", type=float, default=0.85, help="带外上界(>hi 太易, drop)")
    # Optional stratified sampling; otherwise retain all in-range tasks.
    ap.add_argument("--stratify-by", default=None, help="按该模型的难度层分层, 如 gen_8b")
    ap.add_argument("--n-final", type=int, default=None, help="分层后最终题数")
    ap.add_argument("--targets", default="easy:0.2,medium:0.5,hard:0.3",
                    help="Target proportions by difficulty tier")
    args = ap.parse_args()

    tasks = []
    for ds in args.dataset:
        part = build_tasks(ds, args.max_plus)
        for t in part:
            t["dataset"] = ds
        tasks.extend(part)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"built {len(tasks)} tasks from {args.dataset}")

    if not args.skip_filter:
        gens = args.gen_multi or [args.gen]
        tasks = calibrate_difficulty(tasks, gens, args)
        print(f"kept {len(tasks)} in-band tasks (models={gens}, band=[{args.lo},{args.hi}])")

        if args.stratify_by and args.n_final:
            targets = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in args.targets.split(",")}
            tasks = stratified_sample(tasks, args.stratify_by, targets, args.n_final)
            from collections import Counter
            dist = Counter(t["difficulty_tier"][args.stratify_by] for t in tasks)
            print(f"stratified by {args.stratify_by}: {len(tasks)} tasks, tiers={dict(dist)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
