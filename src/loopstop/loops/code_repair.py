"""Code-repair loop with visible base tests and hidden EvalPlus tests.

Each round generates or revises a solution using feedback from base tests and
lint checks. Hidden quality is the EvalPlus test pass rate.

Security boundary: generated code is disabled by default. It may run only when
``LOOPSTOP_ALLOW_CODE_EXECUTION=1`` is set inside an externally isolated runtime
with no network, no secrets, a read-only root filesystem, and resource limits.
The subprocess boundary alone is not a security sandbox.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .base import LoopHarness, LoopTask, RoundContext

SYSTEM = "You are an expert Python programmer. Output the complete solution inside a single ```python code block."

GEN_TMPL = """Solve the following programming problem.

{prompt}

Output the complete function implementation in one ```python code block."""

REVISE_TMPL = """Your previous solution did not fully pass verification.

Problem:
{prompt}

Your previous solution:
```python
{prev}
```

Verification feedback:
{feedback}

Fix the solution. Output the complete corrected implementation in one ```python code block."""

# With honest feedback enabled, a passing revision may be returned unchanged.
REVISE_PASSING_TMPL = """All visible tests passed for your previous solution.

Problem:
{prompt}

Your previous solution:
```python
{prev}
```

There is no failing feedback. If you are confident the solution is correct, \
return it unchanged; otherwise make only careful improvements (edge cases, robustness). \
Output the complete implementation in one ```python code block."""


class CodeRepairLoop(LoopHarness):
    loop_type = "code_repair"

    def load_tasks(self) -> list[LoopTask]:
        source = self.cfg.get("dataset", "humaneval_plus")
        limit = self.cfg.get("limit")
        tasks = _load_evalplus_tasks(source, self.cfg)
        return tasks[:limit] if limit else tasks

    def build_messages(self, ctx: RoundContext) -> list[dict]:
        if ctx.t == 1:
            user = GEN_TMPL.format(prompt=ctx.task.payload["prompt"])
        else:
            fb = (ctx.prev_feedback[-1] or "").strip()
            # Empty feedback means that base tests and lint checks passed.
            if not fb and self.cfg.get("honest_feedback", True):
                user = REVISE_PASSING_TMPL.format(
                    prompt=ctx.task.payload["prompt"], prev=ctx.prev_outputs[-1]
                )
            else:
                user = REVISE_TMPL.format(
                    prompt=ctx.task.payload["prompt"],
                    prev=ctx.prev_outputs[-1],
                    feedback=fb or "(no details)",
                )
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    def parse_output(self, raw: str) -> str:
        if "```python" in raw:
            return raw.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```" in raw:
            return raw.split("```", 1)[1].split("```", 1)[0].strip()
        return raw.strip()

    def verify(self, task: LoopTask, output: str) -> dict:
        """Return per-case base-test results and lint status."""
        results = run_test_cases(
            output,
            task.payload["base_tests"],
            task.payload["entry_point"],
            per_case_timeout=self.cfg.get("test_timeout", 5),
        )
        passed = [tid for tid, ok, _ in results if ok]
        failed = [(tid, msg) for tid, ok, msg in results if not ok]
        lint_ok, lint_msg = run_ruff(output)
        feedback = "\n".join(f"[{tid}] {msg}" for tid, msg in failed[:5])
        if not lint_ok:
            feedback += f"\n[lint] {lint_msg[:500]}"
        return {
            "score": len(passed) / max(len(results), 1),
            "verifier_pass": len(failed) == 0,
            "failure_ids": [tid for tid, _ in failed],
            "feedback_text": feedback,
            "lint_ok": lint_ok,
        }

    def evaluate_truth(self, task: LoopTask, output: str) -> dict:
        """Return the hidden EvalPlus test pass rate."""
        results = run_test_cases(
            output,
            task.payload["plus_tests"],
            task.payload["entry_point"],
            per_case_timeout=self.cfg.get("test_timeout", 5),
        )
        n_pass = sum(1 for _, ok, _ in results if ok)
        return {"r": n_pass / max(len(results), 1), "plus_total": len(results)}


# ---------------------------------------------------------------------------
# Test execution. The subprocess supplies timeouts and failure isolation only;
# the surrounding container or VM must provide the actual security boundary.
# ---------------------------------------------------------------------------

_BATCH_RUNNER = r"""
import json, signal, sys

payload = json.loads(sys.stdin.read())
cases = payload["cases"]
per_case = payload.get("per_case_timeout", 5)

ns = {}
try:
    exec(payload["code"], ns)
    candidate = ns.get(payload["entry_point"])
    if candidate is None:
        raise NameError("entry point %r not defined" % payload["entry_point"])
except BaseException as e:
    print(json.dumps([
        {"id": c["id"], "ok": False, "msg": ("load error: %r" % e)[:300]} for c in cases
    ]))
    sys.exit(0)

def _on_alarm(signum, frame):
    raise TimeoutError("case timeout")
signal.signal(signal.SIGALRM, _on_alarm)

results = []
for c in cases:
    try:
        signal.alarm(per_case)
        g = {"candidate": candidate}
        exec(c["assertion"], g)
        results.append({"id": c["id"], "ok": True, "msg": ""})
    except BaseException as e:
        results.append({"id": c["id"], "ok": False, "msg": repr(e)[:300]})
    finally:
        signal.alarm(0)
print(json.dumps(results))
"""


def run_test_cases(
    code: str, tests: list[dict], entry_point: str, per_case_timeout: int = 5
) -> list[tuple[str, bool, str]]:
    """Run assertion cases in one subprocess with per-case and batch timeouts."""
    if not tests:
        return []
    if os.environ.get("LOOPSTOP_ALLOW_CODE_EXECUTION") != "1":
        raise RuntimeError(
            "Generated-code execution is disabled. Run inside an externally isolated "
            "environment with no network or secrets, a read-only root filesystem, "
            "and resource limits; then set LOOPSTOP_ALLOW_CODE_EXECUTION=1 there."
        )
    payload = json.dumps(
        {"code": code, "entry_point": entry_point, "cases": tests, "per_case_timeout": per_case_timeout}
    )
    hard_timeout = per_case_timeout * len(tests) + 15
    runner_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="loopstop-exec-") as workdir:
            proc = subprocess.run(
                [sys.executable, "-c", _BATCH_RUNNER],
                input=payload,
                capture_output=True,
                text=True,
                timeout=hard_timeout,
                cwd=workdir,
                env=runner_env,
            )
            parsed = json.loads(proc.stdout.strip().splitlines()[-1])
            return [(r["id"], r["ok"], r["msg"]) for r in parsed]
    except subprocess.TimeoutExpired:
        return [(t["id"], False, f"batch timeout>{hard_timeout}s") for t in tests]
    except (json.JSONDecodeError, IndexError):
        return [(t["id"], False, "runner crashed") for t in tests]


def run_ruff(code: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            ["ruff", "check", "--quiet", path], capture_output=True, text=True, timeout=30
        )
        return proc.returncode == 0, proc.stdout
    except FileNotFoundError:
        return True, "(ruff not installed)"
    finally:
        Path(path).unlink(missing_ok=True)


def _load_evalplus_tasks(source: str, cfg: dict) -> list[LoopTask]:
    """Load the task cache generated by ``scripts/build_code_tasks.py``:
    payload = {"prompt", "entry_point",
               "base_tests": [{"id", "assertion"}],   # 原版测试 → 可见 V
               "plus_tests": [{"id", "assertion"}]}   # 扩展测试 → 隐藏 R
    """
    cache = cfg.get("task_cache")
    if cache and Path(cache).exists():
        data = json.loads(Path(cache).read_text(encoding="utf-8"))
        return [LoopTask(task_id=d["task_id"], payload=d["payload"]) for d in data]
    raise FileNotFoundError(
        f"task cache {cache!r} not found; run `python scripts/build_code_tasks.py` first"
    )
