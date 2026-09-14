"""Check local model endpoints with a short request and inspect log-probability output."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loopstop.llm import client_from_config, load_models_config

LOCAL_PREFIXES = ("http://vllm-", "http://localhost", "http://127.")


def main() -> None:
    cfg = load_models_config("configs/models.yaml")
    ok = fail = 0
    for name, ep in cfg["endpoints"].items():
        if not ep["base_url"].startswith(LOCAL_PREFIXES):
            print(f"  {name:16s} SKIP (API endpoint, costs money)")
            continue
        try:
            client = client_from_config(cfg, name)
            res = client.chat(
                [{"role": "user", "content": "Reply with the single word: ok"}],
                temperature=0.0,
                max_tokens=8,
                retries=1,
            )
            lp = "logprob✓" if res.mean_logprob is not None else "logprob✗"
            print(f"  {name:16s} OK   {res.text.strip()[:20]!r}  "
                  f"{res.tokens_in}+{res.tokens_out} tok  {res.latency_ms:.0f}ms  {lp}")
            ok += 1
        except Exception as e:
            print(f"  {name:16s} FAIL {e}")
            fail += 1
    print(f"\n{ok} ok, {fail} failed (未启动的 profile 报 FAIL 属正常)")
    sys.exit(1 if ok == 0 else 0)


if __name__ == "__main__":
    main()
