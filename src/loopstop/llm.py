"""LLM clients for OpenAI-compatible servers and the native Anthropic API."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import requests
import yaml


@dataclass
class ChatResult:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    mean_logprob: Optional[float]


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 600.0,
        supports_logprobs: bool = True,
        provider: str = "openai_compatible",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.supports_logprobs = supports_logprobs
        self.provider = provider

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        seed: Optional[int] = None,
        retries: int = 3,
    ) -> ChatResult:
        if self.provider == "anthropic":
            return self._chat_anthropic(messages, temperature, max_tokens, retries)
        if self.provider != "openai_compatible":
            raise ValueError(f"unsupported LLM provider: {self.provider!r}")

        return self._chat_openai(messages, temperature, max_tokens, seed, retries)

    def _chat_openai(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        retries: int,
    ) -> ChatResult:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if self.supports_logprobs:
            payload["logprobs"] = True

        last_err: Exception | None = None
        for attempt in range(retries):
            t0 = time.monotonic()
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout,
                )
                if resp.status_code == 400 and payload.pop("logprobs", None):
                    continue  # 端点不支持 logprobs, 去掉后重试
                resp.raise_for_status()
                data = resp.json()
                latency = (time.monotonic() - t0) * 1000
                choice = data["choices"][0]
                usage = data.get("usage", {})
                return ChatResult(
                    text=choice["message"]["content"] or "",
                    tokens_in=usage.get("prompt_tokens", 0),
                    tokens_out=usage.get("completion_tokens", 0),
                    latency_ms=latency,
                    mean_logprob=_mean_logprob(choice),
                )
            except (requests.RequestException, KeyError) as e:  # noqa: PERF203
                last_err = e
                time.sleep(min(2**attempt * 2, 30))
        raise RuntimeError(f"chat failed after {retries} retries: {last_err}")

    def _chat_anthropic(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        retries: int,
    ) -> ChatResult:
        if not self.api_key or self.api_key == "EMPTY":
            raise RuntimeError("Anthropic endpoints require ANTHROPIC_API_KEY")

        system_parts: list[str] = []
        conversation: list[dict] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                system_parts.append(str(content))
            elif role in {"user", "assistant"}:
                conversation.append({"role": role, "content": content})
            else:
                raise ValueError(f"unsupported Anthropic message role: {role!r}")

        payload: dict = {
            "model": self.model,
            "messages": conversation,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        last_err: Exception | None = None
        for attempt in range(retries):
            t0 = time.monotonic()
            try:
                resp = requests.post(
                    f"{self.base_url}/messages",
                    json=payload,
                    headers={
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                        "x-api-key": self.api_key,
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                latency = (time.monotonic() - t0) * 1000
                text = "".join(
                    block.get("text", "")
                    for block in data.get("content", [])
                    if block.get("type") == "text"
                )
                usage = data.get("usage", {})
                return ChatResult(
                    text=text,
                    tokens_in=usage.get("input_tokens", 0),
                    tokens_out=usage.get("output_tokens", 0),
                    latency_ms=latency,
                    mean_logprob=None,
                )
            except (requests.RequestException, KeyError, TypeError) as e:  # noqa: PERF203
                last_err = e
                time.sleep(min(2**attempt * 2, 30))
        raise RuntimeError(f"Anthropic chat failed after {retries} retries: {last_err}")


def _mean_logprob(choice: dict) -> Optional[float]:
    lp = choice.get("logprobs")
    if not lp:
        return None
    content = lp.get("content") or []
    vals = [tok.get("logprob") for tok in content if tok.get("logprob") is not None]
    return sum(vals) / len(vals) if vals else None


def load_models_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_env(value: str, field: str) -> str:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        resolved = os.environ.get(env_name)
        if not resolved:
            raise RuntimeError(f"{field} requires environment variable {env_name}")
        return resolved
    return value


def client_from_config(cfg: dict, endpoint_name: str) -> LLMClient:
    """Build a client from an endpoint entry; values may reference ``${ENV_VAR}``."""
    ep = cfg["endpoints"][endpoint_name]
    api_key = _resolve_env(ep.get("api_key", "EMPTY"), "api_key")
    return LLMClient(
        base_url=_resolve_env(ep["base_url"], "base_url"),
        model=_resolve_env(ep["model"], "model"),
        api_key=api_key,
        supports_logprobs=ep.get("supports_logprobs", True),
        provider=ep.get("provider", "openai_compatible"),
    )
