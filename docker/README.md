# Container setup

This directory provides a reference deployment for the model servers and the
CPU analysis harness. It contains no host-specific paths. GPU IDs, model IDs,
ports, image names, and the Hugging Face cache location can be overridden with
the `LOOPSTOP_*` environment variables documented in
`docker/run_containers.sh`.

## Services

| Service | Purpose | Default profile |
|---|---|---|
| `vllm-8b` | 8B generator | `collect` |
| `vllm-32b` | 32B generator or critic | `collect` |
| `vllm-72b` | 72B signal-scale experiment | `large` |
| `vllm-judge` | local writing judge | `judge` |
| `harness` | networked collection and analysis | none |
| `code-exec` | restricted generated-code evaluation | `exec` |

The default GPU topology assumes two CUDA devices. Override the relevant
`LOOPSTOP_GPU_*` and tensor-parallel variables before starting services on a
different host. Model mirrors must be updated consistently in the Docker
environment and `configs/models*.yaml`.

## Build and start

```bash
cd docker
docker compose build harness code-exec
docker compose --profile collect up -d
docker compose logs -f vllm-8b
docker compose run --rm harness python scripts/smoke_test.py
```

For native Anthropic endpoints, define the credentials and exact model IDs
outside the repository:

```bash
export ANTHROPIC_API_KEY=...
export ANTHROPIC_FRONTIER_MODEL=...
export ANTHROPIC_JUDGE_MODEL=...
```

These variables are passed only to the networked `harness` service. They are
not passed to `code-exec`.

## Generated-code execution boundary

The code-repair loop executes generated Python. A subprocess timeout is not a
security sandbox, so execution is disabled unless
`LOOPSTOP_ALLOW_CODE_EXECUTION=1` is explicitly present.

The regular `harness` mounts the repository and may receive API credentials;
it therefore sets `LOOPSTOP_ALLOW_CODE_EXECUTION=0`. Do not override that
setting in the networked container.

The `code-exec` service is the reference restricted environment. It has:

- no network;
- no API credentials;
- no host bind mounts;
- a read-only root filesystem;
- ephemeral `/tmp` and `/work` filesystems;
- dropped Linux capabilities and process/memory limits.

Run an explicitly staged evaluation command with:

```bash
docker compose --profile exec run --rm code-exec COMMAND...
```

The public helper does not automatically transfer generator outputs from the
networked harness into `code-exec`. A trusted external controller must stage
inputs and retrieve outputs without exposing repository files or credentials.
Deployments evaluating genuinely untrusted code should add an independently
reviewed sandbox or VM boundary appropriate to their threat model.

## Reproducibility note

The vLLM image is pinned to `v0.9.1` to describe the archived experimental
environment. Hardware placement and model storage are host-configurable and
need not match the original collection host.
