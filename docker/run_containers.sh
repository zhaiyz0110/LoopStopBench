#!/usr/bin/env bash
# Optional docker-run helpers. Docker Compose is the recommended public entry point.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS="${LOOPSTOP_WORKSPACE:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
HF_CACHE="${LOOPSTOP_HF_CACHE:-$HOME/.cache/huggingface}"
HARNESS_IMAGE="${LOOPSTOP_HARNESS_IMAGE:-loopstop-harness:latest}"
VLLM_IMAGE="${LOOPSTOP_VLLM_IMAGE:-vllm/vllm-openai:v0.9.1}"

# Override these variables for the available GPU topology and model mirrors.
GPU_8B="${LOOPSTOP_GPU_8B:-0}"
GPU_32B="${LOOPSTOP_GPU_32B:-1}"
GPU_LARGE="${LOOPSTOP_GPU_LARGE:-0,1}"
GPU_JUDGE="${LOOPSTOP_GPU_JUDGE:-0,1}"
GPU_JUDGE_SOLO="${LOOPSTOP_GPU_JUDGE_SOLO:-0,1,2,3}"
LARGE_TP="${LOOPSTOP_LARGE_TP:-2}"
JUDGE_TP="${LOOPSTOP_JUDGE_TP:-2}"
JUDGE_SOLO_TP="${LOOPSTOP_JUDGE_SOLO_TP:-4}"

MODEL_8B="${LOOPSTOP_MODEL_8B:-Qwen/Qwen3-8B}"
MODEL_32B="${LOOPSTOP_MODEL_32B:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
MODEL_72B="${LOOPSTOP_MODEL_72B:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
MODEL_JUDGE="${LOOPSTOP_MODEL_JUDGE:-casperhansen/llama-3.3-70b-instruct-awq}"
MODEL_JUDGE_QWEN="${LOOPSTOP_MODEL_JUDGE_QWEN:-Qwen/Qwen2.5-72B-Instruct}"
MODEL_JUDGE_LLAMA="${LOOPSTOP_MODEL_JUDGE_LLAMA:-meta-llama/Llama-3.3-70B-Instruct}"

mkdir -p "$HF_CACHE"

ensure_network() {
  docker network inspect loopstop >/dev/null 2>&1 || docker network create loopstop >/dev/null
}

start_collect() {
  ensure_network
  docker run -d --name vllm-8b --network loopstop --restart unless-stopped \
    --gpus all -e CUDA_VISIBLE_DEVICES="$GPU_8B" --ipc=host \
    -e HF_HOME=/hf -v "$HF_CACHE":/hf -p "${LOOPSTOP_PORT_8B:-8001}":8000 \
    "$VLLM_IMAGE" \
    --model "$MODEL_8B" --dtype bfloat16 --tensor-parallel-size 1 \
    --max-model-len 16384 --gpu-memory-utilization 0.92 --port 8000

  docker run -d --name vllm-32b --network loopstop --restart unless-stopped \
    --gpus all -e CUDA_VISIBLE_DEVICES="$GPU_32B" --ipc=host \
    -e HF_HOME=/hf -v "$HF_CACHE":/hf -p "${LOOPSTOP_PORT_32B:-8002}":8000 \
    "$VLLM_IMAGE" \
    --model "$MODEL_32B" --quantization awq_marlin --dtype float16 \
    --tensor-parallel-size 1 --max-model-len 16384 \
    --gpu-memory-utilization 0.92 --port 8000
}

start_large() {
  ensure_network
  docker run -d --name vllm-72b --network loopstop --restart unless-stopped \
    --gpus all -e CUDA_VISIBLE_DEVICES="$GPU_LARGE" --ipc=host \
    -e HF_HOME=/hf -v "$HF_CACHE":/hf -p "${LOOPSTOP_PORT_72B:-8003}":8000 \
    "$VLLM_IMAGE" \
    --model "$MODEL_72B" --quantization awq_marlin --dtype float16 \
    --tensor-parallel-size "$LARGE_TP" --max-model-len 16384 \
    --gpu-memory-utilization 0.92 --port 8000
}

start_judge() {
  ensure_network
  docker run -d --name vllm-judge --network loopstop --restart unless-stopped \
    --gpus all -e CUDA_VISIBLE_DEVICES="$GPU_JUDGE" --ipc=host \
    -e HF_HOME=/hf -v "$HF_CACHE":/hf -p "${LOOPSTOP_PORT_JUDGE:-8004}":8000 \
    "$VLLM_IMAGE" \
    --model "$MODEL_JUDGE" --quantization awq_marlin --dtype float16 \
    --tensor-parallel-size "$JUDGE_TP" --max-model-len 16384 \
    --gpu-memory-utilization 0.92 --port 8000
}

start_judge_solo() {
  local which="${1:?usage: start_judge_solo qwen|llama}"
  local name model port
  case "$which" in
    qwen)
      name=vllm-judge-qwen
      model="$MODEL_JUDGE_QWEN"
      port="${LOOPSTOP_PORT_JUDGE_SOLO:-8005}"
      ;;
    llama)
      name=vllm-judge-llama
      model="$MODEL_JUDGE_LLAMA"
      port="${LOOPSTOP_PORT_JUDGE_SOLO:-8005}"
      ;;
    *)
      echo "usage: start_judge_solo qwen|llama" >&2
      return 2
      ;;
  esac
  ensure_network
  docker run -d --name "$name" --network loopstop --restart unless-stopped \
    --gpus all -e CUDA_VISIBLE_DEVICES="$GPU_JUDGE_SOLO" --ipc=host \
    -e HF_HOME=/hf -v "$HF_CACHE":/hf -p "$port":8000 \
    "$VLLM_IMAGE" --model "$model" --dtype bfloat16 \
    --tensor-parallel-size "$JUDGE_SOLO_TP" --max-model-len 16384 \
    --gpu-memory-utilization 0.92 --port 8000
}

stop_collect() { docker rm -f vllm-8b vllm-32b 2>/dev/null || true; }
stop_large() { docker rm -f vllm-72b 2>/dev/null || true; }
stop_judge() { docker rm -f vllm-judge 2>/dev/null || true; }
stop_judge_solo() { docker rm -f vllm-judge-qwen vllm-judge-llama 2>/dev/null || true; }

# Networked development/collection harness. Generated-code execution remains
# disabled because this container receives API credentials and mounts the repo.
harness() {
  ensure_network
  docker run --rm -it --network loopstop \
    -v "$WS":/workspace -w /workspace -e PYTHONPATH=/workspace/src \
    -e LOOPSTOP_ALLOW_CODE_EXECUTION=0 \
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    -e ANTHROPIC_FRONTIER_MODEL="${ANTHROPIC_FRONTIER_MODEL:-}" \
    -e ANTHROPIC_JUDGE_MODEL="${ANTHROPIC_JUDGE_MODEL:-}" \
    --security-opt no-new-privileges --pids-limit 512 --memory 16g \
    "$HARNESS_IMAGE" "$@"
}

harness_bg() {
  local name="${1:?usage: harness_bg NAME COMMAND...}"
  shift
  ensure_network
  docker run -d --name "$name" --network loopstop \
    -v "$WS":/workspace -w /workspace -e PYTHONPATH=/workspace/src \
    -e LOOPSTOP_ALLOW_CODE_EXECUTION=0 \
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    -e ANTHROPIC_FRONTIER_MODEL="${ANTHROPIC_FRONTIER_MODEL:-}" \
    -e ANTHROPIC_JUDGE_MODEL="${ANTHROPIC_JUDGE_MODEL:-}" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    --security-opt no-new-privileges --pids-limit 512 --memory 16g \
    "$HARNESS_IMAGE" "$@"
}

# Restricted execution boundary for commands that evaluate generated code.
# It has no network, credentials, or host bind mounts. Inputs must be staged by
# an external trusted controller; automatic cross-container dispatch is not
# implemented by this helper.
harness_exec() {
  docker run --rm -it --network none --read-only \
    --tmpfs /tmp:rw,nosuid,noexec,size=512m \
    --tmpfs /work:rw,nosuid,noexec,size=1g -w /work \
    -e LOOPSTOP_ALLOW_CODE_EXECUTION=1 \
    --security-opt no-new-privileges --cap-drop ALL \
    --pids-limit 128 --memory 4g --cpus 2 \
    "$HARNESS_IMAGE" "$@"
}

usage() {
  echo "usage: $0 {start_collect|start_large|start_judge|start_judge_solo|stop_collect|stop_large|stop_judge|stop_judge_solo|harness|harness_bg|harness_exec} [args...]" >&2
}

if (($# == 0)); then
  usage
  exit 2
fi
"$@"

