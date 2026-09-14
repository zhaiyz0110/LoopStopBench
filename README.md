# LoopStopBench Workspace

对应研究计划 `../plan_1.md`（v1.0）+ 评审修订 `../review_plan_1.md`。
论文工作题目：*To Loop or Not to Loop: Measuring, Benchmarking, and Learning Stopping Decisions in LLM Agent Loops*。

## 运行环境

- 服务器：2 × A800 80G（无原生 FP8 → 32B/72B 走 AWQ INT4），Driver 575.57.08 / CUDA 12.9。
- 一切实验在 Docker 容器中运行，容器定义见 [docker/](docker/)。
- 本地模型经 vLLM 以 OpenAI 兼容接口暴露；harness 容器只通过 HTTP 访问模型，与 GPU 解耦。

## 目录结构（对应 plan_1 §13）

```
workspace/
├── configs/            # 全部实验由 yaml 驱动（§13）
│   ├── models.yaml     #   模型矩阵（review §2.2）：8B/32B-AWQ/72B-AWQ + 裁判 + API
│   ├── code_repair.yaml / writer_critic.yaml / react_qa.yaml   # 三个循环（§5.1–5.3）
│   └── replay.yaml     #   停止策略基准 S1–S8 + λ 档位（§7）
├── docker/             # vLLM serving + harness 容器（compose profiles 管理 GPU 占用）
├── src/loopstop/
│   ├── schema.py       # §5.5 JSONL schema；VisibleHistory 接口隔离隐藏真值
│   ├── llm.py          # OpenAI 兼容客户端（vLLM / API 统一），带 logprob 采集
│   ├── loops/          # §5：三个 harness；base.py 强制跑满 T 轮全程落盘
│   ├── policies/       # §7.1：S1–S8 统一接口 π(h_t)→{continue,stop} + VOI 策略
│   ├── replay/         # §4.2 + §7：离线回放、效用 U、oracle、regret、bootstrap CI
│   ├── analysis/       # §4.3 轨迹分型 + §6.1 测量指标
│   └── estimator/      # §8：六组特征、标签构造、按 task_id 划分、GBT/MLP 训练
├── scripts/            # collect / judge_writing / replay_eval / analyze / smoke_test
├── tests/              # 合成轨迹上的单元测试（容器内 pytest 运行）
└── data/               # 轨迹 JSONL（不入 git），schema 说明见 data/README.md
```

## 快速开始（服务器上）

```bash
cd workspace/docker

# 1) 起采集档模型（8B 在 GPU0，32B-AWQ 在 GPU1）
docker compose --profile collect up -d

# 2) 连通性冒烟测试
docker compose run --rm harness python scripts/smoke_test.py

# 3) 单元测试
docker compose run --rm harness pytest -q

# 4) Pilot（M1-W3：每循环 30 任务，跑满轮数）
docker compose run --rm harness python scripts/collect.py \
    --loop code_repair --pair code_small --seeds 0 1 2 --limit 30 \
    --out data/trajectories/pilot_code_small.jsonl

# 5) 分型与测量指标（M1-W3 pilot 报告的输入）
docker compose run --rm harness python scripts/analyze.py \
    --traj data/trajectories/pilot_code_small.jsonl

# 6) 停止策略离线回放（同一批轨迹评所有策略，§5.5 的 O(1) 设计）
docker compose run --rm harness python scripts/replay_eval.py \
    --traj data/trajectories/pilot_code_small.jsonl --out results/
```

72B 抽样验证 / 裁判团阶段需独占双卡，与采集档互斥：

```bash
docker compose --profile collect down
docker compose --profile large up -d    # 72B-AWQ TP=2
docker compose --profile judge up -d    # Llama-70B-AWQ TP=2（写作真值裁判）
```

## 与里程碑的对应（plan_1 §10）

| 里程碑 | 本仓库需完成的事 |
|---|---|
| M1-W1 | 填 `loops/code_repair.py` 的两处 TODO（EvalPlus 任务适配、真值测试执行），schema 已定稿 |
| M1-W2 | 填 `loops/react_qa.py` 搜索后端、`loops/writer_critic.py` 已可跑（裁判见 judge_writing.py） |
| M1-W3 | `collect.py` 跑三循环 pilot；`analyze.py` 出初步分型统计 |
| M1-W4 | go/no-go：`analyze.py` 的 final≠best 占比 ≥10%（§10.3） |
| M2 | 全量采集（compose collect 档），冻结 `data/trajectories/` 为 v1 |
| M3 | `replay_eval.py` 出 regret 表与 Pareto 数据；测量部分挂 arXiv（评审：默认执行） |
| M4 | `estimator/train.py` + `policies/voi.py`；leave-one-loop-type-out 泛化 |

## 设计不变量（改代码前必读）

1. **采集期永不早停**：`loops/base.py` 强制跑满 T 轮；停止策略只在离线回放中作用（§5.5）。
2. **隐藏真值接口隔离**：策略只能拿到 `VisibleHistory`，类型层面不含 `hidden_truth`（§4.1 R 不可见）。
3. **同一批轨迹评所有策略**：策略评测一律走 `replay/`，不重新采集。
4. **一切超参进 yaml**：策略调参只动 `configs/replay.yaml`，在验证集（20%，按 task_id 划分）上调。
