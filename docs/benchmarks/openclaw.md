# OpenClaw Benchmark

Evaluation of ContextPilot on [OpenClaw](https://openclaw.ai) agent workloads using the [claw-tasks](https://github.com/EfficientContext/ClawTasks) dataset.

## Setup

| | |
|---|---|
| Model | Qwen3-4B-Instruct-2507 |
| Engine | SGLang 0.5.9 |
| GPU | single RTX 5090 |
| Context Length | 131,072 tokens |
| Dataset | 60 tasks, 22 documents (490 KB), ~250 turns |
| Baseline | OpenClaw → SGLang (direct) |
| Treatment | OpenClaw → ContextPilot → SGLang (proxy) |

## Results

```
                                                  Avg          P50          P99
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prompt Tokens
  OpenClaw + SGLang                            45,771       44,570       92,785
  OpenClaw + ContextPilot + SGLang             33,622       32,526       51,581
  Δ                                            -26.5%       -27.0%       -44.4%

Wall Time (s)
  OpenClaw + SGLang                              26.1         25.2         68.8
  OpenClaw + ContextPilot + SGLang               20.8         21.8         50.4
  Δ                                            -20.4%       -13.3%       -26.6%

Completion Tokens
  OpenClaw + SGLang                               765         1004         1024
  OpenClaw + ContextPilot + SGLang                758          981         1024
  Δ                                             -0.9%        -2.3%        +0.0%

Accuracy (substantive output)
  OpenClaw + SGLang                      245/245 (100.0%)
  OpenClaw + ContextPilot + SGLang       245/245 (100.0%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Reproduce

```bash
git clone https://github.com/EfficientContext/ClawTasks.git
cd ClawTasks
python scripts/run_bench.py --gpu 0
python scripts/analyze.py results/results.jsonl
```

## Raw Data

[`openclaw_results.jsonl`](openclaw_results.jsonl) — 490 data points (60 scenarios × 2 arms × ~4 turns).
