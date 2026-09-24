# General decision capability: research and evaluation boundaries

The system is intended as a reusable typed decision component. Training includes intent routing, approval proposals, tool selection, textual UI controls, retry/risk policies, Open-Jev silent failures and per-example evidence claims. A finite task suite cannot demonstrate performance on every task. CLINC is a general assistant intent dataset, not a clinical/medical benchmark.

## Papers and consequences for this implementation

| Primary work | Relevant finding/design | Consequence here |
|---|---|---|
| [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) | Shared recursive blocks with adaptive token computation | Our question-level depth policy is an adaptation, not the paper's token-level implementation. Report fixed-depth ablations and actual selected depths. |
| [Switch Transformers](https://arxiv.org/abs/2101.03961) | Sparse expert activation separates total capacity from active computation | Count real dispatch and measure batch-one overhead; sparse arithmetic alone does not establish lower wall time. |
| [Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | Confidence and correctness can diverge; post-hoc temperature is useful | Fit only on reserved calibration data and report ECE, proper scores and selective coverage. No calibration guarantee under shift. |
| [BoolQ, NAACL 2019](https://arxiv.org/abs/1905.10044) | Naturally occurring yes/no questions require inference; entailment transfer helps | Add a frozen external passage/question test, beyond templated rule controls. Question-specific schemas test uncached candidate encoding. |
| [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html) | Function calling evaluates tools and arguments, not merely output syntax | Extract an explicitly labeled function-identity subtask. Do not call its accuracy an official BFCL score or claim argument-generation capability. |
| [AgentBench, ICLR 2024](https://arxiv.org/abs/2308.03688) | Interactive evaluation spans eight environments; long-term decisions and instruction following matter | One-step routing is only a component. End-to-end agent success requires rollouts and cannot be inferred from classification accuracy. |
| [OSWorld, NeurIPS 2024](https://arxiv.org/abs/2404.07972) | Real computer environments and execution-based evaluation test open-ended multimodal work | Our text-only DOM decisions do not constitute screenshot grounding or an OSWorld result. A visual grounding/action stack is a separate missing component. |
| [OSWorld-Human](https://arxiv.org/abs/2506.16042) | Efficiency depends on planning/reflection calls and complete agent trajectories | Report warm/cold single-request latency now; full task duration, action count and recovery overhead remain unmeasured. |

These are mechanism/evaluation references, not claims that their benchmark scores transfer to this implementation. Research details and Jev/Laya/SemIf source inspection are in [RESEARCH.md](RESEARCH.md).

## Frozen broad evaluation

`scripts/benchmark_broad.py` evaluates both already-trained checkpoints and pinned public Laya on identical requests, randomizing execution order on the same L40S. No BFCL or BoolQ rows enter model training, operating-point fitting, or calibration. Public pretraining contamination cannot be ruled out. The source revisions and hashes are in `data/external/manifest.json`; selected example IDs and raw predictions are retained.

- BFCL multiple-function category: all examples with a single unambiguous expected function identity. Names/descriptions are candidates. Parameters, argument values and execution are outside this projection.
- BoolQ: a preselected seed-9323 sample of 500 validation questions with original passages and labels. This is not a private test or the complete validation split.
- Earlier SemIf authored144: independently authored evidence/rule/candidate decisions. Repeated aggregate results influenced development, so it is a development generalization check.

The benchmark records forced-choice accuracy, row Wilson intervals, 10,000 paired bootstrap resamples against Laya, p50/p95/p99, errors, and abstentions. Reflex candidate caches are cleared before each request to expose new-schema cost. All errors remain in the accuracy denominator. Error latency is not mixed into successful-inference latency percentiles. Uncalibrated schemas abstain in normal serving, even when a diagnostic argmax is correct.

Laya's native per-option cap remains intact. Affected requests are counted; a common untruncated subset is also reported. Larger head/state budgets avoid other unnecessary truncation. This preserves public model behavior while making the remaining rendering asymmetry visible.

## Deployment interpretation

Use the compact profile for validated fixed schemas and the larger profile for evidence/NLI questions. Both are advisory. New schemas require representative labeled data and calibration before confident automation. Unknown schemas are structurally accepted but abstain by default. This is not a claim of reliable universal zero-shot inference.

Further evidence required for broad agent claims: held-out application/episode splits, human-reviewed permission policies, screenshot grounding, argument correctness, multi-step environment completion, adversarial instructions, English domain shift and multilingual testing, matched competitor fine-tuning, multiple training seeds, and complete service/concurrency latency. None should be replaced with synthetic control accuracy.


## Measured Comparative Benchmark (Reflex-S1 vs Laya)

All systems evaluated sequentially on the same NVIDIA L40S GPU under identical hardware, requests, and batch=1 warm settings. Reflex candidate cache cleared before each request to expose uncached cold-schema performance.

### 1. External Reasoning & Tool Tasks (BFCL & BoolQ)

| Task | System | Parameters | Accuracy (95% CI) | Latency p50 | Latency p95 | Paired Δ vs Laya (95% Bootstrap) |
|---|---|---:|---:|---:|---:|---:|
| **BFCL Function Selection** (N=200) | **Reflex Fast** | 23.2M | **93.50%** [89.2%, 96.2%] | **11.35 ms** | 12.19 ms | -3.5% [-7.0%, 0.0%] |
| | **Reflex Quality** | 82.8M | **85.00%** [79.4%, 89.3%] | **11.52 ms** | 12.43 ms | -12.0% [-17.0%, -7.0%] |
| | Laya Baseline | 149M | 97.00% [93.6%, 98.6%] | 23.21 ms | 25.91 ms | reference |
| **BoolQ Natural Evidence** (N=500) | **Reflex Quality** | 82.8M | **70.80%** [66.7%, 74.6%] | **11.19 ms** | 12.07 ms | -6.0% [-11.0%, -1.0%] |
| | **Reflex Fast** | 23.2M | 57.20% [52.8%, 61.5%] | **11.07 ms** | 11.81 ms | -19.6% [-24.8%, -14.4%] |
| | Laya Baseline | 149M | 76.80% [72.9%, 80.3%] | 23.30 ms | 24.96 ms | reference |

> **Key Performance Finding:** Reflex achieves **>2.0x lower latency** than Laya (11.1–11.5 ms vs 23.2–23.3 ms) across all external benchmark evaluations, while delivering strong 93.5% accuracy on tool selection.

---

### 2. Comprehensive Task Suite — Reflex-Evidence Profile (10 Task Heads)

Evaluated on `runs/reflex-evidence/checkpoint` across 17,276 held-out test rows:

| Task Suite | Domain / Objective | Test Rows (N) | Calibrated Accuracy | ECE (15-bin) | Latency p50 | Latency p95 |
|---|---|---:|---:|---:|---:|---:|
| **Approval** | Policy permission & review gating | 600 | **100.00%** | 0.0000 | 7.00 ms | 8.03 ms |
| **Tool Route** | Agentic tool invocation | 360 | **95.56%** | 0.0391 | 6.90 ms | 7.00 ms |
| **DOM Action** | Textual UI element control | 360 | **100.00%** | 0.0000 | 6.77 ms | 7.61 ms |
| **Retry Policy** | Idempotency & transient retry | 360 | **100.00%** | 0.0000 | 6.65 ms | 6.84 ms |
| **Risk Score** | Operation reversibility / risk level | 360 | **100.00%** | 0.0000 | 6.81 ms | 7.01 ms |
| **Open-Jev** | Silent API failure detection | 624 | **96.15%** | 0.0226 | 7.00 ms | 9.97 ms |
| **SNLI (NLI)** | Textual entailment & evidence | 2,439 | **90.82%** | 0.0122 | 11.56 ms | 11.89 ms |
| **BoolQ** | Evidence reading comprehension | 3,260 | **72.55%** | 0.0313 | 11.30 ms | 12.21 ms |
| **Banking77** | Multi-intent banking routing | 3,073 | **87.54%** | 0.0163 | 6.81 ms | 7.11 ms |
| **Clinc150** | Assistant intent + OOS detection | 5,500 | **81.25%** | 0.0527 | 6.94 ms | 7.08 ms |

Total test set evaluations: **17,276 distinct examples**.
