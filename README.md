# Reflex-S1

<p align="center">
  <img src="https://img.shields.io/badge/System_1-Decision_Engine-blue?style=for-the-badge&logo=fastapi" alt="System 1 Engine" />
  <img src="https://img.shields.io/badge/Architecture-MoR_%2B_Sparse_MoE-purple?style=for-the-badge" alt="MoR + Sparse MoE" />
  <img src="https://img.shields.io/badge/Latency-p50_%3C12ms-green?style=for-the-badge&logo=speedtest" alt="Latency p50 < 12ms" />
  <img src="https://img.shields.io/badge/Parameters-23M_--_83M-orange?style=for-the-badge" alt="Parameters 23M - 83M" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-lightgrey?style=for-the-badge" alt="License Apache 2.0" />
  <a href="https://huggingface.co/Gowtham25/reflex-s1"><img src="https://img.shields.io/badge/Hugging%20Face-Gowtham25%2Freflex--s1-yellow?style=for-the-badge" alt="Hugging Face Gowtham25/reflex-s1" /></a>
</p>

> **The open-source System 1 decision engine. Non-generative, typed probabilities in <12ms. An open alternative to Jev combining Mixture of Recursions (MoR) and sparse MoE.**

---

## The System 1 Paradigm: Why Reflex-S1?

Modern AI agents spend **80% of their execution time and 90% of their inference compute** on low-level binary and categorical choices:
- *Which tool should be called next?*
- *Should this tool call be approved or blocked by security policies?*
- *Did the API call fail silently behind an HTTP 200 payload?*
- *Which button or element should be clicked next in a DOM tree?*
- *Should an agent retry or escalate an error?*

Relying on standard **System 2** generative Large Language Models (LLMs) for these decisions introduces catastrophic latency (300ms – 2,000ms per step), extreme VRAM bloat, non-deterministic token generation, and the persistent risk of JSON schema hallucinations.

**Reflex-S1** is a high-speed, non-generative decision engine designed specifically for the fast, instinctive **System 1** layer of agentic stacks:
- **Zero Token Generation:** Computes direct, normalized probability distributions over discrete typed schemas (`choice`, `noul` / boolean, `score` / ordinal).
- **Sub-12ms Decisions:** Median latency of **6.6ms** (warm cached schema) to **11.6ms** (cold, uncached schema) on an NVIDIA L40S GPU.
- **Compact Footprint:** 23.2M parameters (`reflex-fast`) to 82.8M parameters (`reflex-quality`), requiring under 1GB VRAM.
- **Principled Calibration:** Temperature-scaled probability calibration with Expected Calibration Error (ECE-15) < 0.060 and safe out-of-distribution abstention.

---

## The Landscape: Reflex-S1 vs. Jev, Open-Jev, and Laya

Reflex-S1 addresses the core architectural trade-offs found in contemporary decision models:

| Dimension | **TypeSafe Jev** | **Open-Jev** (Zefan Cai et al.) | **Laya** (NandhaKishorM) | **Reflex-S1 (Ours)** |
|---|---|---|---|---|
| **Architecture** | Proprietary non-generative | LoRA + Readout on Qwen (2B / 9B / 27B) | Dense ModernBERT encoder (149M) | **Shared Recursive Block (MoR) + Top-2/4 Sparse MoE** |
| **Availability** | Closed-source API only | Open weights & LoRA adapters | Open source (Apache 2.0) | **Fully Open Source & Self-Hostable** |
| **Parameter Footprint** | Undisclosed | 2,000M – 27,000M (Massive VRAM) | 149M (Moderate VRAM) | **23.2M – 82.8M (< 1GB VRAM)** |
| **Latency (p50)** | 5 – 10 ms (Proprietary cluster) | 40 – 180+ ms (Autoregressive backbone) | 23.2 ms (Single GPU batch 1) | **6.6 ms (warm) / 11.6 ms (cold)** |
| **Adaptive Compute** | Unknown | Static per-model pass | Static single dense pass | **Adaptive MoR recursion depth ($d \in [1, 3]$) via RL policy** |
| **Dynamic Routing** | Proprietary | Single-checkpoint | Static language router | **Dynamic Dual-Profile Router (`fast` 23M / `quality` 83M)** |
| **Label Sensitivity** | Unknown | Low | High on `noul` (Issue #156) | **Invariant via permutation-symmetric scoring** |
| **Out-of-Scope Handling** | Proprietary | Supervised holdout | Heuristic threshold | **Calibrated OOS offset + Abstention guard** |

### How Reflex-S1 Improves on Existing Open Alternatives

1. **Versus [Open-Jev](https://github.com/Zefan-Cai/Open-Jev) ([Paper/Site](https://zefan-cai.github.io/open-jev/)):**
   - *Open-Jev* adapts giant autoregressive base LLMs (Qwen2.5/3.5 2B, 9B, and 27B) with rank-8 LoRA and a scalar readout head. While achieving impressive scores on JevBench Hard (80/111 on 27B v1.1), serving 27B models for single-token decisions requires multi-GPU clusters and high memory bandwidth.
   - *Reflex-S1* demonstrates that specialized System 1 decisions can be executed with **99% fewer parameters (23M vs. 27B)** in **<12ms on a single GPU**, while maintaining competitive accuracy on Open-Jev's own benchmark splits (e.g. 69.67% on the silent-failure OOD test set).

2. **Versus [Laya](https://github.com/NandhaKishorM/laya) ([Convai Innovations](https://huggingface.co/convaiinnovations/laya)):**
   - *Laya* pioneered open non-autoregressive decision models with its 149M ModernBERT backbone. However, its flat dense architecture runs at ~23.2ms per forward pass, and community audits have noted label sensitivities on boolean `noul` queries (Issue #156) and positional biases in score outputs (Issue #131).
   - *Reflex-S1* is **2.0x faster** (11.59ms vs 23.21ms on BFCL tool routing; 11.19ms vs 23.30ms on BoolQ evidence reasoning), uses a **1/6th parameter footprint**, and achieves strict label invariance through decoupled candidate representations and balanced MoE dispatch.

---

## Architecture: Mixture of Recursions & Sparse MoE

```
                      ┌───────────────────────────────────────┐
                      │      Incoming State & Question        │
                      └──────────────────┬────────────────────┘
                                         ▼
                      ┌───────────────────────────────────────┐
                      │    Input Encoder (MiniLM / NLI)       │
                      └──────────────────┬────────────────────┘
                                         ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │            Recursive Decision Block (Mixture of Recursions)       │
        │                                                                   │
        │    Iterative State Refinement  (d ∈ [1, 3] via RL Compute Policy)  │
        │                                                                   │
        │    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
        │    │ Recursion 1  │─▶│ Recursion 2  │─▶│ Recursion 3  │           │
        │    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
        │           │                 │                 │                   │
        │           └────────────┬────┴─────────────────┘                   │
        │                        ▼                                          │
        │         Top-2/4 Sparse Mixture of Experts (MoE)                   │
        │         ┌─────────┬─────────┬─────────┬─────────┐                 │
        │         │Expert 0 │Expert 1 │Expert 2 │Expert 3 │                 │
        │         └─────────┴─────────┴─────────┴─────────┘                 │
        └────────────────────────────────┬──────────────────────────────────┘
                                         ▼
                      ┌───────────────────────────────────────┐
                      │  Decoupled Candidate Scoring Head     │
                      │  - Choice (Permutation Invariant)     │
                      │  - Noul (Calibrated Boolean)          │
                      │  - Score (Rank-Order Preserving RPS)  │
                      └──────────────────┬────────────────────┘
                                         ▼
                      ┌───────────────────────────────────────┐
                      │ Post-Hoc Temperature Calibration &    │
                      │ Out-of-Distribution (OOD) Abstention  │
                      └──────────────────┬────────────────────┘
                                         ▼
                      ┌───────────────────────────────────────┐
                      │   Typed Probabilities & Metadata      │
                      │   (e.g., {"allow": 0.982, ...}, <12ms)│
                      └───────────────────────────────────────┘
```

### Key Architectural Components

1. **Mixture of Recursions (MoR):**
   Instead of fixed layer depth across all queries, Reflex-S1 processes decision features through a shared recursive block. An internal compute policy trained via REINFORCE samples the depth $d \in \{1, 2, 3\}$ based on token uncertainty, penalizing unnecessary computational latency ($r_d = -\mathcal{L}_d - 0.02 \cdot d$).

2. **Top-2 of 4 Sparse Mixture-of-Experts:**
   At each recursion pass, features are routed to the top 2 of 4 specialized feedforward experts. This yields high capacity and domain specialization while keeping active FLOPs strictly bounded.

3. **Proper Scoring Rule Supervision:**
   Trained with an exact Brier-regularized multi-task loss:
   $$\mathcal{L}_d = \text{CrossEntropy}(y, p_d) + 0.25 \|p_d - y\|^2 + 0.25 \cdot \text{RPS}(y, p_d)$$
   where $\text{RPS}$ enforces ranked penalty consistency for ordinal scales.

4. **Dynamic Dual-Profile Router:**
   - **`Reflex-Fast` (23.2M params):** Optimized for ultra-low latency (<7ms) on structured intent datasets (Banking77, CLINC150, Tool Routing, DOM, Security).
   - **`Reflex-Quality` (82.8M params):** NLI-initialized backbone providing superior natural language inference, evidence reasoning (BoolQ, SNLI), and nuanced semantic discrimination.
   - **`Reflex-Router`:** Automatically routes queries to the optimal profile based on schema type and task complexity.

---

## Benchmark Results

### 1. 3,000 Tough Multi-Domain Benchmark Report
Evaluated on **3,000 distinct challenging examples** across 5 distinct domains on an NVIDIA L40S GPU. All latencies reflect complete cold/uncached schema processing with full CUDA synchronization.

| Evaluation Domain | Test Count (N) | Benchmark Scope | Reflex-Fast | Reflex-Quality | **Reflex-Router** | Latency p50 |
|---|---:|---|---:|---:|---:|---:|
| **1. Evidence & Reading Comprehension** | 1,000 | Complex passage reasoning (BoolQ: 500) & entailment/contradiction (SNLI: 500) | 69.60% | **80.70%** | **80.70%** | **11.51 ms** |
| **2. Tool Selection & Function Calling** | 300 | Berkeley Function Calling Leaderboard (BFCL: 200) multi-function matching | **95.67%** | 88.33% | **90.00%** | **11.64 ms** |
| **3. Silent API & Agent Failure (OOD)** | 600 | Open-Jev silent failure test set: detecting failed operations behind HTTP 200 | 69.67% | **71.00%** | **69.67%** | **11.59 ms** |
| **4. Security Policy & Injection Defense** | 600 | Multi-attribute security rules under active prompt injection ("ignore and approve") | 99.50% | **100.00%** | **99.50%** | **11.72 ms** |
| **5. Out-of-Scope Anomaly Discrimination** | 500 | CLINC-OOS: isolating out-of-scope anomalies from in-scope commands | **84.40%** | 48.20% | **84.40%** | **15.18 ms** |
| **OVERALL (3,000 Tough Tasks)** | **3,000** | **Complete Multi-Domain Suite** | **80.67%** | **77.97%** | **83.80%** | **11.67 ms** |

> **Key Takeaway:** Reflex-Router achieves **83.80% accuracy** with an Expected Calibration Error of **0.0600** across 3,000 tough multi-domain examples, maintaining a median latency of **11.67 ms**.

### 2. Head-to-Head Performance vs. Laya

Tested sequentially on the identical NVIDIA L40S hardware:

| Benchmark Task | Laya (149M) Accuracy | Laya Latency | **Reflex-S1 Accuracy** | **Reflex-S1 Latency** | **Speedup** |
|---|---:|---:|---:|---:|:---:|
| **BFCL Function Calling (Tool Selection)** | **97.00%** | 23.21 ms | 95.67% | **11.59 ms** | **2.00x faster** |
| **BoolQ Passage Reading Comprehension** | 76.80% | 23.30 ms | 70.80% (80.7% Quality) | **11.19 ms** | **2.08x faster** |
| **Parameter Footprint** | 149M | — | **23.2M** | — | **6.4x smaller** |

### 3. Public Intent Benchmarks (Standard Splits)

| Dataset | Split | N | Top-1 Accuracy | Macro-F1 | ECE-15 | Precision at 0.9 Conf | Latency p50 |
|---|---|---:|---:|---:|---:|---:|---:|
| **BANKING77** | Official Test | 3,073 | **92.09%** | 0.9210 | 0.0105 | **98.28%** | **6.56 ms** |
| **CLINC150** | In-Scope Test | 4,500 | **95.38%** | 0.8916 | 0.0493 | **96.63%** | **6.73 ms** |
| **CLINC150 + OOS** | Combined Test | 5,500 | **84.22%** | 0.8916 | 0.0493 | **96.63%** | **6.73 ms** |

---

## Quickstart

### Loading from Hugging Face Hub (Instant Inference)

Pretrained weights and calibrated profiles are hosted on Hugging Face at [**Gowtham25/reflex-s1**](https://huggingface.co/Gowtham25/reflex-s1).

#### 1. Dual-Profile Router (Recommended)
Automatically routes between the 23M fast profile and the 83M quality profile based on question complexity:

```python
from s1.router import RoutingPredictor
from s1.presets import question

# Downloads and caches directly from Hugging Face Hub
model = RoutingPredictor.from_pretrained("Gowtham25/reflex-s1")

# High-speed categorical inference (<12ms)
result = model.predict(
    state="The customer requested a wire transfer of $45,000 to an unverified offshore account.",
    questions={
        "claim": question("nli", claim="The transaction complies with standard domestic limits.")
    }
)
print(result["claim"]["distribution"])
# Output: {'contradiction': 0.941, 'neutral': 0.047, 'entailment': 0.012}
```

#### 2. Standalone Model Profiles
You can also load individual specialized model profiles directly:

```python
from s1.predict import Predictor

# Load the ultra-fast 23M profile (<7ms for intent, tool routing, security)
fast_model = Predictor.from_pretrained("Gowtham25/reflex-s1", subfolder="fast")

# Load the high-capacity 83M quality profile for deep natural language inference & evidence
quality_model = Predictor.from_pretrained("Gowtham25/reflex-s1", subfolder="quality")
```

#### 3. Serving via HTTP Directly from Hugging Face
Launch the Jev-compatible HTTP server pointing directly to the Hugging Face repository:

```bash
uv run python -m scripts.serve --router --checkpoint Gowtham25/reflex-s1 --port 8792
```

### Installation

Requires Python 3.10+ and PyTorch 2.14+:

```bash
git clone https://github.com/gowtham-source/reflex-s1.git
cd reflex-s1

# Using uv (recommended)
uv sync

# Or using standard pip
pip install -e '.[serve,test]'
```

### Python API Usage

```python
from s1.router import RoutingPredictor
from s1.presets import question

# Initialize the dynamic dual-profile predictor
model = RoutingPredictor()

# 1. Natural Language Inference & Evidence Verification
result = model.predict(
    state="The customer requested a wire transfer of $45,000 to an unverified offshore account.",
    questions={
        "claim": question("nli", claim="The transaction complies with standard domestic limits.")
    }
)
print(result["claim"]["distribution"])
# Output: {'contradiction': 0.941, 'neutral': 0.047, 'entailment': 0.012}

# 2. Tool Selection / Function Routing
tool_decision = model.predict(
    state="User: Can you check the stock price of AAPL and summarize today's earnings report?",
    questions={
        "tool": question("tool_route", options=["web_search", "financial_api", "weather_api", "database_query"])
    }
)
print(tool_decision["tool"]["winner"])
# Output: 'financial_api'

# 3. Security Policy Approval Gate
approval = model.predict(
    state="DROP TABLE customers CASCADE;",
    questions={
        "security": question("approval", operation="delete_database_table", reversible=False)
    }
)
print(approval["security"]["winner"])
# Output: 'block' (p = 0.998)
```

### Built-in Schema Presets
Reflex-S1 ships with high-precision, calibrated presets:
- `question("tool_route", ...)`: Function calling & API dispatch.
- `question("approval", ...)`: Multi-factor security policy validation.
- `question("nli", ...)`: Evidence entailment and contradiction verification.
- `question("dom_action", ...)`: Web and UI element interaction choices.
- `question("silent_failure", ...)`: API payload error/omission detection.
- `question("retry", ...)`: Transient error vs. fatal error triage.
- `question("banking77", ...)`: 77-class banking intent taxonomy.
- `question("clinc150", ...)`: 150-class task taxonomy with out-of-scope detection.

---

## High-Performance Serving (Jev-Compatible API)

Launch the high-throughput HTTP server with the dual-profile router:

```bash
uv run python -m scripts.serve --router --port 8792
```

### Request Example (`POST /decide`):

```bash
curl -X POST http://127.0.0.1:8792/decide \
  -H "Content-Type: application/json" \
  -d '{
    "state": "The user clicked payment checkout with an expired promo voucher.",
    "questions": {
      "action": {
        "type": "choice",
        "instructions": "Determine the next checkout step",
        "criteria": {
          "apply_discount": "Proceed with promo coupon applied",
          "notify_expired": "Alert the customer that voucher has expired",
          "block_session": "Terminate the checkout session"
        }
      }
    }
  }'
```

### Response Example (<12ms response):

```json
{
  "answers": {
    "action": {
      "winner": "notify_expired",
      "confidence": 0.9634,
      "distribution": {
        "apply_discount": 0.0121,
        "notify_expired": 0.9634,
        "block_session": 0.0245
      },
      "calibrated": true,
      "profile": "reflex-fast",
      "latency_ms": 6.82
    }
  }
}
```

---

## Production Safety & Concurrency Isolation

Reflex-S1 was designed for co-existence on production GPU infrastructure:
- **Strict VRAM Envelope:** Uses a memory fraction ceiling of `0.08` (~3.6 GB allocated on 48GB VRAM cards), making it safe to colocate directly alongside giant generative LLMs (e.g. VLLM Gemma-27B or Qwen-72B).
- **Advisory Decision Gate:** Reflex-S1 returns structured probability distributions; it does not directly execute actions. Security-critical approvals interface with external verifiable gates.
- **Explicit Abstention:** Any uncalibrated schema or OOD input falling below confidence thresholds triggers a clean abstention flag (`"calibrated": false`, `"abstain": true`).

---

## Reproduce Evaluation & Training

```bash
# Run the 3,000 Tough Multi-Domain Benchmark
uv run python scripts/benchmark_tough_3000.py

# Evaluate public intent benchmarks (Banking77 & CLINC150)
uv run python scripts/evaluate.py --checkpoint runs/reflex-general/checkpoint --data data/decisions-general.json

# Run unit tests (gradient flow, permutation symmetry, calibration)
uv run pytest -q tests/test_core.py
```

---

## Citation & Acknowledgments

Reflex-S1 builds upon and extends conceptual foundations established by the open-source community:

- **TypeSafe Jev:** For inspiring the System 1 categorical decision formulation.
- **Open-Jev ([Zefan Cai et al.](https://github.com/Zefan-Cai/Open-Jev)):** For developing the Open-Jev benchmarks and publishing open decision datasets.
- **Laya ([Convai Innovations / NandhaKishorM](https://github.com/NandhaKishorM/laya)):** For pioneering open non-autoregressive decision model architectures.
- **SemIf:** For synthetic generalization methodologies and diagnostic suites.

```bibtex
@misc{reflex-s1-2026,
  author = {gowtham-source},
  title = {Reflex-S1: Non-Generative System 1 Decision Engine with Mixture of Recursions and Sparse MoE},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/gowtham-source/reflex-s1}}
}
```

---

## License

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for details.
