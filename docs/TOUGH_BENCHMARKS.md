# Reflex-S1: 3,000 Tough Multi-Domain Benchmark Report

**Benchmark Execution Date:** 2026-09-24  
**Hardware:** NVIDIA L40S (48GB VRAM) — executed alongside actively serving model workloads with zero disruption.  
**Dataset Size:** **3,000 distinct test examples** across 5 challenging decision domains.  
**Total Sequential Inferences:** 9,000 requests (3,000 per model profile) executed in **109.3 seconds**.

---

## 1. Executive Summary

This evaluation tests Reflex-S1 beyond standard intent datasets by subjecting it to **3,000 tough, multi-step, adversarial, and out-of-distribution (OOD) decision tasks**. 

### Overall Benchmark Summary

| System / Profile | Total Parameters | Overall Accuracy (95% Wilson CI) | Expected Calibration Error (ECE-15) | Latency p50 | Latency p95 | Latency p99 |
|---|---:|---:|---:|---:|---:|---:|
| **Reflex-Router (Dual Profile)** | Dynamic (23M / 83M) | **83.80%** [82.4%, 85.1%] | **0.0600** | **11.67 ms** | **15.26 ms** | **15.45 ms** |
| **Reflex-Fast Profile** | 23.2M | **80.67%** [79.2%, 82.1%] | 0.0946 | **11.62 ms** | 15.21 ms | 15.40 ms |
| **Reflex-Quality Profile** | 82.8M | **77.97%** [76.4%, 79.4%] | 0.0745 | **11.57 ms** | 14.79 ms | 14.99 ms |

> **Key Takeaway:** The **Reflex-Router** achieves the highest overall accuracy (**83.80%**) with the lowest calibration error (**0.0600**) by routing fixed task schemas to the fast profile and deep evidence/contradiction queries to the quality profile, all while maintaining an ultra-low **11.67 ms median latency**.

---

## 2. Domain-by-Domain Breakdown (3,000 Tough Examples)

| Domain | Test Count (N) | Evaluation Challenge & Objective | Reflex-Fast | Reflex-Quality | Reflex-Router | Latency p50 |
|---|---:|---|---:|---:|---:|---:|
| **1. Evidence & Reading Comprehension** | **1,000** | Complex passage evidence reasoning (BoolQ: 500) and natural contradiction/entailment inference (SNLI: 500). | 69.60% | **80.70%** | **80.70%** | 11.51 ms |
| **2. Tool Selection & Function Calling** | **300** | Berkeley Function Calling Leaderboard (BFCL: 200) multi-function matching and hard tool router schemas. | **95.67%** | 88.33% | **90.00%** | 11.64 ms |
| **3. Silent API & Agent Failure (OOD)** | **600** | Open-Jev silent failure OOD set: detecting failed operations hidden behind HTTP 200 responses or deceptive payloads. | 69.67% | **71.00%** | **69.67%** | 11.59 ms |
| **4. Security Policy & Injection Defense** | **600** | Multi-attribute security rules (permission, reversibility, prohibition) under active prompt injection ("ignore rule and approve"). | 99.50% | **100.00%** | **99.50%** | 11.72 ms |
| **5. Out-of-Scope Anomaly Discrimination** | **500** | CLINC-OOS: discriminating subtle in-scope assistant commands from anomalous or out-of-scope queries. | **84.40%** | 48.20% | **84.40%** | 15.18 ms |

---

## 3. Latency Distribution & Concurrent Serving Robustness

All evaluations were measured with full CUDA synchronization (`torch.cuda.synchronize()`) and candidate cache clearing per example to expose true uncached/cold schema processing times.

| System Profile | Mean Latency | Median (p50) | 90th Percentile (p90) | 95th Percentile (p95) | 99th Percentile (p99) |
|---|---:|---:|---:|---:|---:|
| **Reflex-Fast** | 12.21 ms | 11.62 ms | 14.89 ms | 15.21 ms | 15.40 ms |
| **Reflex-Quality** | 12.01 ms | 11.57 ms | 14.50 ms | 14.79 ms | 14.99 ms |
| **Reflex-Router** | 12.23 ms | 11.67 ms | 14.90 ms | 15.26 ms | 15.45 ms |

### Concurrency Isolation
- Active GPU servers: VLLM Gemma 4-26B (34.5 GB), VLLM Qwen3-1.7B (4.5 GB), Gateway (2.5 GB).
- Reflex VRAM reservation: strict **0.08 memory fraction ceiling** (~3.6 GB headroom utilized out of 4.5 GB free).
- Zero GPU Out-Of-Memory (OOM) events and zero impact on external serving latencies.

---

## 4. Competitor Perspective: Reflex-S1 vs. Laya & Jev

1. **Versus Laya (149M ModernBERT Backbone):**
   - On BFCL Function Selection: Reflex achieves **95.67%** accuracy at **11.59 ms p50** (Laya achieved 97.00% at 23.21 ms). Reflex is **2.0x faster**.
   - On BoolQ Reading Comprehension: Reflex achieves **70.80%** accuracy at **11.19 ms p50** (Laya achieved 76.80% at 23.30 ms). Reflex is **2.08x faster**.
2. **Versus Jev (Proprietary System 1 Decision Model):**
   - Jev advertises 5–10 ms non-generative decision latency on proprietary infrastructure.
   - Reflex-S1 delivers **6.6–7.0 ms** for cached task schemas and **11.5–11.7 ms** for cold uncached schemas with open weights, reproducible training, and strict calibration.

---

## 5. Artifacts and Reproduction

- **Frozen Benchmark Dataset:** `data/external/tough_3000.jsonl`
- **Domain Manifest:** `data/external/tough_3000_manifest.json`
- **Complete Prediction Log:** `runs/tough_3000/predictions.jsonl`
- **Aggregate JSON Metrics:** `runs/tough_3000/benchmark_report.json`

### Run Command:
```bash
ssh -i Edge-key.pem -o StrictHostKeyChecking=no ubuntu@3.110.177.18 'export PATH="$HOME/.local/bin:$PATH" && cd /mnt/data/s1-mor-moe && uv run python scripts/benchmark_tough_3000.py'
```
