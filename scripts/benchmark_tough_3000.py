"""Run comprehensive 3,000 tough multi-domain benchmark on Reflex-S1."""
import json
import time
import math
import random
from pathlib import Path
import numpy as np
import torch
from s1.predict import Predictor
from s1.router import RoutingPredictor

def wilson(k, n):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, float(c - h)), min(1.0, float(c + h))]

def compute_ece(confidences, corrects, bins=15):
    if not confidences or not corrects:
        return 0.0
    confidences = np.array(confidences)
    corrects = np.array(corrects)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += mask.mean() * abs(confidences[mask].mean() - corrects[mask].mean())
    return float(ece)

def main():
    root = Path("/mnt/data/s1-mor-moe")
    dataset_path = root / "data/external/tough_3000.jsonl"
    out_dir = root / "runs/tough_3000"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Restrict memory to safe headroom (max 8% of total 46GB ~= 3.6GB)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.08, 0)
        torch.set_num_threads(4)

    print("Loading Reflex-S1 profiles...")
    fast_model = Predictor(str(root / "runs/reflex-general/checkpoint"), device="cuda")
    quality_model = Predictor(str(root / "runs/reflex-evidence/checkpoint"), device="cuda")
    router_model = RoutingPredictor(device="cuda")

    models = {
        "Reflex-Fast": fast_model,
        "Reflex-Quality": quality_model,
        "Reflex-Router": router_model
    }

    # Warmup
    print("Warming up models on GPU...")
    dummy_q = {
        "type": "choice",
        "instructions": "Determine the state.",
        "criteria": {"a": "Alpha option", "b": "Beta option"}
    }
    for m in models.values():
        for _ in range(10):
            m.predict("Dummy input sentence for warmup cache initialization.", {"q": dummy_q})

    rows = [json.loads(line) for line in dataset_path.open()]
    total_n = len(rows)
    print(f"Starting benchmark execution across {total_n} tough examples...")

    records = {name: [] for name in models}
    detailed_logs = []

    start_bench = time.time()
    for idx, r in enumerate(rows):
        domain = r["domain"]
        target = str(r["target"])
        q_spec = r["question"]
        state_text = r["state"]

        for name, model in models.items():
            # Clear candidate cache to capture realistic dynamic-schema / cold costs
            if hasattr(model, "cache"):
                model.cache.clear()
            elif hasattr(model, "fast") and hasattr(model.fast, "cache"):
                model.fast.cache.clear()
                model.general.cache.clear()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            error = None
            pred_choice = None
            conf = 0.0
            abstain = False
            probs = {}

            try:
                res = model.predict(state_text, {"q": q_spec})
                ans = res["answers"]["q"]
                if "choice" in ans:
                    pred_choice = str(ans["choice"])
                elif "noul" in ans:
                    pred_choice = "true" if ans["noul"] >= 0.5 else "false"
                elif "score" in ans:
                    pred_choice = str(int(round(ans["score"])))
                else:
                    pred_choice = None
                conf = float(ans.get("confidence", 0.0))
                abstain = bool(ans.get("abstain", False))
                probs = ans.get("probabilities", {})
            except Exception as e:
                error = str(e)

            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            is_correct = (pred_choice == target) if error is None else False

            rec = {
                "id": r["id"],
                "domain": domain,
                "task": r["task"],
                "system": name,
                "target": target,
                "prediction": pred_choice,
                "correct": is_correct,
                "confidence": conf,
                "abstain": abstain,
                "latency_ms": latency_ms,
                "error": error
            }
            records[name].append(rec)
            detailed_logs.append(rec)

        if (idx + 1) % 500 == 0 or (idx + 1) == total_n:
            elapsed = time.time() - start_bench
            print(f"  Processed {idx + 1}/{total_n} examples ({elapsed:.1f}s elapsed)...")

    # Aggregate Analysis
    domains = sorted(list({r["domain"] for r in rows}))
    report = {
        "total_examples": total_n,
        "elapsed_seconds": round(time.time() - start_bench, 2),
        "systems": {}
    }

    for name, recs in records.items():
        all_latencies = [r["latency_ms"] for r in recs if r["error"] is None]
        all_correct = [r["correct"] for r in recs]
        all_conf = [r["confidence"] for r in recs]
        tot_correct = sum(all_correct)
        tot_errors = sum(1 for r in recs if r["error"] is not None)
        tot_abstain = sum(1 for r in recs if r["abstain"])

        sys_report = {
            "overall": {
                "accuracy": tot_correct / total_n,
                "correct": tot_correct,
                "total": total_n,
                "wilson95": wilson(tot_correct, total_n),
                "ece15": compute_ece(all_conf, all_correct),
                "abstentions": tot_abstain,
                "errors": tot_errors,
                "latency_p50_ms": float(np.percentile(all_latencies, 50)) if all_latencies else 0.0,
                "latency_p90_ms": float(np.percentile(all_latencies, 90)) if all_latencies else 0.0,
                "latency_p95_ms": float(np.percentile(all_latencies, 95)) if all_latencies else 0.0,
                "latency_p99_ms": float(np.percentile(all_latencies, 99)) if all_latencies else 0.0,
                "latency_mean_ms": float(np.mean(all_latencies)) if all_latencies else 0.0,
            },
            "by_domain": {}
        }

        for dom in domains:
            dom_recs = [r for r in recs if r["domain"] == dom]
            dom_n = len(dom_recs)
            dom_correct = sum(r["correct"] for r in dom_recs)
            dom_latencies = [r["latency_ms"] for r in dom_recs if r["error"] is None]
            dom_conf = [r["confidence"] for r in dom_recs]
            dom_corr_list = [r["correct"] for r in dom_recs]
            dom_errors = sum(1 for r in dom_recs if r["error"] is not None)

            sys_report["by_domain"][dom] = {
                "accuracy": dom_correct / dom_n,
                "correct": dom_correct,
                "total": dom_n,
                "errors": dom_errors,
                "wilson95": wilson(dom_correct, dom_n),
                "ece15": compute_ece(dom_conf, dom_corr_list),
                "abstentions": sum(1 for r in dom_recs if r["abstain"]),
                "latency_p50_ms": float(np.percentile(dom_latencies, 50)) if dom_latencies else 0.0,
                "latency_p95_ms": float(np.percentile(dom_latencies, 95)) if dom_latencies else 0.0,
            }

        report["systems"][name] = sys_report

    # Save artifacts
    (out_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2))
    with (out_dir / "predictions.jsonl").open("w") as f:
        for l in detailed_logs:
            f.write(json.dumps(l) + "\n")

    print("\n=== BENCHMARK COMPLETE ===")
    print(f"Report saved to: {out_dir / 'benchmark_report.json'}")
    print(f"Predictions saved to: {out_dir / 'predictions.jsonl'}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'System':15s} | {'Overall Acc':12s} | {'ECE':8s} | {'p50 (ms)':10s} | {'p95 (ms)':10s} | {'p99 (ms)':10s}")
    print("-" * 90)
    for name, s in report["systems"].items():
        o = s["overall"]
        print(f"{name:15s} | {o['accuracy']*100:6.2f}%     | {o['ece15']:.4f}   | {o['latency_p50_ms']:6.2f}ms   | {o['latency_p95_ms']:6.2f}ms   | {o['latency_p99_ms']:6.2f}ms")
    print("=" * 90)

    print("\n--- Domain Breakdown ---")
    for dom in domains:
        print(f"\n[Domain: {dom}]")
        for name, s in report["systems"].items():
            d = s["by_domain"][dom]
            err_str = f" (Errors: {d['errors']})" if d['errors'] > 0 else ""
            print(f"  {name:15s}: Acc {d['accuracy']*100:6.2f}%{err_str}  |  p50: {d['latency_p50_ms']:5.2f}ms  |  p95: {d['latency_p95_ms']:5.2f}ms")

if __name__ == "__main__":
    main()
