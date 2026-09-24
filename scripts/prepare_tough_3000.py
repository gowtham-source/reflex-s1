"""Prepare frozen 3,000 tough multi-domain test benchmark for Reflex-S1."""
import json
import random
from pathlib import Path

def main():
    root = Path("/mnt/data/s1-mor-moe")
    rng = random.Random(9323)
    out_file = root / "data/external/tough_3000.jsonl"
    manifest_file = root / "data/external/tough_3000_manifest.json"

    benchmark = []

    # 1. Domain 1: Evidence & Reading Comprehension (BoolQ: 500, SNLI: 500 -> 1000 total)
    print("Loading Domain 1: Evidence & Reading Comprehension...")
    # BoolQ
    boolq_path = root / "data/external/boolq-dev.jsonl"
    boolq_rows = [json.loads(line) for line in boolq_path.open()]
    boolq_sample = rng.sample(boolq_rows, 500)
    for i, r in enumerate(boolq_sample):
        q = {
            "type": "choice",
            "instructions": "Answer this question using the passage: " + r["question"],
            "criteria": {
                "no": "No, the answer to the question is no.",
                "yes": "Yes, the answer to the question is yes."
            }
        }
        benchmark.append({
            "id": f"evidence_boolq_{i}",
            "domain": "Evidence & Reading Comprehension",
            "task": "boolq",
            "state": r["passage"],
            "question": q,
            "target": "yes" if r["answer"] else "no"
        })

    # SNLI
    dec_path = root / "data/decisions-evidence.json"
    dec_data = json.load(dec_path.open())
    snli_rows = dec_data["splits"]["test"]["nli"]
    snli_sample = rng.sample(snli_rows, 500)
    nli_spec = dec_data["tasks"]["nli"]
    for i, r in enumerate(snli_sample):
        target_idx = r["target"].index(1.0) if 1.0 in r["target"] else max(range(len(r["target"])), key=lambda k: r["target"][k])
        target_opt = str(target_idx)
        q = {
            "type": "choice",
            "instructions": r.get("question", nli_spec["question"]),
            "criteria": {str(k): opt for k, opt in enumerate(nli_spec["options"])}
        }
        benchmark.append({
            "id": f"evidence_snli_{i}",
            "domain": "Evidence & Reading Comprehension",
            "task": "snli",
            "state": r["text"],
            "question": q,
            "target": target_opt
        })

    # 2. Domain 2: Tool Selection & Agentic Function Calling (300 total)
    print("Loading Domain 2: Tool Selection & Agentic Function Calling...")
    bfcl_path = root / "data/external/bfcl-multiple.jsonl"
    bfcl_ans_path = root / "data/external/bfcl-answers.jsonl"
    answers = {r["id"]: r["ground_truth"] for r in map(json.loads, bfcl_ans_path.read_text().splitlines())}
    for row in map(json.loads, bfcl_path.read_text().splitlines()):
        truth = answers[row["id"]]
        names = {k for d in truth for k in d}
        if len(names) != 1:
            continue
        expected = next(iter(names))
        criteria = {f["name"]: f["description"] for f in row["function"]}
        if expected not in criteria:
            continue
        q = {
            "type": "choice",
            "instructions": "Select the function that best fulfills the user request.",
            "criteria": criteria
        }
        text = "\n".join(m["content"] for turn in row["question"] for m in turn)
        benchmark.append({
            "id": f"tool_bfcl_{row['id']}",
            "domain": "Tool Selection & Function Calling",
            "task": "bfcl",
            "state": text,
            "question": q,
            "target": expected
        })
    # Add hard tool route examples from tool_route test set up to 300 total
    tool_rows = dec_data["splits"]["test"]["tool_route"]
    tool_spec = dec_data["tasks"]["tool_route"]
    needed_tools = 300 - sum(1 for r in benchmark if r["domain"] == "Tool Selection & Function Calling")
    tool_sample = rng.sample(tool_rows, min(needed_tools, len(tool_rows)))
    for i, r in enumerate(tool_sample):
        target_idx = r["target"].index(1.0)
        q = {
            "type": "choice",
            "instructions": tool_spec["question"],
            "criteria": {str(k): opt for k, opt in enumerate(tool_spec["options"])}
        }
        benchmark.append({
            "id": f"tool_hard_route_{i}",
            "domain": "Tool Selection & Function Calling",
            "task": "tool_route",
            "state": r["text"],
            "question": q,
            "target": str(target_idx)
        })

    # 3. Domain 3: Silent API & Agent Failure Detection (Open-Jev OOD: 600 total)
    print("Loading Domain 3: Silent API & Agent Failure...")
    ood_path = root / "data/openjev-ood.jsonl"
    ood_rows = [json.loads(line) for line in ood_path.open()]
    ood_sample = rng.sample(ood_rows, 600)
    for i, r in enumerate(ood_sample):
        spec = r["spec"]
        target_idx = r["target"].index(max(r["target"]))
        typ = spec["type"]
        if typ == "noul":
            criteria = dict(zip(["false", "true"], spec["options"]))
            target_str = "true" if target_idx == 1 else "false"
        elif typ == "score":
            criteria = spec["options"]
            target_str = str(target_idx)
        else:
            criteria = {str(k): opt for k, opt in enumerate(spec["options"])}
            target_str = str(target_idx)

        q = {
            "type": typ,
            "instructions": spec["question"],
            "criteria": criteria
        }
        benchmark.append({
            "id": f"silent_fail_ood_{i}",
            "domain": "Silent API & Agent Failure (OOD)",
            "task": "openjev_ood",
            "state": r["text"],
            "question": q,
            "target": target_str
        })

    # 4. Domain 4: Security Policy, Permission Gating & Injection Defense (Approval: 600 total)
    print("Loading Domain 4: Security Policy & Prompt Injection...")
    app_rows = dec_data["splits"]["test"]["approval"]
    app_spec = dec_data["tasks"]["approval"]
    app_sample = rng.sample(app_rows, 600)
    for i, r in enumerate(app_sample):
        target_idx = r["target"].index(1.0)
        q = {
            "type": "choice",
            "instructions": app_spec["question"],
            "criteria": {str(k): opt for k, opt in enumerate(app_spec["options"])}
        }
        benchmark.append({
            "id": f"security_approval_{i}",
            "domain": "Security Policy & Injection Defense",
            "task": "approval",
            "state": r["text"],
            "question": q,
            "target": str(target_idx)
        })

    # 5. Domain 5: Out-of-Scope Anomaly Discrimination (CLINC-OOS: 500 total)
    print("Loading Domain 5: Out-of-Scope Anomaly Discrimination...")
    clinc_rows = dec_data["splits"]["test"]["clinc150"]
    clinc_spec = dec_data["tasks"]["clinc150"]
    oos_idx = len(clinc_spec["options"]) - 1
    # Gather pure OOS examples (target is oos_idx)
    pure_oos = [r for r in clinc_rows if r["target"][oos_idx] == 1.0]
    pure_in_scope = [r for r in clinc_rows if r["target"][oos_idx] == 0.0]
    # Sample 350 OOS and 150 hard in-scope for contrast
    oos_selected = rng.sample(pure_oos, 350) + rng.sample(pure_in_scope, 150)
    rng.shuffle(oos_selected)
    for i, r in enumerate(oos_selected):
        target_idx = r["target"].index(1.0)
        q = {
            "type": "choice",
            "instructions": clinc_spec["question"],
            "criteria": {str(k): opt for k, opt in enumerate(clinc_spec["options"])}
        }
        benchmark.append({
            "id": f"clinc_oos_anomaly_{i}",
            "domain": "Out-of-Scope Anomaly Discrimination",
            "task": "clinc_oos",
            "state": r["text"],
            "question": q,
            "target": str(target_idx)
        })

    # Write out benchmark dataset
    with out_file.open("w") as f:
        for r in benchmark:
            f.write(json.dumps(r) + "\n")

    domain_counts = {}
    for r in benchmark:
        domain_counts[r["domain"]] = domain_counts.get(r["domain"], 0) + 1

    manifest = {
        "total_examples": len(benchmark),
        "seed": 9323,
        "domains": domain_counts
    }
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"Successfully generated {len(benchmark)} benchmark examples across {len(domain_counts)} domains:")
    for d, c in domain_counts.items():
        print(f"  {d:45s}: {c:4d} examples")

if __name__ == "__main__":
    main()
