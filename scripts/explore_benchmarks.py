import json
from pathlib import Path

print("=== CHECKING EXTERNAL & HARD DATASETS ===")
# Check Open-Jev metadata
meta_path = Path("data/openjev-metadata.json")
if meta_path.exists():
    meta = json.load(meta_path.open())
    test_files = [x["rfilename"] for x in meta.get("siblings", []) if "raw/" in x["rfilename"] and "test.jsonl.gz" in x["rfilename"]]
    ood_files = [x["rfilename"] for x in meta.get("siblings", []) if "raw/" in x["rfilename"] and "ood.jsonl.gz" in x["rfilename"]]
    print("Open-Jev test configs:", len(test_files))
    for f in test_files:
        print("  test:", f)
    print("Open-Jev OOD configs:", len(ood_files))
    for f in ood_files:
        print("  ood :", f)

# Check BFCL
bfcl_path = Path("data/external/bfcl-multiple.jsonl")
if bfcl_path.exists():
    lines = [json.loads(line) for line in bfcl_path.open()]
    print("BFCL Multiple count:", len(lines))

# Check BoolQ Dev
boolq_path = Path("data/external/boolq-dev.jsonl")
if boolq_path.exists():
    lines = [json.loads(line) for line in boolq_path.open()]
    print("BoolQ Dev count:", len(lines))

# Check SemIf
semif_path = Path("baselines/semif-authored144.jsonl")
if semif_path.exists():
    lines = [json.loads(line) for line in semif_path.open()]
    print("SemIf Authored144 count:", len(lines))

# Check SNLI test
dec_ev = Path("data/decisions-evidence.json")
if dec_ev.exists():
    data = json.load(dec_ev.open())
    print("Tasks in decisions-evidence.json:")
    for t, rows in data["splits"]["test"].items():
        print(f"  {t:25s} Test rows: {len(rows)}")
