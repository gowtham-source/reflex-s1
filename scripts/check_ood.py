import json
from pathlib import Path

for i, line in enumerate(Path("data/openjev-ood.jsonl").open()):
    r = json.loads(line)
    print(f"Row {i}: task={r.get('task')}, type={r['spec']['type']}, options={r['spec']['options']}")
    if i >= 5:
        break
