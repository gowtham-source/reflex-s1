import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
data=json.loads((root/'data/decisions-general.json').read_text())
(root/'s1/schemas.json').write_text(json.dumps(data['tasks'],indent=2))
