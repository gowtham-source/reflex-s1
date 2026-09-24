"""Download external-only benchmarks; requires optional pyarrow reader."""
import requests,json,hashlib
from pathlib import Path
r=Path(__file__).resolve().parents[1]/'data/external';r.mkdir(parents=True,exist_ok=True)
prior=json.loads((r/'manifest.json').read_text()) if (r/'manifest.json').exists() else {}
s=requests.Session();rev=s.get('https://api.github.com/repos/ShishirPatil/gorilla/commits/main',timeout=60);rev.raise_for_status();rev=prior.get('bfcl_revision',rev.json()['sha']);base=f'https://raw.githubusercontent.com/ShishirPatil/gorilla/{rev}/berkeley-function-call-leaderboard/bfcl_eval/data/'
br=prior.get('boolq_revision') or s.get('https://huggingface.co/api/datasets/google/boolq',timeout=60).json()['sha']
urls={'bfcl-multiple.jsonl':base+'BFCL_v4_multiple.json','bfcl-answers.jsonl':base+'possible_answer/BFCL_v4_multiple.json','bfcl-license.txt':f'https://raw.githubusercontent.com/ShishirPatil/gorilla/{rev}/LICENSE','boolq-dev.parquet':f'https://huggingface.co/datasets/google/boolq/resolve/{br}/data/validation-00000-of-00001.parquet','boolq-train.parquet':f'https://huggingface.co/datasets/google/boolq/resolve/{br}/data/train-00000-of-00001.parquet','boolq-card.md':f'https://huggingface.co/datasets/google/boolq/resolve/{br}/README.md'}
m={'bfcl_revision':rev,'boolq_revision':br,'files':{}}
for name,url in urls.items():
 a=s.get(url,timeout=120);a.raise_for_status();(r/name).write_bytes(a.content);m['files'][name]={'url':url,'sha256':hashlib.sha256(a.content).hexdigest(),'bytes':len(a.content)};print(name,len(a.content),flush=True)
(r/'manifest.json').write_text(json.dumps(m,indent=2))

import pyarrow.parquet as pq
for split in ['train','dev']:
 rows=pq.read_table(r/f'boolq-{split}.parquet').to_pylist()
 p=r/f'boolq-{split}.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
 m['files'][p.name]={'derived_from':f'boolq-{split}.parquet','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(rows)}
(r/'manifest.json').write_text(json.dumps(m,indent=2))
