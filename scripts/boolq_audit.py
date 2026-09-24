"""Separate previously scored BoolQ rows from remaining public validation rows."""
import json,random
from pathlib import Path
import torch
from s1.metrics import metrics

def main():
 root=Path(__file__).resolve().parents[1];data=json.loads((root/'data/decisions-evidence.json').read_text());rows=data['splits']['test']['boolq'];preds=[r for r in map(json.loads,(root/'runs/reflex-evidence/evaluation.predictions.jsonl').read_text().splitlines()) if r['task']=='boolq'];assert len(rows)==len(preds)
 # Same seed and draw as the original benchmark; no earlier random sampling precedes it.
 seen=set(random.Random(9323).sample(range(3270),500));report={}
 for name,ids in [('previously_scored_500',seen),('previously_unscored_2770',set(range(3270))-seen),('all_3270',set(range(3270)))]:
  chosen=[p for r,p in zip(rows,preds) if r['source_id'] in ids];m=metrics(torch.tensor([p['probabilities'] for p in chosen]).clamp_min(1e-12).log(),torch.tensor([p['target'] for p in chosen]));m['full_n']=len(ids);m['length_rejections']=len(ids)-len(chosen);m['full_denominator_accuracy']=m['accuracy']*len(chosen)/len(ids);report[name]=m
 (root/'runs/reflex-evidence/boolq-audit.json').write_text(json.dumps(report,indent=2));print({k:v['full_denominator_accuracy'] for k,v in report.items()})
if __name__=='__main__':main()
