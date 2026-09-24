"""Use existing calibration; never fit on the previously observed test."""
import json
from pathlib import Path
import torch
from s1.predict import Predictor
from s1.data import load_data
from scripts.probe import q_from_spec
from s1.metrics import metrics

def main():
    torch.set_num_threads(4);model=Predictor('runs/reflex-v2/checkpoint');data=load_data('data/decisions.json');spec=data['tasks']['approval'];q=q_from_spec(spec)
    rows=data['splits']['test']['approval'];predictions=[]
    for row in rows:
        a=model.predict(row['text'],{'approval':q})['answers']['approval'];predictions.append(list(a['probabilities'].values()))
    p=torch.tensor(predictions);targets=torch.tensor([r['target'] for r in rows]);report=metrics(p.clamp_min(1e-12).log(),targets)
    prediction=p.argmax(-1);truth=targets.argmax(-1);false_allow=(prediction==0)&(truth!=0)
    report['false_allow_count']=int(false_allow.sum());report['false_allow_confidence_at_0.9']=int((false_allow&(p.max(-1).values>=.9)).sum())
    report['note']='Original v1 approval test revisited after diagnosis; development regression, not a new held-out test.'
    Path('runs/reflex-v2/approval-regression.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
