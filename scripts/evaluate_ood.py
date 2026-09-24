"""Frozen Open-Jev OOD evaluation using training-domain calibration only."""
import json,argparse
from pathlib import Path
import torch
from s1.predict import Predictor
from s1.data import tokenize,candidate_texts,render_state
from s1.metrics import metrics
from s1.predict import schema_hash

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='runs/reflex-openjev/checkpoint');ap.add_argument('--output',default='runs/reflex-openjev/openjev-ood.json');a=ap.parse_args()
    torch.set_num_threads(4);predictor=Predictor(a.checkpoint);model=predictor.model;tok=predictor.tokenizer
    rows=[json.loads(s) for s in Path('data/openjev-ood.jsonl').read_text().splitlines()];report={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for task in sorted({r['task'] for r in rows}):
            subset=[r for r in rows if r['task']==task];spec=subset[0]['spec']
            eligible=[]
            for row in subset:
                try: tokenize(tok,[render_state(row['text'],spec['question'],model.cfg)],'cpu',model.cfg.max_length)
                except ValueError: continue
                eligible.append(row)
            subset=eligible
            if not subset: continue
            opt=tokenize(tok,candidate_texts(spec),'cuda',model.cfg.max_length);_,features=model.encode(opt);logits=[]
            for i in range(0,len(subset),32):
                state=tokenize(tok,[render_state(r['text'],spec['question'],model.cfg) for r in subset[i:i+32]],'cuda',model.cfg.max_length)
                out=model(state,None,['choice','noul','score'].index(spec['type']),option_features=features,adaptive=True)
                logits.append(out['logits'].cpu().float())
            targets=torch.tensor([r['target'] for r in subset]);temp=predictor.calibration.get(schema_hash(spec),{}).get('temperature',1.)
            report[task]=metrics(torch.cat(logits),targets,temp);report[task]['temperature']=temp
    audit=json.loads(Path('data/openjev-import.json').read_text());raw=audit['files']['ood']['raw_count'];retained=sum(r['n'] for r in report.values());correct=sum(r['accuracy']*r['n'] for r in report.values())
    result={'tasks':report,'source_ood_n':raw,'scored_n':retained,'rejected_n':raw-retained,'full_denominator_accuracy_rejections_wrong':correct/raw,'note':'Official OOD, no fitting or temperature refitting. Any filtered rows count as wrong in full-denominator score. Mostly synthetic; reserved Chinese wording/layouts exceed English encoder scope.'}
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
