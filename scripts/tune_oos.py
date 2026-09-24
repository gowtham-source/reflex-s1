"""Choose an OOS operating point on validation, refit temperature on calibration."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from s1.model import DecisionModel
from s1.data import load_data,tokenize,candidate_texts,render_state
from s1.predict import schema_hash
from s1.metrics import metrics,fit_temperature

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='runs/reflex-general/checkpoint');ap.add_argument('--data',default='data/decisions-general.json');ap.add_argument('--output',default='runs/reflex-general/oos-operating-point.json');ap.add_argument('--min-validation-in-scope',type=float,default=.93);a=ap.parse_args()
    torch.set_num_threads(4);model=DecisionModel.load(a.checkpoint,'cuda');tok=AutoTokenizer.from_pretrained(Path(a.checkpoint)/'encoder');data=load_data(a.data);spec=data['tasks']['clinc150'];all_records={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        opts=tokenize(tok,candidate_texts(spec),'cuda',model.cfg.max_length);_,features=model.encode(opts)
        for split in ['validation','calibration','test']:
            rows=data['splits'][split]['clinc150'];out=[]
            for i in range(0,len(rows),64):
                state=tokenize(tok,[render_state(r['text'],spec['question'],model.cfg) for r in rows[i:i+64]],'cuda',model.cfg.max_length)
                out.append(model(state,None,option_features=features,adaptive=True)['logits'].float().cpu())
            all_records[split]=(torch.cat(out),torch.tensor([r['target'] for r in rows]))
    z,y=all_records['validation'];y=y.argmax(-1);oos=y==z.shape[-1]-1
    candidates=[]
    for bias in np.linspace(-2,8,101):
        shifted=z.clone();shifted[:,-1]+=float(bias);pred=shifted.argmax(-1)
        inside=float((pred[~oos]==y[~oos]).float().mean());outside=float((pred[oos]==y[oos]).float().mean())
        candidates.append({'bias':float(bias),'validation_in_scope_accuracy':inside,'validation_oos_recall':outside,'balanced_objective':(inside+outside)/2})
    feasible=[r for r in candidates if r['validation_in_scope_accuracy']>=a.min_validation_in_scope]
    chosen=max(feasible or candidates,key=lambda r:(r['balanced_objective'],-abs(r['bias'])))
    cal_z,cal_y=all_records['calibration'];cal_z=cal_z.clone();cal_z[:,-1]+=chosen['bias'];temperature=fit_temperature(cal_z,cal_y)
    test_z,test_y=all_records['test'];before=metrics(test_z,test_y);adjusted=test_z.clone();adjusted[:,-1]+=chosen['bias'];after=metrics(adjusted,test_y,temperature)
    pred=adjusted.argmax(-1);y=test_y.argmax(-1);oos=y==adjusted.shape[-1]-1
    report={'chosen':chosen,'constraint_feasible':bool(feasible),'temperature':temperature,'before_raw':before,'after':after,'test_in_scope_accuracy':float((pred[~oos]==y[~oos]).float().mean()),'test_oos_recall':float((pred[oos]==y[oos]).float().mean()),'selection':'validation-only balanced in-scope/OOS accuracy subject to in-scope floor; then calibration-only temperature; test not used for selection','grid':candidates}
    path=Path(a.checkpoint)/'calibration.json';calibration=json.loads(path.read_text());key=schema_hash(spec)
    bias=[0.]*len(spec['options']);bias[-1]=chosen['bias'];calibration[key].update({'temperature':temperature,'logit_bias':bias,'operating_point':chosen})
    path.write_text(json.dumps(calibration,indent=2));Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='grid'},indent=2))
if __name__=='__main__':main()
