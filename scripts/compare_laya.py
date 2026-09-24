"""Matched requests/hardware; specialized Reflex vs out-of-box Laya, not equal training."""
import sys,json,random,time,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
from s1.predict import Predictor
from s1.benchmark_env import require_idle_gpu
from s1.metrics import metrics
from scripts.probe import q_from_spec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='runs/reflex-openjev/checkpoint');ap.add_argument('--data',default='data/decisions-openjev.json');ap.add_argument('--output',default='runs/comparison');args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'baselines/laya-source'))
    from laya.agent import Agent
    from laya.common import build_sequence,render_options
    initial_gpu=require_idle_gpu()
    torch.set_num_threads(4);laya=Agent(str(root/'baselines/laya'),device='cuda')
    if laya.device.type!='cuda':raise RuntimeError('Matched GPU comparison requires CUDA for both')
    reflex=Predictor(root/args.checkpoint)
    data=json.loads((root/args.data).read_text());results={};predictions=[];rng=random.Random(7271)
    for task,spec in data['tasks'].items():
        require_idle_gpu()
        rows=list(data['splits']['test'][task]);rng.shuffle(rows);rows=rows[:200];q=q_from_spec(spec)
        # Budget all complete option descriptions, up to source's per-option cap.
        internal=laya._to_internal(q)
        option_lengths=[len(laya.tok(' '+x,add_special_tokens=False)['input_ids'])+1 for x in render_options(internal)]
        if max(option_lengths)>49:raise ValueError('Laya per-option limit would truncate semantic content')
        instruction_length=max(len(laya.tok(spec['type']+' question: '+r.get('question',spec['question']),add_special_tokens=False)['input_ids']) for r in rows)
        head=sum(option_lengths)+instruction_length+32
        laya.cfg['head_max_len']=max(192,head);laya.cfg['max_len']=min(8192,laya.cfg['head_max_len']+512)
        records={name:{'p':[],'time':[]} for name in ['reflex','laya']};targets=[]
        # Warm each on the same first request; alternate execution order to reduce drift.
        for model in [reflex,laya]:
            for _ in range(5):model.predict(rows[0]['text'],{'q':q})
        for i,row in enumerate(rows):
            targets.append(row['target']);q['instructions']=row.get('question',spec['question'])
            for name,model in ([('reflex',reflex),('laya',laya)] if i%2==0 else [('laya',laya),('reflex',reflex)]):
                torch.cuda.synchronize();t=time.perf_counter();a=model.predict(row['text'],{'q':q})['answers']['q'];torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000
                p=[1-a['noul'],a['noul']] if spec['type']=='noul' else list(a['probabilities'].values());p=np.asarray(p);p=p/p.sum()
                records[name]['p'].append(p.tolist());records[name]['time'].append(ms)
                predictions.append({'task':task,'text_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),'system':name,'probabilities':p.tolist(),'target':row['target'],'latency_ms':ms})
        results[task]={'n':len(rows),'laya_head_budget':laya.cfg['head_max_len'],'laya_context_budget':laya.cfg['max_len']}
        for name,r in records.items():
            z=torch.tensor(r['p']).clamp_min(1e-12).log();results[task][name]={'metrics':metrics(z,torch.tensor(targets)),'p50_ms':float(np.percentile(r['time'],50)),'p95_ms':float(np.percentile(r['time'],95)),'p99_ms':float(np.percentile(r['time'],99))}
        print(task,{k:round(v['metrics']['accuracy'],4) for k,v in results[task].items() if isinstance(v,dict)},flush=True)
    # A separate external generalization set; never used for training.
    external=[json.loads(s) for s in (root/'baselines/semif-authored144.jsonl').read_text().splitlines()];ext={n:{'correct':0,'n':0,'errors':0} for n in ['reflex','laya']}
    laya.cfg['max_len']=1024;laya.cfg['head_max_len']=512
    for row in external:
        q={'type':'choice','instructions':row['question'],'criteria':{o['id']:o['description'] for o in row['options']}}
        for name,model in [('reflex',reflex),('laya',laya)]:
            ext[name]['n']+=1
            try:
                a=model.predict(row['state'],{'q':q})['answers']['q'];pred=list(q['criteria']).index(a['choice']);ext[name]['correct']+=int(pred==row['label'])
            except ValueError:ext[name]['errors']+=1;pred=None
            predictions.append({'task':'semif_authored144','id':row['id'],'system':name,'predicted':pred,'label':row['label']})
    report={'initial_gpu':initial_gpu,'final_gpu':require_idle_gpu(),'reflex_checkpoint':reflex.checkpoint_sha256,'laya_model_revision':'1c5edc17a7acd8701df6fc341c0d179f1c62c982','laya_code_revision':'c7527708f9f5220c669d8aa385077cd28d04708a','hardware':torch.cuda.get_device_name(),'seed':7271,'tasks':results,'semif_authored144':ext,'caveats':['Reflex is specialized on these task training splits; Laya is out-of-box. This is a deployment comparison, not matched-training architectural proof.','Same request state/instructions/descriptions and same GPU, but each system uses its own tokenizer and prompt rendering. Laya choice rendering also prefixes option IDs.','Laya budgets enlarged to preserve complete options/state; not deliberately left at high-cardinality truncating defaults.','Laya native temperatures vs Reflex task-fitted calibration; calibration comparison is not controlled.','No live Jev endpoint was tested.','SemIf authored144 is synthetic/model-reviewed, not human-adjudicated; errors count in the denominator.']}
    dest=root/args.output;dest.mkdir(exist_ok=True);(dest/'laya.json').write_text(json.dumps(report,indent=2));(dest/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
if __name__=='__main__':main()
