import argparse,json,time,statistics,hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from s1.model import DecisionModel
from s1.data import load_data,tokenize,candidate_texts,render_state
from s1.metrics import fit_temperature,metrics
from s1.predict import schema_hash,Predictor
from s1.benchmark_env import require_idle_gpu

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default='runs/reflex/checkpoint'); ap.add_argument('--data',default='data/decisions.json'); ap.add_argument('--device',default='cuda'); ap.add_argument('--output',default='runs/reflex/evaluation.json'); ap.add_argument('--batch-size',type=int,default=32)
    args=ap.parse_args(); torch.set_num_threads(4)
    model=DecisionModel.load(args.checkpoint,args.device); tok=AutoTokenizer.from_pretrained(Path(args.checkpoint)/'encoder'); data=load_data(args.data)
    expert_counts=torch.zeros(model.cfg.experts,device=args.device)
    def observe_router(module,inputs,output):
        selected=output.topk(model.cfg.top_k,-1).indices.flatten()
        expert_counts.add_(torch.bincount(selected,minlength=model.cfg.experts))
    router_hook=model.block.moe.router.register_forward_hook(observe_router)
    prediction_rows=[]
    report={'device':torch.cuda.get_device_name() if args.device.startswith('cuda') else args.device,'checkpoint_sha256':hashlib.sha256((Path(args.checkpoint)/'model.pt').read_bytes()).hexdigest(),'tasks':{}}; calibration={}
    with torch.inference_mode(),torch.autocast(device_type=args.device.split(':')[0],dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
        for task,spec in data['tasks'].items():
            options=tokenize(tok,candidate_texts(spec),args.device,model.cfg.max_length); _,features=model.encode(options)
            records={}
            for split in ['calibration','test']:
                rows=data['splits'][split][task]; logits=[]; targets=[]; depths=[]; fixed={d:[] for d in [1,model.cfg.depth]}
                for i in range(0,len(rows),args.batch_size):
                    batch=rows[i:i+args.batch_size]; states=tokenize(tok,[render_state(r['text'],r.get('question',spec['question']),model.cfg) for r in batch],args.device,model.cfg.max_length)
                    if any('question' in r for r in batch):
                        prompts=[text for r in batch for text in candidate_texts({**spec,'question':r.get('question',spec['question'])})]
                        opts=tokenize(tok,prompts,args.device,model.cfg.max_length);_,features=model.encode(opts);features=features.reshape(len(batch),len(spec['options']),-1)
                    typ=['choice','noul','score'].index(spec['type']); sf=model.encode(states)
                    result=model(states,None,typ,option_features=features,adaptive=True,state_features=sf)
                    logits.append(result['logits'].float().cpu()); depths.extend(result['depths'].tolist()); targets.extend(r['target'] for r in batch)
                    if split=='test':
                        for d in fixed:
                            z=model(states,None,typ,option_features=features,adaptive=True,force_depth=d,state_features=sf)
                            fixed[d].append(z['logits'].float().cpu())
                records[split]=(torch.cat(logits),torch.tensor(targets))
                if split=='test':
                    report['tasks'][task]={'mean_depth':float(np.mean(depths)),'fixed_depth_ablation_raw':{str(d):metrics(torch.cat(z),torch.tensor(targets)) for d,z in fixed.items()}}
            temp=fit_temperature(*records['calibration'])
            probs=torch.softmax(records['test'][0]/temp,-1).tolist()
            for row,prob in zip(data['splits']['test'][task],probs):
                prediction_rows.append({'task':task,'text_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),'target':row['target'],'probabilities':prob})
            calibration[schema_hash(spec)]={'temperature':temp,'task':task,'n':len(records['calibration'][0]),'spec':spec,'checkpoint_sha256':report['checkpoint_sha256']}
            if task=='boolq':
                calibration[schema_hash(spec)]['question_prefixes']=['Answer this question using the passage: ']
                calibration[schema_hash(spec)]['scope']='boolq_question_family'
            if task=='nli':
                calibration[schema_hash(spec)]['question_prefixes']=['Assess the claim: ', 'Determine whether the evidence supports this claim: ', 'Evaluate the statement: ']
                calibration[schema_hash(spec)]['scope']='nli_question_family'
            report['tasks'][task].update({'raw':metrics(*records['test']),'calibrated':metrics(*records['test'],temp),'temperature':temp,'synthetic':task not in ['banking77','clinc150','nli','boolq']})
            if task == 'clinc150':
                probs=torch.softmax(records['test'][0]/temp,-1); y=records['test'][1].argmax(-1); pred=probs.argmax(-1); oos=y==probs.shape[-1]-1
                report['tasks'][task]['scope_breakdown']={'in_scope_accuracy':float((pred[~oos]==y[~oos]).float().mean()),'oos_recall':float((pred[oos]==y[oos]).float().mean()),'oos_n':int(oos.sum())}
            print(task,json.dumps(report['tasks'][task]['calibrated']),flush=True)
    (Path(args.checkpoint)/'calibration.json').write_text(json.dumps(calibration,indent=2))
    router_hook.remove()
    report['expert_dispatch_fraction']=(expert_counts/expert_counts.sum()).cpu().tolist()
    report['expert_measurement_scope']='Calibration/test passes including fixed-depth diagnostics; excluded from latency timing.'
    del model
    if args.device.startswith('cuda'):torch.cuda.empty_cache()
    predictor=Predictor(args.checkpoint,args.device)
    latency={}
    for task in data['tasks']:
        if args.device.startswith('cuda'):require_idle_gpu()
        spec=data['tasks'][task]
        criteria=dict(zip([str(i) for i in range(len(spec['options']))],spec['options'])) if spec['type']=='choice' else spec['options'] if spec['type']=='score' else dict(zip(['false','true'],spec['options']))
        qs={task:{'type':spec['type'],'instructions':spec['question'],'criteria':criteria}}
        values=[]; samples=data['splits']['test'][task]
        for i in range(110):
            if args.device.startswith('cuda'):torch.cuda.synchronize()
            sample=samples[i%len(samples)]
            qs[task]['instructions']=sample.get('question',spec['question'])
            t=time.perf_counter(); predictor.predict(sample['text'],qs)
            if args.device.startswith('cuda'):torch.cuda.synchronize()
            if i>=10:values.append((time.perf_counter()-t)*1000)
        latency[task]={'p50_ms':float(np.percentile(values,50)),'p95_ms':float(np.percentile(values,95)),'p99_ms':float(np.percentile(values,99)),'n':len(values),'options':len(spec['options']),'scope':'warm in-process, batch=1, cached schema embeddings; includes tokenization, transfer, model, CPU typed readout; no HTTP; 10 warmups','samples_ms':values}
    report['latency']=latency
    Path(args.output).write_text(json.dumps(report,indent=2))
    Path(args.output).with_suffix('.predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prediction_rows))
if __name__=='__main__':main()
