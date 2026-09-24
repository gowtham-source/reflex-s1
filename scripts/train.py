"""Full/head fine-tuning and calibration-oriented recursion policy learning."""
import argparse, json, random, time, platform, hashlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from s1.model import DecisionModel, ModelConfig
from s1.data import load_data, tokenize, candidate_texts, render_state, batch_candidates
from s1.objectives import training_loss, proper_loss

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/decisions.json'); ap.add_argument('--base',default='base')
    ap.add_argument('--output',default='runs/reflex'); ap.add_argument('--resume')
    ap.add_argument('--epochs',type=int,default=5); ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--mode',choices=['full','heads'],default='full'); ap.add_argument('--device',default='cuda')
    ap.add_argument('--seed',type=int,default=42); ap.add_argument('--depth',type=int,default=3)
    ap.add_argument('--experts',type=int,default=4); ap.add_argument('--rl-weight',type=float,default=.2)
    ap.add_argument('--pair-segments',action='store_true'); ap.add_argument('--encoder-lr',type=float,default=2e-5); ap.add_argument('--head-lr',type=float,default=3e-4)
    ap.add_argument('--condition-question',action='store_true'); ap.add_argument('--max-length',type=int)
    args=ap.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA required; no silent CPU fallback')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    data=load_data(args.data)
    model=DecisionModel.load(args.resume,args.device) if args.resume else DecisionModel(args.base,ModelConfig(depth=args.depth,experts=args.experts,top_k=min(2,args.experts))).to(args.device)
    if args.condition_question:model.cfg.condition_question=True
    if args.pair_segments:model.cfg.pair_segments=True
    if args.max_length:model.cfg.max_length=args.max_length
    tok=AutoTokenizer.from_pretrained(Path(args.resume)/'encoder' if args.resume else args.base)
    if args.mode=='heads':
        for p in model.encoder.parameters(): p.requires_grad_(False)
    encoder=list(model.encoder.parameters()); encoder_ids={id(p) for p in encoder}
    optimizer=torch.optim.AdamW([{'params':[p for p in encoder if p.requires_grad],'lr':args.encoder_lr},
        {'params':[p for p in model.parameters() if id(p) not in encoder_ids],'lr':args.head_lr}],weight_decay=.01)
    # Pre-tokenize without truncation. CPU tensor cache keeps GPU memory bounded.
    tokens={}; options={}; dynamic_options={}
    for task,spec in data['tasks'].items():
        options[task]=tokenize(tok,candidate_texts(spec),'cpu',model.cfg.max_length)
        for split,ts in data['splits'].items():
            rows=ts.get(task,[])
            if rows:
                tokens[split,task]=tokenize(tok,[render_state(r['text'],r.get('question',spec['question']),model.cfg) for r in rows],'cpu',model.cfg.max_length)
                if any('question' in r for r in rows):
                    prompts=[text for r in rows for text in candidate_texts({**spec,'question':r.get('question',spec['question'])})]
                    dynamic_options[split,task]=tokenize(tok,prompts,'cpu',model.cfg.max_length)
    best=float('inf'); step=0; start=time.time(); history=[]
    tasks=list(data['tasks'])
    for epoch in range(args.epochs):
        model.train()
        if args.mode=='heads': model.encoder.eval()
        batches=[]
        for task in tasks:
            indices=list(range(len(data['splits']['train'][task]))); random.shuffle(indices)
            batches.extend((task,indices[i:i+args.batch_size]) for i in range(0,len(indices),args.batch_size))
        random.shuffle(batches)
        for task,idx in batches:
            spec=data['tasks'][task]; typ=['choice','noul','score'].index(spec['type'])
            state={k:v[idx].to(args.device) for k,v in tokens['train',task].items()}
            dynamic=('train',task) in dynamic_options
            if dynamic:
                k_options=len(spec['options']);flat=[i*k_options+j for i in idx for j in range(k_options)]
                opts={k:v[flat].to(args.device) for k,v in dynamic_options['train',task].items()}
            else:opts={k:v.to(args.device) for k,v in options[task].items()}
            targets=torch.tensor([data['splits']['train'][task][i]['target'] for i in idx],device=args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device.split(':')[0],dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
                opts_input,opt_features=batch_candidates(model,opts,len(idx),len(spec['options']),dynamic)
                result=model(state,opts_input,typ,option_features=opt_features)
                loss,parts=training_loss(result,targets,typ==2,rl_weight=args.rl_weight if epoch else 0)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
            optimizer.step(); step+=1
            if step%50==0:
                record={'epoch':epoch+1,'step':step,'task':task,'loss':float(loss.detach()),'seconds':round(time.time()-start,1),**parts}
                print(json.dumps(record),flush=True); history.append(record)
        # Equal-task validation NLL selects checkpoints; test and calibration untouched.
        model.eval(); vals={}; depths={}
        with torch.inference_mode():
            for task in tasks:
                rows=data['splits']['validation'][task]; vals[task]=[]; depths[task]=[]
                opts={k:v.to(args.device) for k,v in options[task].items()}
                for i in range(0,len(rows),args.batch_size):
                    state={k:v[i:i+args.batch_size].to(args.device) for k,v in tokens['validation',task].items()}
                    y=torch.tensor([r['target'] for r in rows[i:i+args.batch_size]],device=args.device)
                    with torch.autocast(device_type=args.device.split(':')[0],dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
                        dynamic=('validation',task) in dynamic_options
                        if dynamic:
                            k_options=len(data['tasks'][task]['options']);flat_start=i*k_options;flat_end=(i+len(y))*k_options
                            batch_opts={k:v[flat_start:flat_end].to(args.device) for k,v in dynamic_options['validation',task].items()}
                        else:batch_opts=opts
                        opts_input,opt_features=batch_candidates(model,batch_opts,len(y),len(data['tasks'][task]['options']),dynamic)
                        result=model(state,opts_input,['choice','noul','score'].index(data['tasks'][task]['type']),option_features=opt_features,adaptive=True)
                    vals[task].extend(proper_loss(result['logits'],y).tolist()); depths[task].extend(result['depths'].tolist())
        score=float(np.mean([np.mean(v) for v in vals.values()]))
        record={'epoch':epoch+1,'validation_proper_equal_task':score,'by_task':{k:float(np.mean(v)) for k,v in vals.items()},'depths':{k:float(np.mean(v)) for k,v in depths.items()}}
        print(json.dumps(record),flush=True); history.append(record)
        if score<best:
            best=score; model.save(out/'checkpoint',tok)
            (out/'selection.json').write_text(json.dumps(record,indent=2))
        (out/'history.json').write_text(json.dumps(history,indent=2))
    manifest={'completed_utc':datetime.now(timezone.utc).isoformat(),'args':vars(args),'seconds':time.time()-start,'steps':step,'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,'python':platform.python_version(),'device':torch.cuda.get_device_name() if args.device.startswith('cuda') else args.device,'data_sha256':hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),'max_gpu_memory_bytes':torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None}
    (out/'training.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
