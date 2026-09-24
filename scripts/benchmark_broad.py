"""Frozen external tasks, same requests/GPU, paired bootstrap and schema-cold timing."""
import sys,json,time,random,hashlib,math,argparse
from pathlib import Path
import numpy as np
import torch
from s1.predict import Predictor
from s1.benchmark_env import require_idle_gpu

def wilson(k,n):
 p=k/n;z=1.96;c=(p+z*z/(2*n))/(1+z*z/n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
 return [max(0,c-h),min(1,c+h)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--quality',default='runs/reflex-plus/checkpoint');ap.add_argument('--output',default='runs/broad');ap.add_argument('--continued',action='store_true');args=ap.parse_args()
 root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'baselines/laya-source'))
 from laya.agent import Agent
 from laya.common import render_options
 initial_gpu=require_idle_gpu()
 torch.set_num_threads(4);random.seed(9323);rng=np.random.default_rng(9323)
 models={'fast':Predictor(root/'runs/reflex-general/checkpoint'),'quality':Predictor(root/args.quality),'laya':Agent(str(root/'baselines/laya'),device='cuda')}
 laya=models['laya'];laya.cfg['max_len']=2048;laya.cfg['head_max_len']=1024
 ext=root/'data/external';answers={r['id']:r['ground_truth'] for r in map(json.loads,(ext/'bfcl-answers.jsonl').read_text().splitlines())};tasks={'bfcl_function_selection':[],'boolq':[]};excluded=[]
 for row in map(json.loads,(ext/'bfcl-multiple.jsonl').read_text().splitlines()):
  truth=answers[row['id']];names={k for d in truth for k in d}
  if len(names)!=1:excluded.append(row['id']);continue
  criteria={f['name']:f['description'] for f in row['function']}
  assert next(iter(names)) in criteria
  # Arguments deliberately omitted: this evaluates function identity only.
  q={'type':'choice','instructions':'Select the function that best fulfills the user request.','criteria':criteria}
  text='\n'.join(m['content'] for turn in row['question'] for m in turn)
  tasks['bfcl_function_selection'].append({'id':row['id'],'state':text,'question':q,'target':next(iter(names))})
 rows=list(map(json.loads,(ext/'boolq-dev.jsonl').read_text().splitlines()));indices=random.sample(range(len(rows)),500)
 for i in indices:
  r=rows[i];q={'type':'choice','instructions':'Answer this question using the passage: '+r['question'],'criteria':{'no':'No, the answer to the question is no.','yes':'Yes, the answer to the question is yes.'}}
  tasks['boolq'].append({'id':f'boolq_dev_{i}','state':r['passage'],'question':q,'target':'yes' if r['answer'] else 'no'})
 (ext/'broad-selected.jsonl').write_text(''.join(json.dumps(dict(r,task=t))+'\n' for t,rows in tasks.items() for r in rows))
 # Unscored warmups, then clear Reflex caches before every scored request.
 for m in models.values():
  for _ in range(5):m.predict('A sample passage.',{'q':{'type':'choice','instructions':'Choose.','criteria':{'a':'first','b':'second'}}})
 reports={};logs=[]
 for task,rows in tasks.items():
  records={n:[] for n in models}
  for i,row in enumerate(rows):
   order=list(models);random.shuffle(order)
   trunc=any(len(laya.tok(' '+x,add_special_tokens=False)['input_ids'])>48 for x in render_options(laya._to_internal(row['question'])))
   for name in order:
    model=models[name]
    if hasattr(model,'cache'):model.cache.clear()
    torch.cuda.synchronize();start=time.perf_counter();error=None;answer={}
    try:answer=model.predict(row['state'],{'q':row['question']})['answers']['q'];pred=answer['choice']
    except ValueError as e:error=str(e);pred=None
    torch.cuda.synchronize();latency=(time.perf_counter()-start)*1000
    record={'task':task,'id':row['id'],'system':name,'correct':pred==row['target'],'prediction':pred,'target':row['target'],'latency_ms':latency,'error':error,'abstain':answer.get('abstain'),'laya_option_truncation':trunc,'probabilities':answer.get('probabilities')}
    records[name].append(record);logs.append(record)
   if (i+1)%100==0:require_idle_gpu();print(task,i+1,flush=True)
  reports[task]={'n':len(rows),'systems':{},'paired_accuracy_difference_vs_laya':{}}
  for name,recs in records.items():
   correct=sum(r['correct'] for r in recs);ts=[r['latency_ms'] for r in recs if not r['error']]
   reports[task]['systems'][name]={'accuracy':correct/len(recs),'correct':correct,'wilson95':wilson(correct,len(recs)),'errors':sum(r['error'] is not None for r in recs),'abstentions':sum(r['abstain'] is True for r in recs),'latency_success_n':len(ts),'p50_ms':float(np.percentile(ts,50)),'p95_ms':float(np.percentile(ts,95)),'p99_ms':float(np.percentile(ts,99))}
  baseline=np.array([r['correct'] for r in records['laya']],dtype=float)
  for name in ['fast','quality']:
   delta=np.array([r['correct'] for r in records[name]],dtype=float)-baseline
   boots=delta[rng.integers(0,len(rows),(10000,len(rows)))].mean(1)
   reports[task]['paired_accuracy_difference_vs_laya'][name]={'difference':float(delta.mean()),'bootstrap95':np.percentile(boots,[2.5,97.5]).tolist()}
  reports[task]['laya_native_option_truncation_rows']=sum(r['laya_option_truncation'] for r in records['laya'])
  common=[i for i in range(len(rows)) if not records['laya'][i]['laya_option_truncation'] and all(not records[n][i]['error'] for n in models)]
  reports[task]['common_untruncated']={'n':len(common),'accuracy':{n:sum(records[n][i]['correct'] for i in common)/len(common) if common else None for n in models}}
  print(task,reports[task],flush=True)
 result={'initial_gpu':initial_gpu,'final_gpu':require_idle_gpu(),'seed':9323,'gpu':torch.cuda.get_device_name(),'tasks':reports,'excluded_ambiguous_bfcl_ids':excluded,'checkpoint_sha256':{n:m.checkpoint_sha256 for n,m in models.items() if hasattr(m,'checkpoint_sha256')},'protocol':[('Original checkpoints frozen before download; no fitting on these datasets.' if not args.continued else 'Continued quality checkpoint trained on separate BoolQ original train rows. The earlier 500 validation scores influenced curriculum choice; repeated evaluation is development, not blinded. BFCL never entered gradients.')+' Upstream pretraining overlap is unknown.','BFCL multiple category projected to function identity using names/descriptions. No argument generation, execution, or official BFCL score.','BoolQ is a seeded 500-row sample of public validation, not private test.','Random interleaved execution, five warmups, batch one, CUDA synchronization, tokenizer/transfers/readout included; candidate cache cleared each request for Reflex. Model load/HTTP/queueing excluded.','Errors count as wrong in accuracy; latency percentiles cover successful requests, error counts shown separately.','Unseen schemas intentionally abstain at runtime. Forced-choice accuracy is diagnostic and not automated coverage.','Laya keeps its native 48-token per-option cap; affected rows disclosed and common untruncated subset reported. State/head budgets 2048/1024.','Paired row bootstrap 10,000 resamples and Wilson intervals; correlated benchmark families may make row intervals optimistic.']}
 out=root/args.output;out.mkdir(exist_ok=True);(out/'benchmark.json').write_text(json.dumps(result,indent=2));(out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in logs))
if __name__=='__main__':main()
