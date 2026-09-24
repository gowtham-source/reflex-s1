import json, hashlib
from pathlib import Path
import torch

def load_data(path):
    data=json.loads(Path(path).read_text())
    seen={}; groups={}
    if not data.get('tasks') or set(data.get('splits',{})) != {'train','validation','calibration','test'}:
        raise ValueError('Require tasks and train/validation/calibration/test splits')
    for task,spec in data['tasks'].items():
        kind=spec.get('type'); options=spec.get('options',[])
        if kind not in ['choice','noul','score'] or not isinstance(spec.get('question'),str) or not spec['question'].strip():
            raise ValueError('Invalid task schema')
        if not 2<=len(options)<=255 or (kind=='noul' and len(options)!=2) or (kind=='score' and len(options)>10):
            raise ValueError('Invalid option count')
        if any(not isinstance(o,str) or not o.strip() for o in options) or len(set(options))!=len(options):
            raise ValueError('Options require distinct nonempty descriptions')
        for ts in data['splits'].values():
            if not ts.get(task): raise ValueError('Every task requires nonempty rows in all four splits')
    for split, tasks in data['splits'].items():
        for task, rows in tasks.items():
            spec=data['tasks'][task]
            for r in rows:
                if not isinstance(r.get('text'),str) or not r['text'].strip(): raise ValueError('Nonempty text required')
                group=r.get('group_id')
                if group is not None:
                    if group in groups and groups[group]!=split: raise ValueError('Group leakage across splits')
                    groups[group]=split
                fingerprint=hashlib.sha256((task+'\0'+r.get('question',spec['question'])+'\0'+r['text'].strip().lower()).encode()).hexdigest()
                if fingerprint in seen and seen[fingerprint] != split:
                    raise ValueError(f'Cross-split duplicate in {task}: {seen[fingerprint]} / {split}')
                seen[fingerprint]=split
                y=r['target']
                if len(y)!=len(spec['options']) or any(not isinstance(v,(int,float)) or not 0<=v<=1 for v in y) or abs(sum(y)-1)>1e-5:
                    raise ValueError('Invalid target distribution')
    return data

def tokenize(tokenizer, texts, device, max_length):
    if texts and isinstance(texts[0],tuple):
        result=tokenizer([x[0] for x in texts],[x[1] for x in texts],padding=True,truncation=False,return_tensors='pt')
    else:result=tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
    if result['input_ids'].shape[1]>max_length:
        raise ValueError(f'Input length {result["input_ids"].shape[1]} exceeds {max_length}; provide bounded evidence')
    return {k:v.to(device) for k,v in result.items()}

def candidate_texts(spec):
    return [spec['question']+' [SEP] '+o for o in spec['options']]


def render_state(text,question,cfg):
    if cfg.condition_question and cfg.pair_segments:return (text,question)
    return text+' [SEP] Decision question: '+question if cfg.condition_question else text

def batch_candidates(model, tokens, batch_size, option_count, dynamic):
    if dynamic:
        _,features=model.encode(tokens)
        return None,features.reshape(batch_size,option_count,-1)
    return tokens,None
