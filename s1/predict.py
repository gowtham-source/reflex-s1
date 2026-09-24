import json, hashlib
from pathlib import Path
from collections import OrderedDict
import torch
from transformers import AutoTokenizer
from .model import DecisionModel
from .data import tokenize, candidate_texts, render_state

def schema_hash(spec):
    return hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()

def normalize_question(q):
    if not isinstance(q,dict): raise ValueError('Question must be an object')
    typ=q.get('type'); instruction=q.get('instructions')
    if typ not in ['choice','noul','score'] or not isinstance(instruction,str) or not instruction.strip():
        raise ValueError('Each question requires a valid type and nonempty instructions')
    criteria=q.get('criteria')
    if typ=='choice':
        if not isinstance(criteria,dict) or not 2<=len(criteria)<=255: raise ValueError('choice requires 2..255 unique option IDs')
        ids=list(criteria)
        if any(not isinstance(k,str) or not k for k in ids): raise ValueError('Option IDs must be nonempty strings')
        options=[str(v) if v else k for k,v in criteria.items()]
    elif typ=='score':
        if not isinstance(criteria,list) or not 2<=len(criteria)<=10: raise ValueError('score requires 2..10 ordered descriptions')
        ids=[str(i) for i in range(len(criteria))]; options=criteria
    else:
        ids=['false','true']; options=['false: statement is false','true: statement is true'] if criteria is None else [criteria['false'],criteria['true']]
    if any(not isinstance(s,str) or not s.strip() for s in options): raise ValueError('Option descriptions must be nonempty strings')
    if len(set(options))!=len(options): raise ValueError('Option descriptions must be distinct')
    return {'type':typ,'question':instruction,'options':options},ids

class Predictor:
    def __init__(self,checkpoint,device='cuda',cache_size=64):
        self.device=device
        self.model=DecisionModel.load(checkpoint,device)
        self.tokenizer=AutoTokenizer.from_pretrained(Path(checkpoint)/'encoder')
        path=Path(checkpoint)/'calibration.json'
        self.calibration=json.loads(path.read_text()) if path.exists() else {}
        digest=hashlib.sha256((Path(checkpoint)/'model.pt').read_bytes()).hexdigest()
        self.calibration={k:v for k,v in self.calibration.items() if v.get('checkpoint_sha256')==digest}
        self.checkpoint_sha256=digest
        self.cache=OrderedDict(); self.cache_size=cache_size

    @torch.inference_mode()
    def predict(self,state,questions,threshold=.9,force_depth=None):
        if not 0<=threshold<=1: raise ValueError('threshold must be in [0,1]')
        if force_depth is not None and not 1<=force_depth<=self.model.cfg.depth: raise ValueError('Invalid recursion depth')
        if not isinstance(questions,dict) or not 1<=len(questions)<=32: raise ValueError('Require 1..32 named questions')
        text=state if isinstance(state,str) else json.dumps(state,sort_keys=True,ensure_ascii=False,allow_nan=False)
        if not text.strip() or len(text)>32768: raise ValueError('State must be nonempty and bounded')
        states=tokenize(self.tokenizer,[text],self.device,self.model.cfg.max_length) if not self.model.cfg.condition_question else None
        answers={}
        unvalidated_script=any(('\u0400'<=ch<='\u052f') or ('\u0600'<=ch<='\u06ff') or ('\u0900'<=ch<='\u097f') or ('\u3040'<=ch<='\u30ff') or ('\u4e00'<=ch<='\u9fff') or ('\uac00'<=ch<='\ud7af') for ch in text)
        with torch.autocast(device_type=self.device.split(':')[0],dtype=torch.bfloat16,enabled=self.device.startswith('cuda')):
            state_features=self.model.encode(states) if states is not None else None
            for name,q in questions.items():
                spec,ids=normalize_question(q); key=schema_hash(spec)
                if self.model.cfg.condition_question:
                    states=tokenize(self.tokenizer,[render_state(text,spec['question'],self.model.cfg)],self.device,self.model.cfg.max_length)
                    state_features=self.model.encode(states)
                if key not in self.cache:
                    opts=tokenize(self.tokenizer,candidate_texts(spec),self.device,self.model.cfg.max_length)
                    _,features=self.model.encode(opts)
                    self.cache[key]=features
                    if len(self.cache)>self.cache_size: self.cache.popitem(last=False)
                else: self.cache.move_to_end(key)
                result=self.model(states,None,['choice','noul','score'].index(spec['type']),option_features=self.cache[key],adaptive=True,force_depth=force_depth,state_features=state_features)
                calibration=self.calibration.get(key)
                if calibration is None:
                    for candidate in self.calibration.values():
                        known=candidate.get('spec',{})
                        if candidate.get('question_prefixes') and spec['type']==known.get('type') and spec['options']==known.get('options') and any(spec['question'].startswith(prefix) for prefix in candidate['question_prefixes']):
                            calibration=candidate;break
                # Depth changes invalidate the fitted adaptive policy temperature.
                calibrated=calibration is not None and force_depth is None
                temperature=calibration['temperature'] if calibrated else 1.
                logits=result['logits'].float()
                if calibrated and 'logit_bias' in calibration:
                    logits=logits+torch.tensor(calibration['logit_bias'],device=logits.device)
                p=(logits/temperature).softmax(-1)[0].cpu().tolist()
                best=max(range(len(p)),key=p.__getitem__); confidence=p[best]
                answer={'probabilities':dict(zip(ids,p)),'confidence':confidence,'abstain':not calibrated or confidence<threshold or unvalidated_script,
                        'abstain_reasons':(["schema_not_calibrated"] if not calibrated else [])+(["below_threshold"] if confidence<threshold else [])+(["script_not_validated_for_english_encoder"] if unvalidated_script else []),
                        'calibration_available':calibrated,'calibration_scope':calibration.get('scope','exact_schema') if calibrated else None,'recursion_depth':int(result['depths'][0]),'temperature':temperature}
                if spec['type']=='choice':answer['choice']=ids[best]
                elif spec['type']=='score':answer['score']=sum(i*v for i,v in enumerate(p))
                else:answer['noul']=p[1]
                answers[name]=answer
        return {'answers':answers,'model':'Reflex-S1','checkpoint_sha256':self.checkpoint_sha256,'execution_authorized':False}

def approval_gate(answer, *, explicitly_authorized=False, reversible=False, forbidden=False):
    """Caller-supplied trusted policy metadata, never extracted from untrusted text."""
    if forbidden: return 'deny'
    if not explicitly_authorized or not reversible or answer.get('abstain',True): return 'review'
    return 'allow' if answer.get('choice')=='allow' else 'review'
