"""Pinned raw-JSONL importer; never places privileged metadata into model input."""
import argparse,gzip,json,hashlib,collections,math
from pathlib import Path
import requests
from transformers import AutoTokenizer
from s1.data import candidate_texts,load_data
from s1.predict import schema_hash
REVISION='c67699e13d0ae25e35b77165a4b6b079bedc8aba'

def convert(row):
    kind=row['kind'];options=row['options'];target=row['target']
    if kind not in ['choice','noul','score']:raise ValueError('unsupported_kind')
    if len(options)!=len(target) or not 2<=len(options)<=255:raise ValueError('invalid_options')
    if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in target) or abs(sum(target)-1)>1e-6:raise ValueError('not_categorical_distribution')
    if len(set(options))!=len(options):raise ValueError('duplicate_options')
    if kind=='noul' and options!=['no','yes']:raise ValueError('unknown_boolean_order')
    if kind=='score':
        if len(options)>10:raise ValueError('unsupported_score_count')
        values=row.get('metadata',{}).get('score_values',list(range(len(options))))
        if values!=list(range(len(options))):raise ValueError('non_rank_score_values')
    # Choice permutations are semantically equivalent; move targets with options.
    if kind=='choice':
        order=sorted(range(len(options)),key=lambda i:options[i]);options=[options[i] for i in order];target=[target[i] for i in order]
    state=row['state'];text=state if isinstance(state,str) else json.dumps(state,sort_keys=True,ensure_ascii=False)
    spec={'type':kind,'question':row['question'],'options':options}
    record={'text':text,'target':target,'source':'Open-Jev/'+row['source'],'group_id':'openjev/'+row['group_id'],'original_id':row['id']}
    return spec,record

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='silent-failure-control-v1');ap.add_argument('--replay',default='data/decisions-v2.json');ap.add_argument('--output',default='data/decisions-openjev.json');args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];local=root/'data/openjev'/args.config;local.mkdir(parents=True,exist_ok=True)
    if args.config not in ['silent-failure-control-v1','release-v2-redistributable','browser-drone-expansion-v1-redistributable','citation-control-v1','entity-alignment-control-v1','amount-extraction-control-v1','email-selection-control-v1','phone-extraction-control-v1','context-retention-control-v1','sponsor-segment-control-v1','ir-control-v1','mailroom-control-v1']:raise ValueError('Unknown config')
    data=load_data(root/args.replay);tok=AutoTokenizer.from_pretrained(root/'base');maxlen=256
    audit={'revision':REVISION,'config':args.config,'files':{},'rejected':{},'retained':{},'max_length':maxlen};pending={};specs={};groups={};seen={};schema_lengths={}
    for split in ['train','validation','calibration','test','ood']:
        path=local/f'{split}.jsonl.gz';url=f'https://huggingface.co/datasets/ZefanCai/Open-Jev/resolve/{REVISION}/raw/{args.config}/{split}.jsonl.gz'
        if not path.exists():
            r=requests.get(url,timeout=120);r.raise_for_status();path.write_bytes(r.content)
        blob=path.read_bytes();audit['files'][split]={'url':url,'sha256':hashlib.sha256(blob).hexdigest()};rejected=collections.Counter();rawcount=0
        with gzip.open(path,'rt') as f:
            for line in f:
                row=json.loads(line);rawcount+=1
                if row['split']!=split:raise ValueError('source_split_mismatch')
                group=row['group_id']
                if group in groups and groups[group]!=split:raise ValueError('Source group crosses official splits')
                groups[group]=split
                try:spec,record=convert(row)
                except ValueError as exc:rejected[str(exc)]+=1;continue
                key='openjev_'+schema_hash(spec)[:16];specs[key]=spec
                if key not in schema_lengths:schema_lengths[key]=max(len(v) for v in tok(candidate_texts(spec))['input_ids'])
                if len(tok(record['text'])['input_ids'])>maxlen or schema_lengths[key]>maxlen:rejected['overlength_no_truncation']+=1;continue
                fingerprint=hashlib.sha256((key+'\0'+record['text'].strip().lower()).encode()).hexdigest()
                if fingerprint in seen and seen[fingerprint]!=split:raise ValueError('Input duplicates across source splits')
                seen[fingerprint]=split
                pending.setdefault(key,{}).setdefault(split,[]).append(record)
        audit['files'][split]['raw_count']=rawcount;audit['rejected'][split]=dict(rejected)
    # Fixed-schema trainer requires coverage in all fitting/evaluation splits.
    valid=[k for k,parts in pending.items() if all(parts.get(s) for s in data['splits'])]
    audit['unsupported_schema_rows']=sum(len(rs) for k,parts in pending.items() if k not in valid for rs in parts.values())
    ood=[]
    for key in valid:
        data['tasks'][key]=specs[key]
        for split in data['splits']:
            data['splits'][split][key]=pending[key][split]
        ood.extend({'task':key,'spec':specs[key],**r} for r in pending[key].get('ood',[]))
    if not valid:raise ValueError('No fixed schemas fit all splits; a row-dynamic trainer or longer context is needed')
    audit['retained']={s:sum(len(pending[k].get(s,[])) for k in valid) for s in ['train','validation','calibration','test','ood']}
    audit['schemas']={k:specs[k] for k in valid}
    path=root/args.output;path.write_text(json.dumps(data));load_data(path)
    (root/'data/openjev-ood.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ood))
    audit['output_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (root/'data/openjev-import.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
