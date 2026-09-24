"""Add BoolQ training; public validation remains evaluation-only."""
import json,hashlib,collections
from pathlib import Path
from transformers import AutoTokenizer
from s1.data import load_data,tokenize,render_state
from s1.model import ModelConfig

def main():
 root=Path(__file__).resolve().parents[1];data=load_data(root/'data/decisions-general.json');ext=root/'data/external';tok=AutoTokenizer.from_pretrained(root/'base-nli');cfg=ModelConfig(condition_question=True,pair_segments=True,max_length=512)
 spec={'type':'choice','question':'Answer this question using the passage: ','options':['No, the answer to the question is no.','Yes, the answer to the question is yes.']};data['tasks']['boolq']=spec
 for split in data['splits']:data['splits'][split]['boolq']=[]
 rows={s:list(map(json.loads,(ext/f'boolq-{s}.jsonl').read_text().splitlines())) for s in ['train','dev']};held={r['passage'].strip().lower() for r in rows['dev']};audit={'excluded':collections.Counter(),'source_manifest':'data/external/manifest.json','note':'BoolQ public development benchmark was inspected before this run. Training uses only original train rows, grouped by passage; public dev remains test. Earlier 500 examples are development results, not fresh confirmation.'};seen=set()
 for origin in ['dev','train']:
  for i,r in enumerate(rows[origin]):
   passage=r['passage'].strip();group=hashlib.sha256(passage.lower().encode()).hexdigest();q=spec['question']+r['question'];key=(group,q)
   if origin=='train' and passage.lower() in held:audit['excluded']['train_passage_in_dev']+=1;continue
   if key in seen:audit['excluded']['duplicate']+=1;continue
   seen.add(key)
   try:tokenize(tok,[render_state(passage,q,cfg)],'cpu',512)
   except ValueError:audit['excluded'][origin+'_overlength']+=1;continue
   bucket=int(group[:8],16)%100;split='test' if origin=='dev' else 'train' if bucket<80 else 'validation' if bucket<90 else 'calibration'
   data['splits'][split]['boolq'].append({'text':passage,'question':q,'target':[float(not r['answer']),float(r['answer'])],'group_id':'boolq/'+group,'source':'BoolQ original '+origin,'source_id':i})
 path=root/'data/decisions-evidence.json';path.write_text(json.dumps(data));load_data(path);audit['counts']={s:len(v['boolq']) for s,v in data['splits'].items()};audit['data_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(root/'data/evidence-manifest.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
