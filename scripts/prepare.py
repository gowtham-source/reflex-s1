"""Fetch pinned public data/model and generate original bounded controls."""
import csv, io, json, random, hashlib, itertools
from pathlib import Path
import requests
from huggingface_hub import HfApi, snapshot_download
ROOT=Path(__file__).resolve().parents[1]

def get(url):
    r=requests.get(url,timeout=90); r.raise_for_status(); return r

def main():
    out=ROOT/'data'; out.mkdir(exist_ok=True)
    manifest={'seed':42,'sources':[]}
    tasks={}; splits={s:{} for s in ['train','validation','calibration','test']}
    def add(task, question, options, typ='choice'):
        tasks[task]={'question':question,'options':options,'type':typ}
        for s in splits: splits[s][task]=[]
    seen=set()
    def row(split, task, text, label, source):
        fingerprint=(task,text.strip().lower())
        if fingerprint in seen: return
        seen.add(fingerprint)
        y=[float(i==label) for i in range(len(tasks[task]['options']))]
        splits[split][task].append({'text':text,'target':y,'source':source})
    repo='PolyAI-LDN/task-specific-datasets'
    sha='57ec275d8078af65b7731c2a98be812d844a6d6b'
    contents={}
    for name in ['train.csv','test.csv']:
        url=f'https://raw.githubusercontent.com/{repo}/{sha}/banking_data/{name}'
        r=get(url); contents[name]=list(csv.DictReader(io.StringIO(r.text)))
        manifest['sources'].append({'url':url,'sha256':hashlib.sha256(r.content).hexdigest(),'license':'CC-BY-4.0 (upstream repository LICENSE)','task':'banking77'})
        (out/name).write_bytes(r.content)
    labels=sorted({r['category'] for r in contents['train.csv']})
    add('banking77','Which banking intent is expressed?', [s.replace('_',' ') for s in labels])
    rng=random.Random(42)
    for label in labels:
        rows=[r for r in contents['train.csv'] if r['category']==label]; rng.shuffle(rows)
        for i,r in enumerate(rows):
            split='validation' if i<10 else 'calibration' if i<20 else 'train'
            row(split,'banking77',r['text'],labels.index(label),'banking77')
    for r in contents['test.csv']: row('test','banking77',r['text'],labels.index(r['category']),'banking77')

    repo='clinc/oos-eval'; sha='828f8093932c8fe6ca7936c3d2e52903b1c523de'
    url=f'https://raw.githubusercontent.com/{repo}/{sha}/data/data_full.json'
    r=get(url); clinc=r.json(); (out/'clinc_raw.json').write_bytes(r.content)
    manifest['sources'].append({'url':url,'sha256':hashlib.sha256(r.content).hexdigest(),'license':'CC-BY-3.0 (upstream LICENSE)','task':'clinc150'})
    labels=sorted({v[1] for v in clinc['train']})+['out of scope']
    add('clinc150','Which assistant intent is expressed?', [s.replace('_',' ') for s in labels])
    for origin, dest in [('test','test'),('train','train'),('val','validation')]:
        combined=clinc[origin]+clinc.get('oos_'+origin,[])
        for i,(text,label) in enumerate(combined):
            target='calibration' if origin=='val' and i%2 else dest
            row(target,'clinc150',text,len(labels)-1 if label=='oos' else labels.index(label),'clinc150')
    add('approval','Under the supplied authorization policy, what should happen?',
        ['allow: explicitly authorized reversible action','review: missing authorization or confirmation','deny: forbidden target or operation'])
    add('tool_route','Select the tool for the explicitly requested operation.',
        ['search: retrieve information','calculator: perform arithmetic','browser: interact with webpage','filesystem: manage local files','calendar: manage appointments','clarify: request missing instructions'])
    add('dom_action','Choose the requested visible and enabled UI action.', ['click Save','click Cancel','click Search','click Next','ask for clarification'])
    add('retry','Does the supplied retry policy permit another attempt?', ['false: do not retry','true: retry permitted'],'noul')
    add('risk','Rate the operation risk using this rubric.', ['low: local reversible read','medium: reversible modification','high: external or irreversible effect'],'score')
    # Independent template families across splits; generated controls, not real trajectories.
    templates={
      'train':['Request {id}: {body}', 'Context {id}. {body}', 'Task {id} — {body}'],
      'validation':['Review case {id}: {body}'],
      'calibration':['Decision record {id}. Evidence: {body}'],
      'test':['Observed workflow {id}. Current facts follow. {body}']}
    ops=['read file','list directory','open document','inspect page','write draft','edit note','rename local file','archive draft']
    for split in splits:
        n=1800 if split=='train' else 360
        for i in range(n):
            render=lambda body: rng.choice(templates[split]).format(id=f'{split}-{i}',body=body)
            forbidden=rng.choice([True,False]); auth=rng.choice([True,False]); reversible=rng.choice([True,False])
            body=f'Operation: {rng.choice(ops)}. Authorization: {"explicit" if auth else "absent"}. Reversible: {str(reversible).lower()}. Forbidden: {str(forbidden).lower()}. Policy: deny forbidden operations; otherwise allow only explicit authorization and reversible actions; otherwise review.'
            row(split,'approval',render(body),2 if forbidden else 0 if auth and reversible else 1,'original synthetic control')
            actions=[['Find reference material','Look up a fact','Retrieve documentation','Search for information'],['Calculate a sum','Multiply the numbers','Compute a ratio','Evaluate arithmetic'],['Click the login button','Navigate the website','Submit a browser form','Interact with a webpage'],['Rename a local file','Create a directory','Read a local document','Move a local file'],['Schedule an appointment','Reschedule a meeting','Cancel a calendar event','Create an appointment'],['Do the thing','Handle that','Take care of it','Proceed somehow']]
            y=i%6; phr=actions[y][3] if split=='test' else rng.choice(actions[y][:3])
            row(split,'tool_route',render(phr+f'. Reference item {rng.randrange(10000)}.'),y,'original synthetic control')
            y=i%5; buttons=['Save','Cancel','Search','Next']; requested=buttons[y] if y<4 else 'Delete'
            disabled=rng.choice([True,False])
            body=f'Goal: {requested}. UI elements: '+', '.join(f'{b} ({"disabled" if disabled and b==requested else "enabled"}, visible)' for b in rng.sample(buttons,4))
            row(split,'dom_action',render(body),4 if disabled or y==4 else y,'original synthetic DOM control')
            transient=rng.choice([True,False]); remaining=rng.choice([True,False]); idempotent=rng.choice([True,False])
            body=f'Transient error: {transient}. Retry budget available: {remaining}. Idempotent operation: {idempotent}. Policy: retry only if all three facts are true.'
            row(split,'retry',render(body),int(transient and remaining and idempotent),'original synthetic control')
            y=i%3; desc=['read-only local operation with no external effect','reversible local modification with undo','irreversible external transfer'][y]
            row(split,'risk',render(f'Operation {rng.choice(ops)}. Actual effect: {desc}.'),y,'original synthetic control')
    data={'tasks':tasks,'splits':splits}
    (out/'decisions.json').write_text(json.dumps(data))
    model_id='sentence-transformers/all-MiniLM-L6-v2'
    revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
    snapshot_download(model_id,revision=revision,local_dir=ROOT/'base',allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','README.md'])
    manifest['encoder']={'repo':model_id,'revision':revision,'license':'Apache-2.0'}
    manifest['counts']={s:{t:len(v) for t,v in ts.items()} for s,ts in splits.items()}
    manifest['dataset_sha256']=hashlib.sha256((out/'decisions.json').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__': main()
