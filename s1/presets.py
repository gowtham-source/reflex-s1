"""Task schemas used in actual training. These functions perform no actions."""
import json
from pathlib import Path

def question(task, *, claim=None):
    schemas=json.loads(Path(__file__).with_name('schemas.json').read_text())
    if task=='silent_failure':task=next(k for k in schemas if k.startswith('openjev_'))
    spec=schemas[task];kind=spec['type'];options=spec['options']
    if kind=='choice':
        if task=='nli':ids=['supported','insufficient','contradicted']
        elif task=='approval':ids=['allow','review','deny']
        else:ids=[s.split(':')[0].replace(' ','_') for s in options]
        criteria=dict(zip(ids,options))
    elif kind=='score':criteria=options
    else:criteria=dict(zip(['false','true'],options))
    instruction=spec['question']
    if task=='nli':
        if not isinstance(claim,str) or not claim.strip():raise ValueError('NLI requires a nonempty claim')
        instruction='Assess the claim: '+claim
    return {'type':kind,'instructions':instruction,'criteria':criteria}
