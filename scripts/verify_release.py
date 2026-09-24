"""Check saved artifacts, API probabilities/calibration and checkpoint reload on GPU."""
import argparse,json,hashlib
from pathlib import Path
import torch
from fastapi.testclient import TestClient
from s1.predict import Predictor,approval_gate
from scripts.serve import create_app
from scripts.probe import q_from_spec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='runs/reflex-general/checkpoint');ap.add_argument('--output',default='runs/reflex-general/verification.json');a=ap.parse_args();torch.set_num_threads(4)
    data=json.loads(Path('data/decisions-general.json').read_text());model=Predictor(a.checkpoint);checks={}
    for task in ['banking77','clinc150','approval','nli']:
        row=data['splits']['test'][task][0];spec=data['tasks'][task];q=q_from_spec(spec);q['instructions']=row.get('question',spec['question'])
        result=model.predict(row['text'],{task:q});answer=result['answers'][task];p=list(answer['probabilities'].values())
        assert abs(sum(p)-1)<1e-5 and all(0<=v<=1 for v in p)
        assert answer['calibration_available'] and result['execution_authorized'] is False
        checks[task]={'normalized':True,'calibration_scope':answer['calibration_scope']}
    q={'type':'choice','instructions':'An unseen task schema','criteria':{'a':'first alternative','b':'second alternative'}}
    assert model.predict('some evidence',{'q':q})['answers']['q']['abstain']
    checks['unknown_schema_abstains']=True
    nq=q_from_spec(data['tasks']['tool_route']);ans=model.predict('这是一个测试',{'q':nq})['answers']['q']
    assert ans['abstain'] and 'script_not_validated_for_english_encoder' in ans['abstain_reasons'];checks['unvalidated_script_abstains']=True
    assert approval_gate({'choice':'allow','abstain':False})=='review';checks['no_implicit_authorization']=True
    # Real saved-model ASGI integration. No public server is exposed.
    app=create_app(a.checkpoint)
    with TestClient(app) as client:
        response=client.post('/decide',json={'state':'Find reference material','questions':{'route':nq}})
        assert response.status_code==200 and 'answers' in response.json()
        assert client.post('/decide',json={'state':'x','questions':{},'threshold':2}).status_code==422
        checks['api']=True
    weights=Path(a.checkpoint)/'model.pt';metadata=Path(a.checkpoint)/'calibration.json'
    result={'checks':checks,'checkpoint_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'calibration_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest(),'torch':torch.__version__,'device':torch.cuda.get_device_name()}
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
