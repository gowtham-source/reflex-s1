import torch
import pytest
from transformers import BertConfig,BertModel
from s1.model import DecisionModel,ModelConfig,SparseExperts
from s1.objectives import proper_loss,training_loss
from s1.predict import normalize_question,approval_gate

@pytest.fixture
def model(tmp_path):
    torch.set_num_threads(2)
    BertModel(BertConfig(vocab_size=100,hidden_size=32,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64)).save_pretrained(tmp_path)
    return DecisionModel(tmp_path,ModelConfig(width=16,heads=2)).eval()

def tokens(n,length):
    return {'input_ids':torch.randint(1,100,(n,length)),'attention_mask':torch.ones(n,length,dtype=torch.long)}

def test_adaptive_matches_fixed_and_option_permutation(model):
    s,o=tokens(2,9),tokens(4,5)
    with torch.no_grad():
        all_out=model(s,o)
        for d in [1,2,3]:
            a=model(s,o,adaptive=True,force_depth=d)['logits']
            torch.testing.assert_close(a,all_out['all_logits'][:,d-1])
        order=torch.tensor([2,0,3,1]); perm={k:v[order] for k,v in o.items()}
        z=model(s,perm,adaptive=True,force_depth=2)['logits']
        torch.testing.assert_close(z,all_out['all_logits'][:,1,order],rtol=1e-5,atol=1e-5)

def test_sparse_dispatch_equals_dense_reference():
    m=SparseExperts(8,4,2); x=torch.randn(2,3,8,requires_grad=True)
    y,_=m(x); p=m.router(x).softmax(-1); w,i=p.topk(2,-1);w=w/w.sum(-1,keepdim=True)
    ref=torch.zeros_like(x)
    for e,expert in enumerate(m.experts):ref+=expert(x)*(w*(i==e)).sum(-1,keepdim=True)
    torch.testing.assert_close(y,ref)
    y.square().sum().backward(); assert m.router.weight.grad.abs().sum()>0

def test_policy_and_encoder_gradients(model):
    model.train();out=model(tokens(3,8),tokens(4,5)); y=torch.eye(4)[:3]
    loss,_=training_loss(out,y);loss.backward()
    assert model.depth_policy[-1].weight.grad.abs().sum()>0
    assert model.encoder.embeddings.word_embeddings.weight.grad.abs().sum()>0

def test_proper_score_soft_target_optimum():
    y=torch.tensor([[.2,.3,.5]])
    assert proper_loss(y.log(),y)<proper_loss(torch.tensor([[0.,0.,5.]]),y)

def test_validation_and_approval():
    with pytest.raises(ValueError):normalize_question({'type':'choice','instructions':'pick','criteria':{'x':'same','y':'same'}})
    a={'choice':'allow','abstain':False}
    assert approval_gate(a)=='review'
    assert approval_gate(a,explicitly_authorized=True,reversible=True)=='allow'
    assert approval_gate(a,explicitly_authorized=True,reversible=True,forbidden=True)=='deny'

def test_save_roundtrip(model,tmp_path):
    # Validate state dict round trip independently of tokenizer files.
    path=tmp_path/'weights.pt';torch.save(model.state_dict(),path)
    clone={k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(torch.load(path,weights_only=True))
    for k,v in model.state_dict().items():torch.testing.assert_close(v,clone[k])


def test_dataset_rejects_group_and_text_leakage(tmp_path):
    import json
    from s1.data import load_data
    spec={'type':'choice','question':'Pick one','options':['first','second']}
    data={'tasks':{'a':spec},'splits':{s:{'a':[{'text':s,'target':[1,0],'group_id':s}]} for s in ['train','validation','calibration','test']}}
    path=tmp_path/'data.json';path.write_text(json.dumps(data));load_data(path)
    data['splits']['test']['a'][0]['group_id']='train'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Group leakage'):load_data(path)
    data['splits']['test']['a'][0]['group_id']='test'
    data['splits']['test']['a'][0]['text']=' TRAIN '
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Cross-split duplicate'):load_data(path)

def test_openjev_import_keeps_targets_aligned_and_excludes_metadata():
    from scripts.import_openjev import convert
    r={'kind':'choice','options':['z','a'],'target':[.8,.2],'state':{'text':'public'},'question':'Pick',
       'metadata':{'gold':'SECRET_ORACLE'},'source':'unit','group_id':'g','id':'i'}
    spec,row=convert(r)
    assert spec['options']==['a','z'] and row['target']==[.2,.8]
    assert 'SECRET_ORACLE' not in row['text'] and 'metadata' not in row
    r['target']=[1,1]
    with pytest.raises(ValueError,match='not_categorical'):convert(r)

def test_dynamic_questions_and_conditioning(model):
    from s1.data import render_state,batch_candidates
    model.cfg.condition_question=True
    assert 'claim' in render_state('evidence','claim',model.cfg)
    opts,features=batch_candidates(model,tokens(6,5),2,3,True)
    assert features.shape==(2,3,32)
    result=model(tokens(2,9),opts,option_features=features)
    assert result['all_logits'].shape==(2,3,3)

def test_pair_renderer_is_explicit_and_backward_compatible(model):
    from s1.data import render_state
    assert render_state('state','question',model.cfg)=='state'
    model.cfg.condition_question=True
    assert isinstance(render_state('state','question',model.cfg),str)
    model.cfg.pair_segments=True
    assert render_state('state','question',model.cfg)==('state','question')
