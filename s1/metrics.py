import numpy as np
import torch
from sklearn.metrics import f1_score

def fit_temperature(logits, targets):
    # A fixed grid is deterministic and avoids unstable tiny-set optimization.
    scores=[]
    for t in np.geomspace(.1,10,161):
        loss=-(targets*torch.log_softmax(logits/t,-1)).sum(-1).mean().item()
        scores.append((loss,float(t)))
    return min(scores)[1]

def metrics(logits, targets, temperature=1., threshold=.9):
    p=torch.softmax(logits/temperature,-1).numpy(); y=targets.argmax(-1).numpy()
    pred=p.argmax(-1); conf=p.max(-1); correct=pred==y
    ece=0.
    for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        mask=(conf>lo)&(conf<=hi)
        if mask.any(): ece+=mask.mean()*abs(conf[mask].mean()-correct[mask].mean())
    selected=conf>=threshold
    n=len(y); accuracy=float(correct.mean()); z=1.96
    center=(accuracy+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n))/(1+z*z/n)
    return {'n':n,'accuracy':accuracy,'accuracy_wilson95':[float(max(0,center-half)),float(min(1,center+half))],
            'macro_f1':float(f1_score(y,pred,labels=list(range(p.shape[1])),average='macro',zero_division=0)),
            'soft_target_rows':int(((targets.numpy()>0).sum(-1)>1).sum()),
            'reference_mass_at_prediction':float(targets.numpy()[np.arange(n),pred].mean()),
            'nll':float(-(targets.numpy()*np.log(p.clip(1e-12))).sum(-1).mean()),
            'brier':float(((p-targets.numpy())**2).sum(-1).mean()),'ece15':float(ece),
            'coverage_at_0.9':float(selected.mean()),'accuracy_at_0.9':float(correct[selected].mean()) if selected.any() else None,
            'errors_at_0.9':int((~correct[selected]).sum()),
            'selection_curve':[{
                'threshold':float(t),'accepted':int((conf>=t).sum()),
                'errors':int(((conf>=t)&(~correct)).sum()),
                'coverage':float((conf>=t).mean())
            } for t in [0.,.5,.8,.9,.95,.99,.999]]}
