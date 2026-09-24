import argparse
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from s1.predict import Predictor
import threading

class Request(BaseModel):
    state: str|dict|list
    questions: dict
    threshold: float=Field(default=.9,ge=0,le=1)

def create_app(checkpoint,device='cuda',router=False):
    import torch
    torch.set_num_threads(4)
    if router:
        from s1.router import RoutingPredictor
        model=RoutingPredictor.from_pretrained(checkpoint, device=device) if not Path(checkpoint).exists() and '/' in checkpoint else RoutingPredictor(device=device)
    else:model=Predictor(checkpoint,device)
    lock=threading.Lock(); app=FastAPI(title='Reflex-S1 research API')
    @app.get('/health')
    def health():return {'status':'ready','model':'Reflex-S1'}
    @app.post('/decide')
    def decide(request:Request):
        try:
            with lock: return model.predict(request.state,request.questions,request.threshold)
        except (ValueError,KeyError,TypeError) as e:raise HTTPException(422,str(e))
    return app
if __name__=='__main__':
    import uvicorn
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/reflex-general/checkpoint');p.add_argument('--device',default='cuda');p.add_argument('--port',type=int,default=8792);p.add_argument('--router',action='store_true');a=p.parse_args()
    uvicorn.run(create_app(a.checkpoint,a.device,a.router),host='127.0.0.1',port=a.port)
