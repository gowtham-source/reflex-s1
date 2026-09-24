"""Local typed decision example using the exact fitted schema."""
import json
from pathlib import Path
from s1.predict import Predictor
from scripts.probe import q_from_spec

def main():
    data=json.loads(Path('data/decisions-general.json').read_text())
    model=Predictor('runs/reflex-general/checkpoint')
    print(json.dumps(model.predict('Find reference material. Reference item 123.',{'route':q_from_spec(data['tasks']['tool_route'])}),indent=2))
if __name__=='__main__':main()
