"""Dual-checkpoint serving router: ultra-fast known schemas, high-capacity evidence/reasoning."""
from pathlib import Path
from .predict import Predictor, normalize_question, schema_hash

class RoutingPredictor:
    def __init__(self, fast_checkpoint='runs/reflex-general/checkpoint', general_checkpoint=None, device='cuda'):
        if general_checkpoint is None:
            if Path('runs/reflex-evidence/checkpoint').exists():
                general_checkpoint = 'runs/reflex-evidence/checkpoint'
            else:
                general_checkpoint = 'runs/reflex-plus/checkpoint'
        self.fast = Predictor(fast_checkpoint, device)
        self.general = Predictor(general_checkpoint, device)
        # Fast profile handles validated fixed task schemas (<7ms)
        self.fast_tasks = {'banking77', 'clinc150', 'tool_route', 'dom_action', 'retry', 'risk', 'approval'}

    @classmethod
    def from_pretrained(cls, repo_id_or_path="Gowtham25/reflex-s1", device='cuda', token=None, **kwargs):
        p = Path(repo_id_or_path)
        if not p.exists():
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(repo_id=repo_id_or_path, token=token, **kwargs))
        fast_path = p / "fast" if (p / "fast").exists() else p
        general_path = p / "quality" if (p / "quality").exists() else p
        return cls(fast_checkpoint=str(fast_path), general_checkpoint=str(general_path), device=device)

    def predict(self, state, questions, threshold=0.9):
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 32:
            raise ValueError('Require 1..32 questions')
        groups = {'fast': {}, 'general': {}}
        for name, q in questions.items():
            spec, _ = normalize_question(q)
            cal = self.fast.calibration.get(schema_hash(spec), {})
            task = cal.get('task', '')
            route = 'fast' if task in self.fast_tasks or task.startswith('openjev_') else 'general'
            groups[route][name] = q
        answers = {}
        selected = {}
        checksums = {}
        for name, qs in groups.items():
            if not qs:
                continue
            model = self.fast if name == 'fast' else self.general
            result = model.predict(state, qs, threshold)
            answers.update(result['answers'])
            selected.update({k: name for k in qs})
            checksums[name] = model.checkpoint_sha256
        return {
            'answers': {k: answers[k] for k in questions},
            'model': 'Reflex-S1 router',
            'selected_models': selected,
            'checkpoint_sha256': checksums,
            'execution_authorized': False
        }
