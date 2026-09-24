# Reflex-S1

A trained, non-generative System 1 decision model combining a shared recursive decision block, top-2/4 sparse experts, and a reinforcement-learning compute policy. Returns typed choice, boolean and ordinal distributions. Built separately under `/mnt/data/s1-mor-moe`, with two saved GPU-trained profiles and a fine-tuning framework.

**Scope:** intent routing, tool selection, approval proposals, textual UI controls, retry/risk policies, agent silent-failure detection, and evidence claims. Broad unseen-schema quality is measured separately and remains a limitation. CLINC150 is an assistant-intent dataset, not clinical data. Screenshot grounding and complete computer-use trajectories are not implemented.

- [Measured results and actual Laya comparison](docs/RESULTS.md)
- [Architecture and primary-source research](docs/RESEARCH.md)
- [Broader benchmark papers, protocol and limits](docs/BROAD_BENCHMARKS.md)
- [Fine-tuning and new-schema calibration](docs/FINETUNING.md)
- [Open-Jev dataset import and split audit](docs/OPENJEV_DATASET.md)
- [Attribution and licenses](docs/ATTRIBUTION.md)

Jev's proprietary RLCD is not sufficiently disclosed to reproduce. This implementation uses supervised proper-score fitting plus a genuine sampled computation-depth RL objective; it is an open adaptation, not a recovered Jev algorithm. The MoR component operates at question level in our head, rather than token-level recursion throughout the foundation encoder.

## Saved profiles

| Profile | Path | Intended use |
|---|---|---|
| fast | `runs/reflex-general/checkpoint` | Compact 23M model; validated fixed task schemas |
| quality | `runs/reflex-plus/checkpoint` | Larger NLI-initialized encoder; stronger evidence inference, not better on every task |

Both retain configuration, tokenizer, weights, calibration and hashes. The optional router sends known calibrated intent/control schemas to fast and evidence/new schemas to quality. This is an explicit serving policy; unknown schemas still abstain. Model weights remain advisory and never execute an action.

## Run locally

The existing runtime is `/mnt/data/workspace/.venv/bin/python`; existing workspace code and dependencies were not changed. For a fresh environment install this package with `python -m pip install -e '.[serve,test]'`. Observed versions are in `environment.txt`.

```bash
cd /mnt/data/s1-mor-moe
/mnt/data/workspace/.venv/bin/python -m scripts.serve --router
# Or one profile:
/mnt/data/workspace/.venv/bin/python -m scripts.serve --checkpoint runs/reflex-general/checkpoint
```

Only run one server per port. The API binds to `127.0.0.1:8792`; `POST /decide` accepts `state`, `questions`, and optional `threshold`. Requests are serialized for predictable cache/model access. Reported GPU microbenchmarks exclude HTTP, queueing and model load; concurrent service throughput is not established.

```python
from s1.router import RoutingPredictor
from s1.presets import question

model = RoutingPredictor()
result = model.predict(
    'The upload returned HTTP 200, but the stored object is empty.',
    {'claim': question('nli', claim='The file was successfully stored.')})
print(result)
# Other presets: tool_route, approval, dom_action, retry, risk,
# silent_failure, banking77, clinc150.
```

Custom questions accept `type='choice'`, natural-language `instructions`, and `criteria={id: description}`. `noul` is boolean; `score` takes ordered descriptions. New schemas produce probabilities but abstain until validated/calibrated. Exact presets preserve training schema calibration; the NLI claim family has a separately fitted family temperature. The external trusted `approval_gate` requires explicit permission and additional conditions; a model verdict does not grant permission.

Final profiles allow 512 tokenizer tokens for combined state/question and each candidate prompt. Overlength input raises an error. Choice supports 2–255 distinct options and 1–32 named questions per request. Different questions encode state separately; options within a question share its encoding. Reused exact schemas cache candidate embeddings. Clearly unvalidated scripts trigger abstention; English domain shift remains possible.

## Reproduce evaluation

```bash
python -m scripts.evaluate --data data/decisions-general.json --checkpoint runs/reflex-general/checkpoint --output runs/reflex-general/evaluation.json
python -m scripts.tune_oos --checkpoint runs/reflex-general/checkpoint --output runs/reflex-general/oos-operating-point.json
python -m scripts.compare_laya --checkpoint runs/reflex-general/checkpoint --data data/decisions-general.json --output runs/comparison-general
python -m scripts.benchmark_broad
python -m scripts.verify_release
python -m pytest -q
```

Run benchmarks sequentially on an otherwise idle GPU. `evaluate` refits calibration, so apply `tune_oos` afterwards to restore the deployed unknown-intent operating point. Existing reports preserve raw/deployed scores and row predictions. Do not fit on external benchmark results and then describe them as a fresh blinded test.

Training recipes and data format are documented in the fine-tuning guide. Public intent, SNLI and Open-Jev data are pinned; original controls are explicitly synthetic. Validation selects checkpoints, a separate split fits temperatures, and broad external evaluations never enter gradients. No live Jev API was tested. Laya comparisons use its actual public checkpoint; our task-specific training and its out-of-box deployment are not equal training budgets.
