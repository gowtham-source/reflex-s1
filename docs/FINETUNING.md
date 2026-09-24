# Fine-tuning and future specialization

The framework supports full-encoder or head-only supervised/soft-target fine-tuning today. It does not yet implement LoRA, distributed training, exact optimizer-state resumption, or live environment RL. `--resume` loads trained weights and creates a fresh optimizer; it is a warm start.

Use this JSON structure (at least two options; every split needs nonempty rows for each task):

```json
{
  "tasks": {
    "route": {"type": "choice", "question": "Which route?", "options": ["search", "clarify"]}
  },
  "splits": {
    "train": {"route": [{"text": "Find a paper", "target": [1, 0], "source": "human"}]},
    "validation": {"route": [{"text": "Search the manual", "target": [1, 0], "source": "human"}]},
    "calibration": {"route": [{"text": "Retrieve guidance", "target": [1, 0], "source": "human"}]},
    "test": {"route": [{"text": "Do something", "target": [0, 1], "source": "human"}]}
  }
}
```

This tiny example demonstrates format only; it is not enough for meaningful training/calibration. `target` can be a normalized soft distribution. Label smoothing is not automatically applied. Noul has two options ordered false/true; score uses ascending rubric levels. Use entity/episode/customer/time splits before exporting, not random individual-row splits from the same trajectories. The loader catches exact normalized cross-split duplicates but cannot detect semantic or episode leakage.

```bash
python -m scripts.train --data data/my-domain.json \
  --resume runs/reflex/checkpoint --output runs/my-domain \
  --mode heads --epochs 5
python -m scripts.evaluate --data data/my-domain.json \
  --checkpoint runs/my-domain/checkpoint --output runs/my-domain/evaluation.json
```

Use `--mode full` to update the encoder as well. Mix old task examples with new domain data to reduce forgetting, keep the final test frozen, and refit calibration after any weights/schema/depth change. The trainer saves config, weights and tokenizer, so exported inference does not require the original base-model download.

For computer use: collect goal + accessibility-tree/DOM + allowed action candidates + actual environment outcome, grouped by website and episode. Include blocked/hidden/disabled controls and a clarify/abstain action. Current prototype uses textual state; adding screenshots requires a vision encoder and a separately evaluated grounding head.

For approvals: collect independently reviewed policy outcomes, explicit policy context and adversarially conflicting untrusted content. Measure false-allow rate and recall for dangerous actions, not just total accuracy. The included trusted policy gate remains outside learned inference.

For policy RL: the implemented action is recursion depth. Extend to real tools only with an explicit environment, logged action propensities/outcomes and off-policy evaluation. Rewarding the selected class's correctness alone encourages overconfidence; retain proper scoring objectives and independent calibration.

Before claiming improvement, train `--depth 1 --experts 1 --rl-weight 0` using the same data/seed/epochs and compare quality, warm/cold latency, memory and expert/depth utilization. Inference-only fixed-depth comparisons do not replace retraining ablations. Run multiple seeds before statistical conclusions.

An optional `group_id` on each row enables strict cross-split group checks. Use globally unique episode IDs consistently across task heads; otherwise different decisions from the same episode can leak into test data.


## Dynamic questions and the broader encoder path

Rows may contain `question` to override the task's default question, while retaining that task's option vocabulary. The trainer encodes per-row candidates and reshapes their features into B×K×H without losing gradients. This supports evidence claims and other changing questions; it does not yet batch different option counts within one task.

Pass `--condition-question --max-length 512` to concatenate each question with its state before the encoder. The model config stores this mode, and inference uses the same renderer. This improves early state/question interaction while giving up state-encoder sharing between distinct questions. Candidate embeddings remain cached by exact schema. Existing 256-token shared-state checkpoints keep their original behavior.

After calibration, `scripts/tune_oos.py` optionally fits the CLINC-specific OOS bias on validation, then refits temperature on calibration. Keep its saved operating-point report with the checkpoint. Running ordinary `scripts.evaluate` afterwards refits unadjusted temperatures and resets this offset, so apply OOS tuning last when reproducing that deployment.

The optional `s1.router.RoutingPredictor` keeps the fast specialist and broader question-conditioned checkpoint resident. Its fixed, inspectable serving rule uses the fast checkpoint for known intent/approval/tool/DOM/retry/risk/silent-failure schemas and the larger checkpoint for NLI and new questions. This is a serving policy over two trained checkpoints, separate from the internal token-wise sparse MoE and decision-level recursive routing. Both checkpoint IDs are returned for auditability. It does not provide automatic authorization or a new calibration guarantee.

For NLI, calibration can be bound to the explicitly trained question-prefix family with the same option descriptions/order, rather than each unique claim string. Inference returns `calibration_scope="nli_question_family"` for those requests. Other unfamiliar schemas remain uncalibrated and abstain. This family fit still does not guarantee calibration on a new domain.

For encoders initialized from paired-input NLI models, `--pair-segments` uses the tokenizer's native text-pair representation when `--condition-question` is enabled. This is saved in the model config; older models preserve their original renderer. Use `--encoder-lr` and `--head-lr` to control adaptation versus head learning. The larger experiment uses `--base base-nli --encoder-lr 0.00001 --epochs 3 --batch-size 64 --condition-question --pair-segments --max-length 512`.

## Use the trained decision presets

```python
from s1.predict import Predictor, approval_gate
from s1.presets import question

model = Predictor('runs/reflex-general/checkpoint', device='cuda')
result = model.predict('Please move tomorrow’s meeting to Friday.',
                       {'intent': question('clinc150')})
# Tool routing: question('tool_route')
# API result verification: question('silent_failure')
# Evidence: question('nli', claim='The operation completed successfully.')
# Textual UI controls: question('dom_action')
# Approval proposal: question('approval'), then approval_gate with trusted flags
```

Presets contain the actual trained schema descriptions, so they preserve exact-schema calibration compatibility. They do not execute a selected action. The DOM preset is the bounded Save/Cancel/Search/Next control used in tests; it is not a general visual UI grounding model.
