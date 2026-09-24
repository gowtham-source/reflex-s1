"""Generate the model card and readable results exclusively from saved artifacts."""
import json,hashlib
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    load=lambda p:json.loads((root/p).read_text())
    v1=load('runs/reflex/evaluation.json');dense=load('runs/dense/evaluation.json');v2=load('runs/reflex-v2/evaluation.json');train=load('runs/reflex-v2/training.json');probe=load('runs/reflex-v2/probes.json');reg=load('runs/reflex-v2/approval-regression.json')
    lines=['# Measured results — Reflex-S1','',
      'Measured on the local NVIDIA L40S, 2026-09-22. This is a research prototype, not a universal agent model. Checkpoints, split hashes, training history and row-level probability files are retained. Numbers below are generated from saved results.','',
      '## Final checkpoint','',
      '`runs/reflex-v2/checkpoint` is the final engineering checkpoint: 23,242,121 parameters. Five initial epochs followed by three warm-start epochs with broader approval controls and replay of the intent tasks. Validation selects the checkpoint within each run. The second run was motivated by observed v1 approval failure; repeated public-test scores are engineering measurements, not a fresh confirmatory study.','',
      '| Task | N | Accuracy | Macro-F1 | ECE | Coverage at 0.9 | Accuracy among accepted | p50 / p95 ms |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for task,r in v2['tasks'].items():
        m=r['calibrated'];lat=v2['latency'][task]; acc=m['accuracy_at_0.9']
        lines.append(f"| {task} | {m['n']} | {m['accuracy']:.2%} | {m['macro_f1']:.4f} | {m['ece15']:.4f} | {m['coverage_at_0.9']:.2%} | {acc:.2%} | {lat['p50_ms']:.2f} / {lat['p95_ms']:.2f} |" if acc is not None else f'| {task} | no accepted predictions |')
    scope=v2['tasks']['clinc150']['scope_breakdown']
    lines+=['',f"CLINC in-scope accuracy: **{scope['in_scope_accuracy']:.2%}**. Out-of-scope recall: **{scope['oos_recall']:.2%}**, on {scope['oos_n']} OOS examples. The combined score includes those difficult OOS cases; reporting only in-scope accuracy would hide this weakness.",'',
      '**Only BANKING77 and CLINC are real public intent datasets.** Approval/tool-route/DOM/retry/risk rows are generated controls. The approval v2 challenge uses new question-form facts and a new operation, but implements the same finite rule system; success is not broad approval-policy understanding or prompt-injection security. DOM scores measure text about enabled buttons, not screenshots or completed browser tasks.','',
      f"Revisited v1 approval regression: {reg['accuracy']:.2%}, false-allow predictions {reg['false_allow_count']}, false-allow predictions at confidence ≥0.9: {reg['false_allow_confidence_at_0.9']}. This was already observed during development and is not a new held-out benchmark.",'',
      '## Initial matched training comparison','',
      'Same original data, seed, five epochs, encoder and checkpoint-selection rule. Dense has one expert and one recursion; the proposed model has four experts and up to three recursions. This combined ablation cannot attribute differences separately to MoE, MoR or RL. One seed does not establish superiority.','',
      '| Task | MoR+MoE v1 | Dense | MoR+MoE p50 | Dense p50 |','|---|---:|---:|---:|---:|']
    for task in v1['tasks']:
        lines.append(f"| {task} | {v1['tasks'][task]['calibrated']['accuracy']:.2%} | {dense['tasks'][task]['calibrated']['accuracy']:.2%} | {v1['latency'][task]['p50_ms']:.2f} ms | {dense['latency'][task]['p50_ms']:.2f} ms |")
    lines+=['','The v2 approval data differ from this baseline comparison, so v2 is not a matched-compute ablation against the dense checkpoint.','',
      '## Recursion and expert use','',
      'Final mean recursion depths: '+', '.join(f"{t}={r['mean_depth']:.3f}" for t,r in v2['tasks'].items())+'.','',
      'Expert dispatch fractions: '+str(v2.get('expert_dispatch_fraction'))+'. Measured over calibration/test diagnostic passes, including forced-depth passes.','',
      'The depth policy has favored the cheapest depth. Adaptive routing exists and is trained with REINFORCE, but these workloads do not demonstrate useful adaptive-depth specialization. See `fixed_depth_ablation_raw` in each evaluation JSON for the actual effect of forcing one versus three recursions. No unmeasured speedup from recursion is claimed.','',
      '## Latency scope and cold schemas','',
      'Warm benchmark: batch one, 100 requests/task, 10 warmups, CUDA synchronization, tokenization + GPU transfer + model + CPU typed readout; schema embeddings cached. No HTTP, network, queueing or model load. Separate cold-schema probes clear only the schema cache, not model weights.','',
      '| Probe | p50 ms | p95 ms |','|---|---:|---:|']
    for name,r in probe['latency'].items():lines.append(f"| {name} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} |")
    lines+=['',f"Authored stress probes: {sum(r['correct'] for r in probe['diagnostics'])}/{len(probe['diagnostics'])}; these small post-training probes are diagnostic, not a representative benchmark. API smoke: {probe['api_smoke']}; invalid-request status: {probe['invalid_request_status']}.",'',
      '## Resources and reproducibility','',
      f"V1 training: {load('runs/reflex/training.json')['seconds']:.1f} s. Dense training: {load('runs/dense/training.json')['seconds']:.1f} s. V2 fine-tuning: {train['seconds']:.1f} s. V2 peak PyTorch allocated memory: {train['max_gpu_memory_bytes']/2**20:.1f} MiB (not total driver reservation).",'',
      'Reproduce v2 with `python -m scripts.augment_approval`, then `python -m scripts.train --data data/decisions-v2.json --resume runs/reflex/checkpoint --output runs/reflex-v2 --epochs 3`, then `python -m scripts.evaluate --data data/decisions-v2.json --checkpoint runs/reflex-v2/checkpoint --output runs/reflex-v2/evaluation.json`. Run probes with the matching checkpoint/output arguments.','',
      'All original and updated results remain in `runs/`. The initial data manifest retains raw-source hashes. The augmentation manifest records the v2 dataset hash and seed. The trained checkpoint binds fitted calibration to its own weight SHA-256. Seven unit tests cover routing parity, permutation behavior, gradient flow, proper scoring, authorization gates and leakage checks; GPU evaluation and the API smoke provide integration coverage.','',
      '## Remaining limits','',
      'Unknown-language/unknown-domain confidence can be wrong. A high-confidence model answer is not permission. Approval gates require trusted external authorization metadata; this model performs no actions. New schemas abstain until separately calibrated, but matching a schema does not detect distribution shift. Temperature scaling on easy synthetic data can select the grid floor and yield extreme probabilities; do not generalize those scores beyond the controls.','',
      'No screenshot grounding, autonomous computer-use success rate, external Jev/Laya/SemIf head-to-head run, distributed training, quantization or multi-seed uncertainty study was completed. The research and implementation are usable; production-level universal capability remains unestablished.']
    (root/'docs/RESULTS.md').write_text('\n'.join(lines)+'\n')
    card={'name':'Reflex-S1-v2','status':'research prototype','parameters':train['parameters'],'checkpoint':'runs/reflex-v2/checkpoint','checkpoint_sha256':v2['checkpoint_sha256'],'architecture':'MiniLM shared-state encoder; decision-level MoR; top-2/4 MoE; depth-policy RL; proper-score decision training','trained_on':'BANKING77, CLINC150+OOS, original synthetic controls','modalities':['text','serialized JSON'],'unsupported':['screenshot grounding','text generation','action execution'],'in_scope_clinc_accuracy':scope['in_scope_accuracy'],'oos_recall':scope['oos_recall'],'benchmark_file':'runs/reflex-v2/evaluation.json','calibration_note':'Exact-schema calibration availability is not a shift detector'}
    (root/'MODEL_CARD.json').write_text(json.dumps(card,indent=2))
if __name__=='__main__':main()
