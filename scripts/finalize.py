"""Summarize the final Open-Jev specialization without hiding earlier failures."""
import json
from pathlib import Path
from scripts.report import main as previous_report

def main():
    root=Path(__file__).resolve().parents[1];load=lambda s:json.loads((root/s).read_text())
    previous_report()
    (root/'docs/RESULTS_V2.md').write_text((root/'docs/RESULTS.md').read_text())
    e=load('runs/reflex-openjev/evaluation.json');ood=load('runs/reflex-openjev/openjev-ood.json');tr=load('runs/reflex-openjev/training.json');probe=load('runs/reflex-openjev/probes.json');audit=load('data/openjev-import.json')
    task=next(k for k in e['tasks'] if k.startswith('openjev_'));scope=e['tasks']['clinc150']['scope_breakdown']
    lines=['# Final measured results: Reflex-S1 + Open-Jev','',
      'Final checkpoint: `runs/reflex-openjev/checkpoint`. The model was actually trained and evaluated on the local L40S. It is a compact research model, not a validated universal agent or autonomous approval authority. The earlier matched dense comparison and failed approval iterations remain in [RESULTS_V2.md](RESULTS_V2.md).','',
      '## Final quality and latency','',
      '| Task | Test examples | Accuracy | ECE15 | Accepted at ≥0.9 | Accuracy among accepted | Warm p50 / p95 ms |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in e['tasks'].items():
        m=r['calibrated'];lat=e['latency'][name];accepted='none' if m['accuracy_at_0.9'] is None else f"{m['accuracy_at_0.9']:.2%}";label='Open-Jev silent failure' if name==task else name
        lines.append(f"| {label} | {m['n']} | {m['accuracy']:.2%} | {m['ece15']:.4f} | {m['coverage_at_0.9']:.2%} | {accepted} | {lat['p50_ms']:.2f} / {lat['p95_ms']:.2f} |")
    lines+=['',f"CLINC in-scope accuracy is {scope['in_scope_accuracy']:.2%}; OOS recall is {scope['oos_recall']:.2%}. Both matter: the combined score includes 1,000 OOS cases, and confidence gating does not reliably identify all unfamiliar requests.",'',
      'BANKING77 and CLINC are public natural-language intent datasets. All remaining tasks are synthetic controls, including the externally published Open-Jev corpus. Their accuracy is not an end-to-end agent/browser success rate. Approval performance on newly phrased evidence is weak; automatic approval is not qualified. Even a correct predicted label does not grant permission.','',
      '## Open-Jev integration and distribution shift','',
      f"Pinned dataset revision: `{audit['revision']}`. Actual training subset: `{audit['config']}`. Retained official splits: `{audit['retained']}`. All source groups and split labels were preserved, with no metadata/target leakage into inputs.",'',
      f"Official OOD: {ood['source_ood_n']} examples, {ood['scored_n']} scored, {ood['rejected_n']} overlength and explicitly rejected. **Full-denominator OOD accuracy with rejections counted as wrong: {ood['full_denominator_accuracy_rejections_wrong']:.2%}.** OOD probabilities use training-domain calibration; there is no OOD temperature fitting.",'',
      'See [OPENJEV_DATASET.md](OPENJEV_DATASET.md) for the audit and importer. The main test and OOD figures here are from the dedicated specialization, not copied from Open-Jev model reports. Only one of its configurations was used for training.','',
      '## Architectural evidence','',
      f"Parameters: {tr['parameters']:,}. Four experts, top-2 dispatch, shared recursion block, maximum depth three. Measured expert fractions: {e['expert_dispatch_fraction']}.",'',
      'Mean chosen depth: '+', '.join(f"{name}={r['mean_depth']:.3f}" for name,r in e['tasks'].items())+'.','',
      'All earlier validation sets favored depth one. The learned policy and computation skipping are implemented, but a useful adaptive-depth advantage has not been demonstrated. A separately trained dense baseline was faster and won on some controls; MoR+MoE v1 improved the two public intent scores. This is mixed single-seed evidence, not proof of architectural superiority. The final model received additional training, so comparing it directly to that five-epoch dense baseline would confound architecture, data and compute.','',
      'The probability heads optimize cross entropy, Brier and ordinal scores. The actual RL component is a contextual-bandit depth policy rewarded for proper-score quality minus compute cost. This is openly specified RLCD-inspired work, not a reproduction of Jev’s unpublished algorithm.','',
      '## Measurement scope','',
      'Warm request timings: batch one, 100 measurements per task, 10 warmups, cached schema embeddings, synchronized CUDA. Includes tokenization, device transfers and CPU typed result construction. Excludes model loading, HTTP/network, concurrency and queueing. Requests contain at most 256 tokens per state/candidate. Cold-schema probes keep model weights resident.','',
      '| Probe | p50 ms | p95 ms |','|---|---:|---:|']
    for name,r in probe['latency'].items():lines.append(f"| {name} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} |")
    lines+=['',f"Authored stress probes: {sum(r['correct'] for r in probe['diagnostics'])}/{len(probe['diagnostics'])}. These were inspected during development, so they are diagnostics. ASGI API smoke status: {probe['api_smoke']['status']}; invalid request status: {probe['invalid_request_status']}.",'',
      '## Training and verification','',
      f"Open-Jev stage: three epochs, {tr['steps']} optimizer updates, {tr['seconds']:.1f} seconds, peak PyTorch allocated memory {tr['max_gpu_memory_bytes']/2**20:.1f} MiB. The model warm-started the validation-selected v2 checkpoint. V1, dense, v2 and Open-Jev checkpoints and history remain available.",'',
      'Validation selects weights; separate calibration fits temperature. Test and OOD are not gradient inputs. Earlier public test results were inspected during development; repeated reporting is disclosed. Tests include option permutation, fixed/adaptive parity, sparse dispatch reference parity, nonzero gradients, proper soft-target scoring, trusted approval gating, split/group leakage and Open-Jev target remapping/metadata exclusion.','',
      'Artifacts: `evaluation.json`, `evaluation.predictions.jsonl`, `openjev-ood.json`, `probes.json`, `training.json`, `selection.json`, `history.json`, fitted checkpoint calibration and exact environment versions. Raw latency samples are retained. The API exposes typed predictions and never executes actions.','',
      '## Remaining work before broad deployment','',
      'Representative human-reviewed authorization data; real computer-use trajectories and screenshot grounding; stronger OOS/multilingual evaluation; many more natural-language decision schemas; fused sparse kernels and concurrent serving tests; separate MoE/MoR/RL ablations and multiple seeds. Full/head-only fine-tuning works now; LoRA and distributed training are not implemented.']
    (root/'docs/RESULTS.md').write_text('\n'.join(lines)+'\n')
    card={'name':'Reflex-S1-OpenJev','status':'research prototype','checkpoint':'runs/reflex-openjev/checkpoint','sha256':e['checkpoint_sha256'],'parameters':tr['parameters'],'architecture':'shared MiniLM encoder; decision-level MoR; top-2/4 MoE','training':'proper decision scores + REINFORCE computation policy; three specialization stages','dataset_revision':audit['revision'],'openjev_config':audit['config'],'quality':{name:r['calibrated'] for name,r in e['tasks'].items()},'ood':ood,'limitations':['approval generalization is weak','depth policy favors depth 1','no screenshots or external actions','no claim of proprietary RLCD reproduction'],'report':'docs/RESULTS.md'}
    (root/'MODEL_CARD.json').write_text(json.dumps(card,indent=2))
    schemas=json.loads((root/'data/decisions-openjev.json').read_text())['tasks'];(root/'examples').mkdir(exist_ok=True)
    (root/'examples/openjev-request.json').write_text(json.dumps({'state':'{"status":"success","required":"receipt","receipt":null,"message":"The payment was rejected."}', 'questions':{'silent_failure':{'type':'noul','instructions':schemas[task]['question'],'criteria':{'false':'no','true':'yes'}}}},indent=2))
if __name__=='__main__':main()
