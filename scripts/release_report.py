"""Final profiles and competitor evidence, assembled from measured results."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone

def main():
    root=Path(__file__).resolve().parents[1];load=lambda p:json.loads((root/p).read_text());profiles={}
    for name,run in [('fast','reflex-general'),('quality','reflex-plus')]:
        e=load(f'runs/{run}/evaluation.json');t=load(f'runs/{run}/training.json');o=load(f'runs/{run}/oos-operating-point.json');comparison=load('runs/comparison-'+('general' if name=='fast' else 'plus')+'/laya.json')
        profiles[name]={'run':run,'evaluation':e,'training':t,'oos':o,'comparison':comparison,'ood':load(f'runs/{run}/openjev-ood.json')}
    lines=['# Reflex-S1: trained profiles and competitive evaluation','',
      f"Report generated {datetime.now(timezone.utc).isoformat()}. Both profiles are trained on the local L40S and reloadable from saved artifacts. **This is a measured research system, not a proven universal Jev replacement.**",'',
      '| Profile | Checkpoint | Parameters | Backbone |', '|---|---|---:|---|']
    for name,r in profiles.items():lines.append(f"| {name} | `runs/{r['run']}/checkpoint` | {r['training']['parameters']:,} | {'MiniLM-L6, 384-wide' if name=='fast' else 'NLI MiniLMv2-L6, 768-wide'} |")
    lines+=['','Both use our shared recursive decision block, top-2/4 sparse experts and typed probability heads. The actual RL component selects computation depth using proper-score reward minus compute cost. Jev’s proprietary RLCD has not been reproduced. The final profiles condition the state encoder on each question; candidate scoring is non-autoregressive. Cross-question encoder sharing is limited to the earlier fast-specialist prototype.','',
      '## Full test results','',
      '| Task | Data type | Fast accuracy | Quality accuracy | Fast p50 ms | Quality p50 ms |','|---|---|---:|---:|---:|---:|']
    for task in profiles['fast']['evaluation']['tasks']:
        vals=[]
        for name in ['fast','quality']:
            r=profiles[name];m=r['oos']['after'] if task=='clinc150' else r['evaluation']['tasks'][task]['calibrated'];vals.append(m['accuracy'])
        label='Open-Jev silent failure' if task.startswith('openjev_') else task;kind='public natural language' if task in ['banking77','clinc150','nli'] else 'synthetic control'
        lines.append(f"| {label} | {kind} | {vals[0]:.2%} | {vals[1]:.2%} | {profiles['fast']['evaluation']['latency'][task]['p50_ms']:.2f} | {profiles['quality']['evaluation']['latency'][task]['p50_ms']:.2f} |")
    lines+=['','CLINC numbers above include the deployed OOS offset selected on validation and temperature fitted on calibration. The unadjusted model scores remain in each `evaluation.json`; the deployed scores are in `oos-operating-point.json`. Timings in this full-test table precede that tiny offset; the matched comparison below times the deployed version.','',
      '| Profile | CLINC in-scope accuracy | OOS recall | CLINC ECE | Open-Jev full-denominator OOD accuracy |','|---|---:|---:|---:|---:|']
    for name,r in profiles.items():lines.append(f"| {name} | {r['oos']['test_in_scope_accuracy']:.2%} | {r['oos']['test_oos_recall']:.2%} | {r['oos']['after']['ece15']:.4f} | {r['ood']['full_denominator_accuracy_rejections_wrong']:.2%} |")
    lines+=['','Open-Jev OOD counts all 1,920 examples, including all length rejections as wrong. The same original 1,838-example projection is offered to both models; their different tokenizers can cause additional rejections, recorded in the OOD JSON. OOD contains language/layout shift; neither English encoder is established as multilingual. Synthetic approval/DOM/retry/risk accuracy does not establish production authorization or completed computer-use success.','',
      '## Actual Laya comparison','',
      'Each profile was compared to pinned public Laya weights on 200 identical test requests per task, with the same GPU and alternating execution order. Laya option/context budgets were enlarged to avoid truncation. Our checkpoints received task-specific training; Laya weights were out-of-box. This is a deployable-artifact comparison, not matched training budgets or architectural proof. See [COMPETITION.md](COMPETITION.md).','',
      '| Task | Fast / Laya accuracy | Quality / Laya accuracy | Fast / Laya p50 ms | Quality / Laya p50 ms |','|---|---:|---:|---:|---:|']
    for task in profiles['fast']['comparison']['tasks']:
        f=profiles['fast']['comparison']['tasks'][task];q=profiles['quality']['comparison']['tasks'][task]
        label='Open-Jev silent failure' if task.startswith('openjev_') else task
        lines.append(f"| {label} | {f['reflex']['metrics']['accuracy']:.1%} / {f['laya']['metrics']['accuracy']:.1%} | {q['reflex']['metrics']['accuracy']:.1%} / {q['laya']['metrics']['accuracy']:.1%} | {f['reflex']['p50_ms']:.2f} / {f['laya']['p50_ms']:.2f} | {q['reflex']['p50_ms']:.2f} / {q['laya']['p50_ms']:.2f} |")
    lines+=['','## Separate generalization check — including unfavorable results','',
      '| Profile | SemIf authored144 correct | Laya correct |','|---|---:|---:|']
    for name,r in profiles.items():
        ext=r['comparison']['semif_authored144'];lines.append(f"| {name} | {ext['reflex']['correct']}/{ext['reflex']['n']} | {ext['laya']['correct']}/{ext['laya']['n']} |")
    lines+=['','This external authored/model-reviewed fixture was never gradient training data. Its earlier aggregate results influenced engineering direction, so repeated scores are development results, not newly blinded confirmation. It tests broader evidence/rules/candidate selection than our trained task suite and prevents us from claiming universal superiority based on narrow wins. No live TypeSafe Jev API was tested.','',
      '## Fresh external task checks','', 'See [BROAD_BENCHMARKS.md](BROAD_BENCHMARKS.md) for primary papers and the frozen protocol. Full measurements and paired confidence intervals are in `runs/broad/benchmark.json`. The BFCL projection tests function identity, not full tool calls; BoolQ tests passage-based yes/no decisions. Unseen schemas abstain in deployment; reported forced-choice accuracy is diagnostic.','', '## What was implemented and verified','',
      '- Custom typed decision architecture with real sparse dispatch, shared recursive weights and an RL depth policy.',
      '- Full/head-only fine-tuning, dynamic questions with gradients, native paired tokenization, soft labels and per-schema/family calibration.',
      '- Pinned Open-Jev raw-data importer preserving official groups/splits and excluding privileged metadata.',
      '- Validation-selected unknown-intent operating point, unsupported-script abstention and a separate trusted approval gate.',
      '- Local FastAPI service, Python inference, optional two-checkpoint serving router, checkpoint hashes and reproducible scripts.',
      '- Unit tests cover sparse/dense-reference parity, permutation invariance, adaptive/fixed parity, gradients, proper scores, group leakage, target remapping and renderer compatibility. Saved-model/API integration verification is recorded separately.','',
      '## Recursion, resources and limits','']
    for name,r in profiles.items():
        lines.append(f"**{name}:** last stage {r['training']['seconds']:.1f} s / {r['training']['steps']} updates; peak allocated GPU memory {r['training']['max_gpu_memory_bytes']/2**20:.1f} MiB. Mean chosen depths: "+', '.join(f"{t}={v['mean_depth']:.3f}" for t,v in r['evaluation']['tasks'].items())+'.')
        lines.append('')
    lines+=['The policy often favors depth one; we do not claim useful recursive specialization without the measured depth/quality trade-off. Earlier dense/recurrent comparisons and failures remain in the archived stage reports and raw JSON. Larger-vs-small differences also include different upstream training, tokenizers and fine-tuning histories.','',
      'Latency is batch-one in-process inference with CUDA synchronization. Warm task schemas cache candidate embeddings; changing questions require new candidate encodings. Model load, networking, queueing and concurrent serving are excluded. Full p50/p95/p99 samples are retained. Row-level confidence intervals do not account for within-family/image correlation; multiple seeds and group bootstrap are still needed for general superiority claims.','',
      'Remaining gaps: real browser/screenshot grounding, representative human-reviewed approval policies, multilingual/OOD robustness, broader unseen schemas, matched competitor fine-tuning, fused sparse kernels and multi-seed evaluation. No external actions execute from a model verdict.']
    broad=load('runs/broad/benchmark.json')
    lines+=['','## External results (frozen before download)','','| Task | Fast accuracy / p50 ms | Quality accuracy / p50 ms | Laya accuracy / p50 ms |','|---|---:|---:|---:|']
    for task,result in broad['tasks'].items():
        cells=[f"{result['systems'][name]['accuracy']:.1%} / {result['systems'][name]['p50_ms']:.2f}" for name in ['fast','quality','laya']]
        lines.append('| '+task+' (n='+str(result['n'])+') | '+' | '.join(cells)+' |')
    lines+=['','New-schema timings clear candidate caches. Raw reports include p95/p99, errors and native Laya option-truncation counts plus a common untruncated subset. This table is not an official BFCL or BoolQ leaderboard submission.']
    prior=root/'docs/RESULTS.md'
    if prior.exists() and not (root/'docs/RESULTS_OPENJEV.md').exists():(root/'docs/RESULTS_OPENJEV.md').write_text(prior.read_text())
    prior.write_text('\n'.join(lines)+'\n')
    card={'name':'Reflex-S1','status':'research; specialized competition demonstrated, broad superiority not established','profiles':{},'report':'docs/RESULTS.md','research':'docs/RESEARCH.md','no_external_action_execution':True}
    for name,r in profiles.items():
        cp=Path('runs')/r['run']/'checkpoint';card['profiles'][name]={'checkpoint':str(cp),'parameters':r['training']['parameters'],'weights_sha256':r['evaluation']['checkpoint_sha256'],'calibration_sha256':hashlib.sha256((root/cp/'calibration.json').read_bytes()).hexdigest(),'comparison':r['comparison']['semif_authored144']}
    (root/'MODEL_CARD.json').write_text(json.dumps(card,indent=2))
if __name__=='__main__':main()
