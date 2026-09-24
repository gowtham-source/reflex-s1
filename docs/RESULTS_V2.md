# Measured results — Reflex-S1

Measured on the local NVIDIA L40S, 2026-09-22. This is a research prototype, not a universal agent model. Checkpoints, split hashes, training history and row-level probability files are retained. Numbers below are generated from saved results.

## Final checkpoint

`runs/reflex-v2/checkpoint` is the final engineering checkpoint: 23,242,121 parameters. Five initial epochs followed by three warm-start epochs with broader approval controls and replay of the intent tasks. Validation selects the checkpoint within each run. The second run was motivated by observed v1 approval failure; repeated public-test scores are engineering measurements, not a fresh confirmatory study.

| Task | N | Accuracy | Macro-F1 | ECE | Coverage at 0.9 | Accuracy among accepted | p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 3073 | 92.09% | 0.9210 | 0.0105 | 77.71% | 98.28% | 6.56 / 6.75 |
| clinc150 | 5500 | 84.22% | 0.8916 | 0.0493 | 77.60% | 96.63% | 6.73 / 6.88 |
| approval | 600 | 53.50% | 0.4478 | 0.4661 | 98.50% | 53.13% | 6.93 / 7.00 |
| tool_route | 360 | 83.33% | 0.7778 | 0.1148 | 83.06% | 100.00% | 6.63 / 6.81 |
| dom_action | 360 | 100.00% | 1.0000 | 0.0000 | 100.00% | 100.00% | 6.84 / 6.89 |
| retry | 360 | 100.00% | 1.0000 | 0.0000 | 100.00% | 100.00% | 6.65 / 6.77 |
| risk | 360 | 100.00% | 1.0000 | 0.0000 | 100.00% | 100.00% | 6.77 / 6.92 |

CLINC in-scope accuracy: **95.38%**. Out-of-scope recall: **34.00%**, on 1000 OOS examples. The combined score includes those difficult OOS cases; reporting only in-scope accuracy would hide this weakness.

**Only BANKING77 and CLINC are real public intent datasets.** Approval/tool-route/DOM/retry/risk rows are generated controls. The approval v2 challenge uses new question-form facts and a new operation, but implements the same finite rule system; success is not broad approval-policy understanding or prompt-injection security. DOM scores measure text about enabled buttons, not screenshots or completed browser tasks.

Revisited v1 approval regression: 100.00%, false-allow predictions 0, false-allow predictions at confidence ≥0.9: 0. This was already observed during development and is not a new held-out benchmark.

## Initial matched training comparison

Same original data, seed, five epochs, encoder and checkpoint-selection rule. Dense has one expert and one recursion; the proposed model has four experts and up to three recursions. This combined ablation cannot attribute differences separately to MoE, MoR or RL. One seed does not establish superiority.

| Task | MoR+MoE v1 | Dense | MoR+MoE p50 | Dense p50 |
|---|---:|---:|---:|---:|
| banking77 | 90.92% | 90.66% | 6.73 ms | 5.93 ms |
| clinc150 | 85.78% | 83.75% | 6.86 ms | 6.00 ms |
| approval | 73.61% | 76.39% | 6.82 ms | 6.15 ms |
| tool_route | 93.33% | 100.00% | 6.73 ms | 6.04 ms |
| dom_action | 100.00% | 100.00% | 6.91 ms | 6.10 ms |
| retry | 100.00% | 100.00% | 6.74 ms | 6.11 ms |
| risk | 100.00% | 100.00% | 6.67 ms | 6.06 ms |

The v2 approval data differ from this baseline comparison, so v2 is not a matched-compute ablation against the dense checkpoint.

## Recursion and expert use

Final mean recursion depths: banking77=1.000, clinc150=1.000, approval=1.000, tool_route=1.000, dom_action=1.000, retry=1.000, risk=1.000.

Expert dispatch fractions: [0.20941467583179474, 0.2823026478290558, 0.2368306815624237, 0.27145200967788696]. Measured over calibration/test diagnostic passes, including forced-depth passes.

The depth policy has favored the cheapest depth. Adaptive routing exists and is trained with REINFORCE, but these workloads do not demonstrate useful adaptive-depth specialization. See `fixed_depth_ablation_raw` in each evaluation JSON for the actual effect of forcing one versus three recursions. No unmeasured speedup from recursion is claimed.

## Latency scope and cold schemas

Warm benchmark: batch one, 100 requests/task, 10 warmups, CUDA synchronization, tokenization + GPU transfer + model + CPU typed readout; schema embeddings cached. No HTTP, network, queueing or model load. Separate cold-schema probes clear only the schema cache, not model weights.

| Probe | p50 ms | p95 ms |
|---|---:|---:|
| warm_route | 6.35 | 6.50 |
| cold_route | 11.22 | 11.50 |
| warm_151_options | 6.85 | 6.90 |
| cold_151_options | 15.30 | 15.46 |
| five_questions | 16.98 | 17.11 |

Authored stress probes: 10/10; these small post-training probes are diagnostic, not a representative benchmark. API smoke: {'status': 200, 'valid_answers': True}; invalid-request status: 422.

## Resources and reproducibility

V1 training: 267.5 s. Dense training: 189.7 s. V2 fine-tuning: 189.1 s. V2 peak PyTorch allocated memory: 805.6 MiB (not total driver reservation).

Reproduce v2 with `python -m scripts.augment_approval`, then `python -m scripts.train --data data/decisions-v2.json --resume runs/reflex/checkpoint --output runs/reflex-v2 --epochs 3`, then `python -m scripts.evaluate --data data/decisions-v2.json --checkpoint runs/reflex-v2/checkpoint --output runs/reflex-v2/evaluation.json`. Run probes with the matching checkpoint/output arguments.

All original and updated results remain in `runs/`. The initial data manifest retains raw-source hashes. The augmentation manifest records the v2 dataset hash and seed. The trained checkpoint binds fitted calibration to its own weight SHA-256. Seven unit tests cover routing parity, permutation behavior, gradient flow, proper scoring, authorization gates and leakage checks; GPU evaluation and the API smoke provide integration coverage.

## Remaining limits

Unknown-language/unknown-domain confidence can be wrong. A high-confidence model answer is not permission. Approval gates require trusted external authorization metadata; this model performs no actions. New schemas abstain until separately calibrated, but matching a schema does not detect distribution shift. Temperature scaling on easy synthetic data can select the grid floor and yield extreme probabilities; do not generalize those scores beyond the controls.

No screenshot grounding, autonomous computer-use success rate, external Jev/Laya/SemIf head-to-head run, distributed training, quantization or multi-seed uncertainty study was completed. The research and implementation are usable; production-level universal capability remains unestablished.
