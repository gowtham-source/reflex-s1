# Final measured results: Reflex-S1 + Open-Jev

Final checkpoint: `runs/reflex-openjev/checkpoint`. The model was actually trained and evaluated on the local L40S. It is a compact research model, not a validated universal agent or autonomous approval authority. The earlier matched dense comparison and failed approval iterations remain in [RESULTS_V2.md](RESULTS_V2.md).

## Final quality and latency

| Task | Test examples | Accuracy | ECE15 | Accepted at ≥0.9 | Accuracy among accepted | Warm p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| banking77 | 3073 | 91.93% | 0.0119 | 79.37% | 98.36% | 6.52 / 6.73 |
| clinc150 | 5500 | 85.51% | 0.0453 | 78.96% | 96.43% | 6.78 / 6.86 |
| approval | 600 | 45.33% | 0.5440 | 98.50% | 45.69% | 6.76 / 6.95 |
| tool_route | 360 | 96.11% | 0.0213 | 88.89% | 99.38% | 6.77 / 6.86 |
| dom_action | 360 | 100.00% | 0.0000 | 100.00% | 100.00% | 6.72 / 6.90 |
| retry | 360 | 100.00% | 0.0000 | 100.00% | 100.00% | 6.62 / 6.77 |
| risk | 360 | 100.00% | 0.0000 | 100.00% | 100.00% | 6.65 / 6.88 |
| Open-Jev silent failure | 624 | 97.44% | 0.0131 | 93.27% | 99.14% | 6.79 / 6.97 |

CLINC in-scope accuracy is 95.27%; OOS recall is 41.60%. Both matter: the combined score includes 1,000 OOS cases, and confidence gating does not reliably identify all unfamiliar requests.

BANKING77 and CLINC are public natural-language intent datasets. All remaining tasks are synthetic controls, including the externally published Open-Jev corpus. Their accuracy is not an end-to-end agent/browser success rate. Approval performance on newly phrased evidence is weak; automatic approval is not qualified. Even a correct predicted label does not grant permission.

## Open-Jev integration and distribution shift

Pinned dataset revision: `c67699e13d0ae25e35b77165a4b6b079bedc8aba`. Actual training subset: `silent-failure-control-v1`. Retained official splits: `{'train': 6432, 'validation': 360, 'calibration': 264, 'test': 624, 'ood': 1838}`. All source groups and split labels were preserved, with no metadata/target leakage into inputs.

Official OOD: 1920 examples, 1838 scored, 82 overlength and explicitly rejected. **Full-denominator OOD accuracy with rejections counted as wrong: 69.53%.** OOD probabilities use training-domain calibration; there is no OOD temperature fitting.

See [OPENJEV_DATASET.md](OPENJEV_DATASET.md) for the audit and importer. The main test and OOD figures here are from the dedicated specialization, not copied from Open-Jev model reports. Only one of its configurations was used for training.

## Architectural evidence

Parameters: 23,242,121. Four experts, top-2 dispatch, shared recursion block, maximum depth three. Measured expert fractions: [0.2333919107913971, 0.28832337260246277, 0.24817194044589996, 0.23011277616024017].

Mean chosen depth: banking77=1.000, clinc150=1.000, approval=1.000, tool_route=1.000, dom_action=1.000, retry=1.000, risk=1.000, openjev_1cb36753666fb3d8=1.000.

All earlier validation sets favored depth one. The learned policy and computation skipping are implemented, but a useful adaptive-depth advantage has not been demonstrated. A separately trained dense baseline was faster and won on some controls; MoR+MoE v1 improved the two public intent scores. This is mixed single-seed evidence, not proof of architectural superiority. The final model received additional training, so comparing it directly to that five-epoch dense baseline would confound architecture, data and compute.

The probability heads optimize cross entropy, Brier and ordinal scores. The actual RL component is a contextual-bandit depth policy rewarded for proper-score quality minus compute cost. This is openly specified RLCD-inspired work, not a reproduction of Jev’s unpublished algorithm.

## Measurement scope

Warm request timings: batch one, 100 measurements per task, 10 warmups, cached schema embeddings, synchronized CUDA. Includes tokenization, device transfers and CPU typed result construction. Excludes model loading, HTTP/network, concurrency and queueing. Requests contain at most 256 tokens per state/candidate. Cold-schema probes keep model weights resident.

| Probe | p50 ms | p95 ms |
|---|---:|---:|
| warm_route | 6.50 | 6.63 |
| cold_route | 11.28 | 11.40 |
| warm_151_options | 6.74 | 6.76 |
| cold_151_options | 14.66 | 14.86 |
| five_questions | 15.77 | 15.83 |

Authored stress probes: 10/10. These were inspected during development, so they are diagnostics. ASGI API smoke status: 200; invalid request status: 422.

## Training and verification

Open-Jev stage: three epochs, 4230 optimizer updates, 223.2 seconds, peak PyTorch allocated memory 1182.4 MiB. The model warm-started the validation-selected v2 checkpoint. V1, dense, v2 and Open-Jev checkpoints and history remain available.

Validation selects weights; separate calibration fits temperature. Test and OOD are not gradient inputs. Earlier public test results were inspected during development; repeated reporting is disclosed. Tests include option permutation, fixed/adaptive parity, sparse dispatch reference parity, nonzero gradients, proper soft-target scoring, trusted approval gating, split/group leakage and Open-Jev target remapping/metadata exclusion.

Artifacts: `evaluation.json`, `evaluation.predictions.jsonl`, `openjev-ood.json`, `probes.json`, `training.json`, `selection.json`, `history.json`, fitted checkpoint calibration and exact environment versions. Raw latency samples are retained. The API exposes typed predictions and never executes actions.

## Remaining work before broad deployment

Representative human-reviewed authorization data; real computer-use trajectories and screenshot grounding; stronger OOS/multilingual evaluation; many more natural-language decision schemas; fused sparse kernels and concurrent serving tests; separate MoE/MoR/RL ablations and multiple seeds. Full/head-only fine-tuning works now; LoRA and distributed training are not implemented.
