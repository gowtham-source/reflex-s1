# What “competitive” means in this project

A broad win over Jev or Laya cannot be inferred from a small specialized benchmark. We measure separate axes: matched-request correctness, held-out decision generalization, calibrated selective error, model size, and warm/cold latency.

## Actual Laya baseline

- Weights: `convaiinnovations/laya`, revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
- Runtime source: Laya repository revision `c7527708f9f5220c669d8aa385077cd28d04708a`.
- Both models run sequentially on the same L40S; order alternates per example. State, question and option descriptions match; each implementation retains its own prompt/tokenizer.
- Laya receives enlarged option/context budgets to preserve its input rather than being penalized by high-cardinality default truncation. Its code limits individual descriptions; the harness checks this constraint.
- Reflex was specialized on the task training splits. Laya is the public out-of-box checkpoint. This comparison measures those deployable artifacts, not equal-data training or architectural superiority.
- Laya native temperature buckets differ from our fitted per-schema temperatures. Native Laya code clamps an out-of-range high-cardinality temperature. Accuracy is unaffected by that positive scalar; calibration is not directly controlled.

The original comparison is in `runs/comparison/laya.json`, with per-row predictions. In that run, Reflex was faster and better on the task-specific subsets but scored 57/144 on SemIf's generalization suite versus Laya's 89/144. This unfavorable result is why a broader curriculum and question-conditioned variant were developed.

SemIf's authored144 suite is project-authored synthetic/model-reviewed content, not independently human-adjudicated real-world ground truth. Its labels were never added to training, but the aggregate result influenced engineering direction. Later scores therefore measure continued development, not a newly blinded benchmark. Its source revision and MIT license are retained under baselines/research provenance.

## Jev

We researched Jev's public primary materials and Open-Jev's implementation/data. We have not run a live TypeSafe Jev API comparison. Published latency numbers from different clients, prompts, hardware and networks cannot establish a head-to-head win. This project includes no private Jev weights or recovered proprietary RLCD implementation.

## Criteria for further promotion

Require stronger unknown-intent recall, approval performance under paraphrase and adversarial evidence, external generalization, real browser trajectories, multiple seeds, and matched train/tuning budgets. Retain a slow-model/human fallback where the fast model is not qualified. Sparse experts and recursive routing are implementation choices to test, not benefits assumed in advance.

Reported Wilson intervals are row-level descriptive intervals. They do not account for correlated questions from one image, synthetic family or episode. Group bootstrap intervals and repeated training seeds are needed before population-level superiority claims.
