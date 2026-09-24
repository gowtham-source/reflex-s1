# System 1 decision models: source review and design

Research date: 2026-09-22. This report separates source claims, inspected implementation, our design, and measurements. Repository snapshots inspected: Open-Jev `ed45657bf726c3b77408942830e5578f99df904e`; Laya `c7527708f9f5220c669d8aa385077cd28d04708a`; SemIf `1f2dea3e25379f9dfc98cb83c324f00ab5deda37`.

## What Jev establishes—and what remains unknown

TypeSafe describes Jev as a non-generative interface taking state and typed questions, producing distributions over predefined outputs in parallel. Its launch names a new architecture, sampler, and Reinforcement Learning for Calibrated Decisions (RLCD), but does not publish enough algorithmic detail to reproduce these. The advertised speed comparison includes differing output workloads and service paths. Its workflow references are other models' probabilities, not necessarily human ground truth. We should reproduce the useful contract without asserting access to proprietary internals. Type validity prevents malformed output, not wrong decisions. [Primary announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

Consequently, this project uses the term **open calibration-oriented decision training**, inspired by the stated aim of RLCD. It does not present ordinary cross entropy as a recovered proprietary reinforcement-learning algorithm. Its actual RL component is a sampled computation policy with an explicitly defined reward.

## Alternatives inspected

| System | Mechanism | Strength | Important limit for our design |
|---|---|---|---|
| Open-Jev | Adapted Qwen backbone, scalar scores per candidate, normalized probabilities | Flexible natural-language candidates; mature provenance/evaluation machinery | Repeated candidate processing and large backbones can dominate latency |
| Laya | Bidirectional encoder, option-marker representations, transformer decision head | Compact non-generative typed predictions | Option descriptions compete for a bounded token budget |
| SemIf | Frozen model, restricted native answer logits; optional prefix reuse | Avoids answer decoding and extra training | Prompt/answer-token sensitivity and conditional probabilities require validation |
| Reflex-S1 | Shared state encoder, separately encoded candidates, recursive cross-attention, sparse experts | Candidate-count-independent state encoding; independently bounded option descriptions | Small pretrained encoder limits reasoning, multilingual breadth, and visual understanding |

**Open-Jev.** The inspected trainer combines cross entropy and a Brier term and fits temperature on a calibration split. Released artifacts include adapters, a decision head and temperature, not an independent pretrained foundation model. Its implementation rejects overlength inputs. Its prefix-cache report records probability parity failures even when argmax stays unchanged. We adopt explicit bounded inputs and probability-level regression tests. We do not inherit its reported accuracy. A later stage explicitly imports its public silent-failure corpus, as audited below. [Repository](https://github.com/Zefan-Cai/Open-Jev), inspected `jev/model.py`, `jev/train.py`, `docs/inference-latency.md`.

**Laya.** `common.py` gathers option-marker hidden states, scores them, and computes action-policy features. Its proper reward combines log, spherical and ordinal scores. Its README distinguishes temperature-fitted calibration from raw confidence and reports multilingual/high-cardinality failure modes. These motivate separate schema calibration, a 151-way intent test, and option embeddings that do not squeeze all labels into one sequence. Our own architecture does not incorporate Laya code or weights. A separate pinned Laya installation is used only as an actual competitor baseline. [Repository](https://github.com/NandhaKishorM/laya), inspected `laya/common.py`, `laya/agent.py`.

**SemIf.** Direct restricted-vocabulary logits remove decoding while retaining the frozen language model. The project compares this against candidate reranking and tests reordered options. Its reuse variants change some probabilities and decisions. Its README explicitly distinguishes model agreement from ground truth and hardware comparisons from architectural gains. We adopt those distinctions in our test plan. [Repository](https://github.com/TheoLeeCJ/SemIf), inspected `docs/METHOD.md`, `docs/CALIBRATION.md`. The user-supplied [AI/TLDR page](https://ai-tldr.dev/tools/semif/) was reviewed as a discovery summary; technical conclusions use the repository.

## MoR and MoE are different axes

The MoR paper combines recursive weight sharing with learned token-level computation depth. It discusses expert-choice/token-choice routing and recursion-aware KV caching/sharing. Authors have several affiliations; calling it solely a DeepMind model obscures the primary collaboration. Our decision-level recursion borrows the weight-sharing/adaptive-depth principle, **not** the full token-routing or decoder KV-cache implementation. [MoR paper](https://arxiv.org/abs/2507.10524), [official implementation](https://github.com/raymin0223/mixture_of_recursions).

MoE selects different parameter subsets for different representations. Switch Transformers provides an established sparse-routing reference, but its large-scale speedups cannot be extrapolated to a small batch-one model. Our four-expert layer uses top-2 routing, a load-balancing auxiliary loss, and router z regularization. PyTorch dispatch overhead can erase arithmetic savings; measured latency and a dense baseline must decide whether this is useful. [Switch Transformers](https://arxiv.org/abs/2101.03961).

## Proposed architecture: Reflex-S1

1. A pretrained six-layer MiniLM encoder reads state once. The initialized encoder is from [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2); exact revision is in the data manifest. Full fine-tuning updates it.
2. The same encoder embeds `question [SEP] candidate description`. These vectors are cached only in inference, under exact schema hashes. Options have separate sequences, avoiding a shared label-token budget.
3. Project state tokens and candidate vectors to 128 dimensions. Initialize each candidate query with its option embedding, pooled-state projection, and decision-type embedding.
4. Reuse one cross-attention + sparse MoE block for 1–3 iterations. Queries attend to state tokens; candidates do not attend to each other. The design is equivariant to candidate permutation at a fixed depth; the depth policy pools candidates symmetrically.
5. A state/schema-dependent categorical policy selects recursion depth per question. Inference physically removes completed batch rows from later blocks. This is question-level MoR, not token-level MoR throughout MiniLM. Every state still traverses the pretrained encoder.
6. A scalar head plus a learned cosine-similarity term scores candidates. Softmax yields a categorical distribution. Choice returns a supplied ID; Noul returns P(true); Score returns the expected ordinal index. Multi-question inference reuses state encoding but processes question heads sequentially in this prototype.

The proposed contribution is this *combination and decision-level placement*, not a proven new fundamental algorithm. No exhaustive novelty search, publication priority claim, or superiority claim is made. Possible extensions include fused expert dispatch, batched heterogeneous question heads, vision tokens, policy-conditioned routing and real feedback trajectories.

## Training objective and its limits

For target distribution y and prediction p at recursion depth d:

`L_d = CE(y,p_d) + 0.25 ||p_d-y||² + 0.25 RPS(y,p_d)`

The RPS term applies only to ordinal scores, normalized by K−1. For soft labels, the squared-distance Brier formulation differs from expected one-hot Brier by a constant independent of p; optimization is unchanged. This is supervised proper-score fitting and can learn soft teacher distributions.

The depth policy samples d and receives `r_d = -stop_gradient(L_d) - 0.02*d`. REINFORCE uses the detached policy-weighted reward over all depths as a baseline, plus entropy regularization. All recursion outputs receive supervision; the RL objective trains the compute policy. This is contextual-bandit learning over computational actions, **not** reward learning from live computer-use trajectories. The first epoch warms up prediction heads without policy RL.

The full loss adds MoE load balancing and router z-loss. A separate calibration split fits a positive scalar temperature for each exact schema. Temperature improves a measured distribution; it is not a guarantee under shift. [Calibration reference](https://arxiv.org/abs/1706.04599).

Compute cost can push all decisions to depth one. If this happens, it is reported as policy collapse/preference, not celebrated as successful adaptive reasoning. Deeper computation must earn its cost on harder, independently labeled tasks.

## Data and evaluation protocol

BANKING77 supplies 77 banking intents. CLINC supplies 150 assistant intents plus an out-of-scope class. Pinned original files and SHA-256 hashes are retained. Public training data supplies training/validation/calibration; public test data is kept out of fitting. Exact normalized duplicate texts are excluded across splits. The resulting BANKING test count is therefore slightly smaller than the official split; results are not directly identical to leaderboard protocols. CLINC is imbalanced because out-of-scope data are included. [BANKING77 source](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data), [CLINC source](https://github.com/clinc/oos-eval).

Original approval, retry, risk, tool-selection and textual-DOM tasks provide deterministic controls with disjoint rendered examples. Their finite rules recur across splits; identifiers and wrapper templates vary. They are **not** evidence of broad real-world generalization. No screenshots, browser rollouts, hidden website state, or external actions are involved.

Model selection uses equal-task validation loss. Calibration uses only its own split. Test reports accuracy, Wilson intervals, macro-F1, NLL, Brier, 15-bin ECE, coverage and errors at 0.9 confidence. Repeated test-informed tuning would invalidate a confirmatory interpretation. Fixed-depth ablations use the same trained weights, so they isolate inference depth but do not prove the benefit of recursive training. A separately trained dense baseline addresses a different comparison.

Latency reports p50/p95/p99 over 100 warm requests after 10 warmups with CUDA synchronization, including tokenization/transfers/CPU readout, batch one, and cached schema embeddings. These measurements exclude HTTP, queueing and load time; cold-schema and multi-question probes are reported separately. No direct speed/quality comparison to Jev, Laya or SemIf is established without matched prompts, labels and hardware.

## Deployment interpretation

Supported now: bounded English text/JSON decision input, dynamic candidates, typed outputs, local API, full/head-only future fine-tuning, abstention when a schema has no fitted temperature or confidence is below threshold. Calibration availability is metadata, not a domain-shift detector.

Approval predictions are advisory: a separate gate requires trusted explicit authorization, reversibility and a non-forbidden operation. Untrusted state cannot supply those trusted flags. This model does not execute anything. New schemas default to abstention; an operation-specific validation set is needed to establish useful thresholds.

Not established: frontier reasoning, robust prompt-injection defense, screenshot grounding, end-to-end browser success, multilingual competence, high-confidence correctness under shift, hard real-time guarantees, or optimal accuracy/latency. A model cannot be trained “perfectly”; these require representative labels, controlled comparisons, and measured acceptance thresholds.

## Added source: public Open-Jev dataset

The user additionally supplied the [Hugging Face Open-Jev dataset](https://huggingface.co/datasets/ZefanCai/Open-Jev). Its pinned schema, overlap and split audit, actual selected training corpus, and reproducible adapter are documented in [OPENJEV_DATASET.md](OPENJEV_DATASET.md). This adds an independently published synthetic agent-reliability task to the implementation, without confusing its target labels with measured probabilities from our model.

## Competitor-driven generalization iteration

An actual pinned Laya checkpoint was run on matched requests on the same L40S. The small specialized model was faster and stronger on its trained task families, but weaker on SemIf's separate authored144 suite. That result motivates the experimental `condition_question` mode: state and question are encoded jointly, followed by the existing recursive sparse-expert candidate scorer. State encoding is then reused across candidates **within a question**, not across different questions. This trades some multi-question throughput for earlier state/question interaction.

The broader curriculum adds SNLI premise/hypothesis classification with per-row questions. This requires a dynamic-question training path; the previous fixed-schema-only trainer was insufficient. The corpus is sampled by image/caption group, and image groups that cross the selected official partitions are removed. Exact selection counts and archive hash are saved. SNLI is a human-annotated textual entailment resource, not an agent trajectory dataset. [SNLI primary source](https://nlp.stanford.edu/projects/snli/).

A separate CLINC operating-point procedure selects an OOS logit offset on validation, with an in-scope accuracy constraint; it then fits temperature on calibration. This changes decisions and is explicitly reported, not hidden as scalar temperature calibration. Its output does not establish universal unknown-intent detection. The English-only runtime also abstains on several clearly unsupported scripts; that deterministic guard does not detect English domain shift.

## Larger encoder experiment

A bounded additional run substitutes `cross-encoder/nli-MiniLM2-L6-H768`, revision `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`, for the original sentence encoder. Its model card identifies a six-layer, 768-wide MiniLMv2 distilled from RoBERTa-Large and trained on SNLI/MultiNLI. The native classification head is discarded; the shared encoder initializes our own MoR+MoE decision model. The tokenizer's native paired-input formatting is used for state/question conditioning. This is transfer learning, not a from-scratch foundation model. [Primary model card](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768).

Its existing NLI training means comparisons cannot treat NLI as a completely unseen pretraining domain. The larger run uses the same local task mixture and distinct validation/calibration/test roles; it is an engineering alternative with different upstream training and parameter count, not a pure scale-only ablation.

Broader paper review and fresh external tool/evidence evaluation: [BROAD_BENCHMARKS.md](BROAD_BENCHMARKS.md).
