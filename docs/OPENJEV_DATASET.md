# Open-Jev dataset integration

Source: [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev), pinned revision `c67699e13d0ae25e35b77165a4b6b079bedc8aba`.

The dataset publishes 12 configurations with official train/calibration/validation/test/OOD splits. The composite configurations overlap, so their totals must not be added as unique training examples. Targets include distributions; metadata can contain privileged control labels. Original generated records have CC0 provenance, with upstream exceptions retained in third-party notices. These are mostly synthetic controls, not TypeSafe's proprietary training data. Raw JSONL preserves original records and avoids needing a Parquet dependency. [Pinned dataset card](https://huggingface.co/datasets/ZefanCai/Open-Jev/blob/c67699e13d0ae25e35b77165a4b6b079bedc8aba/README.md).

## What this project actually uses

We downloaded **silent-failure-control-v1** using all five official splits. It is a practical agent reliability task: decide whether an API response body demonstrates that an operation failed to deliver its required result. Its official source counts are 6,432 train, 360 validation, 264 calibration, 624 test, and 1,920 OOD. Its OOD layout/language shift is substantially harder for our English encoder. It does not prove detection of arbitrary production API failures.

The adapter in `scripts/import_openjev.py`:

- Pins every download and records compressed-file SHA-256 hashes.
- Uses only the decoded state, question, kind and option descriptions as model inputs. Metadata, target vectors, original IDs and split labels are excluded from inputs.
- Preserves source `group_id` boundaries across all five splits and rejects cross-split duplicate inputs.
- Keeps soft target distributions intact. For Choice, sorts options and permutes targets identically; Noul false/true order and ordinal option order are preserved.
- Rejects unsupported noncategorical targets and non-rank score scales rather than silently changing their semantics.
- Rejects sequences over the model's 256-token bound and records the counts. No hidden truncation.
- Keeps OOD in a separate file, never in training or calibration.
- Fits the current fixed-schema trainer only to schemas represented in all four non-OOD splits. Dynamic one-off question/candidate schemas that cannot satisfy this are counted as unsupported; this is a trainer limitation, not evidence those source rows are defective.

All 6,432 train, 360 validation, 264 calibration and 624 test examples were retained. The OOD conversion retained 1,838 and rejected 82 overlength examples. The OOD report includes **full-denominator accuracy with those rejected examples counted as wrong**, in addition to accuracy among scored examples.

The new training run, `runs/reflex-openjev`, warm-starts the prior MoR+MoE checkpoint and includes replay of the original intent and control tasks. It is a continued specialization run, not an independent model trained on all Open-Jev configurations. Only this one configuration was used for optimization. The release and browser/drone configurations were researched but not incorporated into the trained mixture.

## Reproduce

```bash
python -m scripts.import_openjev
python -m scripts.train --data data/decisions-openjev.json \
  --resume runs/reflex-v2/checkpoint --output runs/reflex-openjev --epochs 3
python -m scripts.evaluate --data data/decisions-openjev.json \
  --checkpoint runs/reflex-openjev/checkpoint --output runs/reflex-openjev/evaluation.json
python -m scripts.evaluate_ood
```

Use `data/openjev-import.json` for the exact source URLs, hashes, retained/rejected counts and selected schemas. `data/openjev/` retains raw compressed splits, the source README, LICENSE-DATA and THIRD_PARTY_NOTICES. The importer also supports selecting other published configurations; review its rejection audit and source-specific semantics before training. Do not combine the two overlapping composite configurations without a cross-config deduplication audit.

The final test/OOD metrics and actual expert/depth utilization are in the run's JSON artifacts. No metrics in the source dataset card are inherited as performance claims for this model.

The later broader question-conditioned variant raises its input bound to 512 tokens, but its OOD evaluation retains the previously frozen 1,838-row projection so the evaluated population does not change mid-experiment. The 82 original exclusions still count as errors in the full-denominator score. A future full-length OOD reevaluation must be reported as a different protocol rather than silently replacing this one.
