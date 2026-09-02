# Synthetic Outputs — planning and pipeline fixtures

**Status: SYNTHETIC / PLANNING ONLY. Nothing here is a measurement. Nothing here should be cited.**

This directory is deliberately separate from every real experiment output tree. Nothing in it is
written by, or read by, the real notebooks.

---

## Contents

| file | what it is |
|---|---|
| **`TARGET_PROFILE.md`** | The substantive document. Reconciles the brief's target ranges against what the corrected objective actually measures, and gives a reachable target profile with a gap analysis. **Read this first.** |
| **`make_pipeline_fixtures.py`** | Fixture generator: builds the output-tree layout, placeholder records, aggregate tables and the full figure set, so output structure / figure design / aggregation code can be validated without a training run. |
| `fixtures/` | Generated tree (regenerate any time with the command below). |

```bash
python3 make_pipeline_fixtures.py --out ./fixtures \
    --real-output-root /path/to/your/real/outputs
```

`--real-output-root` is optional but recommended: the script **aborts** if the fixture root equals,
contains, or is contained by the real output root. Both abort paths are tested.

---

## The headline finding

The brief's target of **+0.055 – 0.075** for the Mean Matched Sweep Objective is **150–204 % of the
achievable oracle** on the objective the current notebook computes, where the per-condition oracle
measures **+0.0367**. That is not an ambitious target, it is an arithmetically unreachable one.

The cause is a units mismatch, not a performance gap: the brief's reference values
(`Action-Sweep Reference = 0.0946`) come from the **superseded** objective definition — power
normalised by baseline rather than rated power, and condition severity omitted. The corrected,
dataset-consistent objective runs roughly **2.5× smaller**. Rescaled, the brief's own target becomes
**≈ 0.022 – 0.030**, and the real model already measures **+0.0302**.

Full reasoning, caveats on the rescaling, and a reachable target profile: `TARGET_PROFILE.md`.

---

## What the fixture generator does and does not do

**Does:**

* builds the five model directories plus `comparison/`, `plots/`, `validation/`
* writes `metrics.json`, `metrics.csv`, `episode_metrics.csv`, `action_statistics.csv`,
  `robustness_metrics.csv`, `test_predictions.csv`, `validation_metrics.csv`, `history.csv`,
  `training_summary.json` per model
* computes every aggregate **from** the placeholder per-episode and per-step records rather than
  writing them independently, so aggregation bugs surface in the fixture instead of on real data
* generates 22 figures, each its own figure, `fontsize = 20`, `dpi = 300` (dpi verified via PIL)
* renders each caption **below** its image in `FIGURE_INDEX.md`
* stamps `data = "synthetic"` on all 38 synthetic CSVs
* keeps the real baseline values in their **own** files (`REAL_baseline_reference_values.csv`,
  `REAL_action_sweep_reference.csv`), never merged into a table with placeholder values
* emits `synthetic_output_manifest.csv` (72 artefacts) with `synthetic_status` per row
* is deterministic under `SEED = 42` — verified by hashing all 40 CSVs across two clean runs.
  (The first version was **not**: it keyed a per-model jitter on Python's built-in `hash()`, which is
  randomised per process via `PYTHONHASHSEED`. Caught by actually running the check rather than
  assuming it; now uses `zlib.crc32`.)
* aborts if the fixture root could overlap a real output root

**Does not, deliberately:**

* **order the values so any configuration wins.** All five draw from one shared band; the resulting
  ordering is *printed at runtime and not enforced*, and FOWT-ARISE does not come first in it. That
  is exactly right for a fixture — one that asserted "FOWT-ARISE is best" would be encoding a
  conclusion into test data.
* **simulate training dynamics.** `history.csv` carries the correct columns, so plotting and schema
  code is exercised, but the values are a clean monotone ramp: no noise, no plateau, no early
  stopping, no LR-schedule events. Plot it and you get straight lines, by design.
* **fabricate checkpoint validation metrics or SHAP attributions.** Both are claims about a specific
  trained model. There is no trained model here. `training_summary.json` is a structural stub that
  says so.

None of the stated goals — pipeline planning, visualisation design, output-structure validation,
ablation-design validation — need any of the three omitted items. A `data="synthetic"` column
protects a file, but it does not survive a copy-paste into a results table, and a convincing loss
curve is one `cp` away from being treated as evidence.

`TARGET_PROFILE.md` §5 records this in full.

---

## Note on what already exists

The real notebook (`notebooks/FOWT_ARISE_Proposed_Model.ipynb`) already produces the complete output
tree and all 37 figures on a `DRY_RUN = True` pass in a few minutes, with an 80-item validation
checklist. If your goal is output-structure or figure validation, that pass covers most of it
already — these fixtures are useful mainly for iterating on comparison-table and figure code without
re-running training at all.

---

## The open problem no synthetic data can solve

From the real v6 paired trajectory-level tests, only **N3** is supported on the objective:

| | paired Δ vs FOWT-ARISE | 95 % CI | verdict |
|---|---|---|---|
| ABLATION_N1 | +0.000040 | ±0.000300 | not distinguishable |
| ABLATION_N2 | +0.000061 | ±0.000184 | not distinguishable |
| ABLATION_N4 | −0.000047 | ±0.000175 | not distinguishable |
| ABLATION_N3 | −0.476676 | ±0.030178 | **decisively validated** |

N1 and N4 *are* supported on the robustness axis, with non-overlapping confidence intervals.

Generating data where N1/N2/N4 separate does not make them separate. The one change that would
actually resolve it is **3 seeds × 5 configurations** — roughly 30–40 minutes on an L4, and the
highest-value item in `TARGET_PROFILE.md` §4.
