# FOWT-ARISE — Proposed Model Notebook

**FOWT-ARISE: A Physics-Informed Adaptive Reinforcement Learning Framework for Real-Time Structural
Load Relief in Floating Offshore Wind Turbines Under Imperfect IoT Observations**

Deliverable: **`FOWT_ARISE_Proposed_Model.ipynb`** — a single, self-contained, top-to-bottom
executable research notebook.

---

## What the notebook implements

It turns the physics-informed FLOATBench fatigue benchmark into a sequential, IoT-aware decision
environment in which **one** RL policy learns what structural load is controllable, which actuator to
use, how aggressively to act, and when **not** to act — while balancing fatigue reduction against
power and actuation cost.

Concretely, it:

1. loads and audits the transitions + action-sweep datasets, resolving every column by exact-match
   against candidate lists (never substring heuristics, never an invented column);
2. builds trajectory-level train/val/test splits keyed on `(tower, episode_id)`;
3. constructs the physics-informed state, the joint 3-D action space, and the multi-objective reward;
4. trains FOWT-ARISE and **four ablations** with a single shared training function;
5. evaluates each on the test split **once**, via counterfactual nearest-action matching against the
   sweep grid, under clean **and** IoT-degraded observations;
6. runs paired significance tests, a checkpoint-noise analysis, SHAP explanations, ~60 figures, a
   validation checklist, and a final summary.

### Base algorithm, kept separate from the novelties

An **offline actor–critic** in the TD3+BC / CQL family: twin critics with their own encoder, target
networks, target-policy smoothing, delayed policy updates, a conservative CQL penalty, and a
behaviour-support regulariser. N1–N4 sit *on top* of this and are toggled independently.

---

## The four novelties

| | Novelty | Implementation | Ablated by |
|---|---|---|---|
| **N1** | Physics-informed load-aware state | Grouped encoder over four physically distinct blocks; the state carries load-path decomposition, governing-section fatigue, controllable-load share, and aerodynamic control-authority gains | `use_n1=False` → flat MLP over the conventional state; the physics block is **absent from the input**, not zeroed |
| **N2** | Adaptive multi-actuator control | **One** joint head for pitch/yaw/IPC, gated by a learned differentiable control-authority gate with an explicit *expected-net-benefit* head | `use_gate=False` → gate and benefit head removed; the head stays **joint over all three actuators** |
| **N3** | Fatigue–power–actuation multi-objective reward | `λ_fat·f·s − λ_pow·p − λ_act·u − λ_sm·σ`, all terms dimensionless and normalised | reward and target-selection reward reduce to the load-relief term alone |
| **N4** | IoT-degradation-aware robust RL | Gaussian noise / dropout / persistent bias / stale observations, mixed clean-degraded training, plus a clean-vs-degraded **consistency loss** | trains clean only, `λ_robust = 0`; still **evaluated** on clean *and* degraded |

The no-action behaviour is **learned**. There is no `if random: action = 0` and no fixed threshold
anywhere in the notebook.

---

## Expected input datasets

```
DATASET_PATH/
    transitions_ref.parquet
    transitions_opt1.parquet
    transitions_opt2.parquet      # 129,600 rows x 89 cols total, 3 towers
ACTION_SWEEP_PATH/
    action_sweep_ref.parquet
    action_sweep_opt1.parquet
    action_sweep_opt2.parquet     # 1,455,300 rows x 27 cols total
```

### Schema note — please read

The project brief documents seven structural columns (`damage_max`, `damage_base_section`,
`damage_top_section`, `damage_mean_section`, `damage_max_section_id`, `damage_ratio_max`,
`controllable_share_max`) and four environmental columns (`mean_wind_speed`, `std_wind_speed`,
`wave_hs`, `wave_tp`) as belonging to the **transitions** dataset. Verified against the real files,
**all eleven exist only in the action-sweep dataset.** The transitions use a different, richer family:

| role | transitions | action sweep |
|---|---|---|
| governing-section damage, no control | `damage_baseline` | `damage_max` |
| load-path base / top section | `damage_baseline_base_section` / `..._top_section` | `damage_base_section` / `damage_top_section` |
| controllable load share | `controllable_share_mean`, `controllable_share_max_section` | `controllable_share_max` |
| environment | `true_*` / `meas_*` / `valid_*` triples | `mean_wind_speed`, `std_wind_speed`, `wave_hs`, `wave_tp` |
| yaw action | `action_yaw_setpoint_deg` | `action_yaw_error_deg` |

The notebook's resolver **accepts either family**, records which name it used in
`common/schema_mapping.json`, prints a **schema discrepancy report** naming every documented column
that turned out to live elsewhere, and raises with the candidate list if a required role cannot be
resolved at all. Nothing is invented or silently renamed.

The brief's "1,200 episodes" is the **per-tower** count. ×3 towers = **3,600 trajectories**, which is
what the notebook splits on — exactly the trap the brief's §14 warns about.

### Action-outcome leakage

`damage_ratio`, `del_ratio`, `damage_controlled*`, `ct_ratio` and `cp_ratio` are **consequences of the
action that was taken**. Observing them would tell the policy how well its own action worked before it
chose it, so they are classified as action-outcome and excluded from the state by an explicit
assertion. Only condition-only quantities (`damage_baseline*`, `controllable_share_*`,
`damage_weight`, aerodynamic gains) enter N1's physics block.

---

## Configuration

Edit **only** the block at the top of Section 02:

```python
DATASET_PATH           = "CHANGE_THIS"
ACTION_SWEEP_PATH      = "CHANGE_THIS"
OUTPUT_ROOT            = "CHANGE_THIS"
RESUME_FROM_CHECKPOINT = ""     # "" = fresh; else a path to a checkpoint .pt
BASELINE_DIR           = ""     # "" = skip baseline comparison
DRY_RUN                = True   # True => tiny run exercising every code path
```

Everything else — column names, action limits, normalisation, architecture, reward weights, output
paths, seeds, evaluation functions, plotting — is derived automatically. Output directories are
created before anything is written.

Two defaults worth knowing about, both documented in-notebook:

* **`REWARD_WEIGHT_PRESET = "dataset_consistent"`** uses λ = (2.0, 1.0, 0.05, 0.02), which
  *reproduces the dataset's own `reward` column to floating-point precision* (enforced by an assertion).
  That keeps every number comparable with the dataset and with baselines scored against it. Switching
  to `"spec_default"` uses the brief's illustrative λ = (1.0, 0.5, 0.25, 0.1); the check then
  downgrades to a warning and cross-comparability is explicitly forfeited.
* **`Q_IMPROVEMENT_COEF = 0.0`** — the actor is driven by best-action imitation rather than by
  `dQ/da`. Section 25 *measures* whether the critic can rank actions (Spearman ρ, top-1 agreement,
  regret vs decision range) so you can disagree with this default from evidence rather than taking it
  on trust. The critics are still built, trained, conservatively regularised and logged.

---

## How to run

### 1. Dry run (do this first)

```python
DRY_RUN = True
```

Run all cells. Uses `NUM_EPOCHS_DRY = 2` and reduced SHAP sizes, and validates loading, preprocessing,
splitting, model construction, forward/backward passes, checkpointing, resume, evaluation,
counterfactual matching, robustness, all four ablations, plots, SHAP, and every output file.

Verified on the real dataset: **executes clean, checklist 80 PASS / 0 WARNING / 0 FAIL.**

### 2. Full experiment

```python
DRY_RUN = False      # -> NUM_EPOCHS_FULL = 60 with early stopping
```

Trains and evaluates all five experiments sequentially, freeing GPU memory between them.

### 3. Resume from a checkpoint

Every experiment auto-resumes from its own `checkpoints/latest.pt` if present, so simply re-running
after an interruption continues rather than restarting. To resume FOWT-ARISE from a specific file:

```python
RESUME_FROM_CHECKPOINT = "/path/to/outputs/FOWT_ARISE/checkpoints/latest.pt"
```

You will see:

```
[RESUME] Resuming from checkpoint: .../latest.pt
[RESUME] Starting epoch: 3
[RESUME] best matched_sweep_objective so far = +0.027365 (epoch 1), patience counter = 1, 2 history rows preserved
```

Model, both optimisers, both schedulers, epoch, best-validation state, early-stopping counter, feature
scaler and RNG states are all restored, and `history.csv` is **appended to**, never truncated. An
explicitly supplied `RESUME_FROM_CHECKPOINT` that does not exist is a **hard error** — silently
starting a fresh run would discard the experiment you meant to continue. A checkpoint whose
architecture flags do not match the target experiment also raises.

*Verified: resuming a 2-epoch run with a 4-epoch budget starts at epoch 3 and produces
`history.csv` with epochs `[1, 2, 3, 4]`.*

---

## Output structure

```
OUTPUT_ROOT/
├── FOWT_ARISE/
│   ├── checkpoints/           best.pt, latest.pt
│   ├── plots/                 10 individual training figures
│   ├── shap/                  importance csv, values npy, feature names json, metadata json,
│   │                          summary + beeswarm + per-actuator beeswarms
│   ├── history.csv            38 epoch-wise metric columns
│   ├── config.json            the exact configuration this experiment ran under
│   ├── metrics.json / .csv
│   ├── test_predictions.csv         per-transition action + matched counterfactual outcome
│   ├── validation_predictions.csv
│   ├── episode_metrics.csv          per-trajectory audit trail behind every aggregate
│   ├── iot_robustness.csv           per degradation mode
│   ├── critic_ranking_diagnostic.json
│   ├── state_feature_manifest.json  source / derivation / normalisation per feature
│   └── training_summary.json        parameter counts, best epoch, early-stopping reason
├── ABLATION_N1/ … ABLATION_N4/      same layout
├── comparison/
│   ├── final_ablation_comparison.csv / .xlsx / .json
│   ├── final_baseline_comparison.csv / .xlsx / .json
│   ├── paired_objective_significance.csv
│   ├── paired_robustness_significance.csv
│   ├── checkpoint_sensitivity.json
│   ├── validation_checklist.json
│   └── comparison_plots/            19 individual comparison figures
└── common/
    ├── environment.json             python/torch/numpy/pandas/CUDA versions
    ├── schema_mapping.json          every resolved role + the discrepancy report
    ├── data_summary.json
    ├── split_summary.json
    ├── state_feature_manifest.json
    ├── reward_component_stats.csv
    └── train/val/test_trajectories.csv
```

All figures are **individual** (no subplot grids), `fontsize = 20`, `dpi = 300`.

---

## Interpreting the final tables

### `final_ablation_comparison.csv`

One row per experiment plus five **reference policies** (`do_nothing`, `ipc_half`, `ipc_only`,
`behaviour_logged`, `ORACLE_best_of_sweep`), because an absolute objective value means nothing without
the achievable band. `% of Oracle` divides by the per-condition best of the sweep grid — the ceiling
any policy restricted to that grid could reach.

**Direction matters and is stated on every axis label:**

* **lower is better** — Mean/Median DEL Ratio, Power Loss %, Actuator Duty Proxy, IoT Performance Gap,
  Robustness Drop %
* **higher is better** — Mean Matched Sweep Objective, Fatigue Relief %, Clean/IoT-Degraded
  Performance, % of Oracle
* **read jointly, not in isolation** — Fatigue Relief % against Power Loss %. A policy can always buy
  more load relief by feathering; the trade-off scatter plot exists for exactly this reason.

### What "Mean Matched Sweep Objective" is, and is not

The action-sweep file has **no native `sweep_objective` column** and none is fabricated. The objective
is derived from verified sweep quantities using the same reward definition as the training reward
(formula in Section 18a), with severity anchored on each condition's zero-action row and power loss
normalised by **rated** power (derived from the data, never hard-coded).

It is a **counterfactual estimate** of what the policy's action *would* have produced at the same
physical condition, obtained by matching to the nearest sweep action. It is **not** a measured
transition reward and is never presented as one. Two honesty measures accompany it:
**Action-Sweep Coverage %** (uncovered rows stay `NaN`, never matched to an unrelated condition) and
**mean match distance** (how far the requested action sat from the nearest grid point).

**Absolute DEL in physical units is unavailable** from action-sweep data, so fatigue is reported as
DEL **ratio**, median DEL ratio, and fatigue relief % — never as a fabricated absolute.

### Section 35 — what the ablations actually support

Do not read the ablation ranking off the table alone. Section 35 prints three analyses:

1. **Paired significance on the objective.** All experiments share the same test trajectories, so
   differences are paired trajectory-by-trajectory with a 95% CI. Far more sensitive than comparing
   independent means.
2. **Checkpoint-selection sensitivity.** `best.pt` is an `argmax` over a noisy validation curve. Each
   comparison is judged against `max(spread of the ablation, spread of FOWT-ARISE)` — not a pooled
   median, which would be dominated by the tight runs and would wave through a noisy one. A
   difference smaller than that floor is printed as **UNRESOLVED**, and unequal epoch budgets from
   early stopping are flagged as a second confound (best-of-40 beats best-of-15 on luck alone).
3. **Paired significance on robustness.** The objective here is dominated by the reward definition
   (N3), which can make a component whose contribution is *decision stability* look inert. The same
   paired test therefore also runs on per-trajectory action shift under degradation.

A component can come out **supported on one axis and contradicted on the other**, and both are
printed. Where an ablation beats the full model, that is reported as measured.

### `final_baseline_comparison.csv`

If `BASELINE_DIR` contains RB-FOWT / CQL / IQL outputs they are **read, never retrained or modified**.
A missing baseline is simply absent, with the message *"Baseline comparison unavailable because
BASELINE_DIR was not supplied"* — never filled in with invented values. A partially available
baseline contributes only the metrics it actually has; the rest stay `NaN`.

---

## Scientific-integrity properties

* **No fabricated metrics.** No value is ever assigned to a desired number; nothing is adjusted after
  calculation; no result table is hard-coded. Every figure in every table comes from model evaluation.
* **No episode leakage.** Splitting is trajectory-level, asserted disjoint at trajectory *and* row
  level, with a frozen test set re-checked by `assert_no_test_leakage()` before every use.
* **No action-outcome leakage.** Asserted explicitly against a list of 14 outcome columns.
* **Test set touched once.** After training, validation, early stopping and checkpoint selection.
  Nothing downstream of the test evaluation feeds back.
* **Scalers and action bounds from TRAIN only.**
* **SHAP background from TRAIN, explained rows from TEST** — the correct split, with provenance
  recorded in `shap_metadata.json`.
* **Fair-ablation audit.** Section 34 *asserts* that seed, split, batch size, budget, learning rates,
  γ, CQL α, gradient clipping, early stopping, LR schedule, action dim, latent dim, rated power and
  Wöhler exponent are byte-identical across all five experiments, **and** that each ablation differs
  from FOWT-ARISE in exactly one intended respect. It raises if not.
* **NaN, not 0, for inapplicable metrics.** Ablation N3's power/duty/smoothness components are
  reported as `NaN`; writing `0` would assert it achieved zero power loss rather than that it did not
  account for power.

---

## Verification performed

* **Static analysis** — AST check across all 49 code cells in execution order: syntax clean; every
  module-level name bound by an earlier statement or cell (immediate scope); every name used inside a
  function or class body bound somewhere in the notebook (deferred scope); all required output tokens
  present. Guards against the `NameError` / undefined-helper class of failure.
* **Execution** — `DRY_RUN = True` against the real dataset: executes clean, all five experiments
  trained and evaluated, SHAP succeeded for **3/3** actuator heads, **19** comparison figures,
  complete output tree, checklist **80 PASS / 0 WARNING / 0 FAIL**.
* **Resume** — verified continuing a 2-epoch run under a 4-epoch budget.

Two real metric bugs were found by that dry run and fixed: `Actuator Duty Proxy` and
`Mean Action Magnitude` were computing the same quantity, and magnitude was measured from `0` rather
than from the neutral action (which scored a do-nothing policy at 0.667 instead of 0.0). Duty now uses
the dataset's own actuator-duty model on the policy's physical action; magnitude is deviation from
neutral. `do_nothing` now correctly reports 0.0 for both.

**Not claimed:** the notebook has not been executed to completion with `DRY_RUN = False` on your
hardware. The 2-epoch dry-run numbers exercise the machinery and are not scientific results — run the
full experiment for those.
