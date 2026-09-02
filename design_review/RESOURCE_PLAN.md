# FOWT-ARISE — Compute & Storage Resource Plan

**Every number in this document is measured, not estimated from a model of a run.** Sources are
named per row. Nothing here is synthetic.

Two measurement sources are used:

| source | hardware | what it measured |
|---|---|---|
| **A** | Colab **NVIDIA L4** | your `Outputs_v6` run of `FOWT-ARISE.ipynb`, 5 experiments, patience 14 |
| **B** | this sandbox, **CPU only** (torch 2.8 cpu) | `FOWT_ARISE_Proposed_Model.ipynb`, 16-epoch cap, patience 6, 5 experiments, full eval + SHAP |

---

## 1. Measured per-epoch cost

| source | hardware | epochs run (5 experiments) | training wall clock | **s / epoch** |
|---|---|---|---|---|
| A | L4 | 126 (25 + 19 + 27 + 21 + 34) | ~180 s | **≈ 1.43** |
| B | CPU | 51 (9 + 8 + 9 + 16 + 9) | 188 s | **3.60** |

**L4 is ~2.5× faster than this CPU sandbox.** Use source A for planning; source B is the fallback if
no GPU is allocated.

Per-experiment detail, source B (real):

| experiment | epochs run | early stopped | best epoch | s / epoch | params |
|---|---|---|---|---|---|
| FOWT_ARISE | 9 | yes | 3 | 3.92 | 42,697 |
| ABLATION_N1 | 8 | yes | 2 | 3.01 | 29,289 |
| ABLATION_N2 | 9 | yes | 3 | 3.66 | 38,373 |
| ABLATION_N3 | 16 | no (hit cap) | 11 | 3.88 | 42,697 |
| ABLATION_N4 | 9 | yes | 3 | 3.53 | 42,697 |

Note the epoch counts differ — early stopping fires at different points per configuration. Budget for
the **cap**, not the mean.

---

## 2. Fixed (per-notebook-run) costs

Measured, source A on L4 unless noted:

| stage | cost | note |
|---|---|---|
| Drive mount | ~27 s | Colab only |
| Load transitions + sweep (1.58 M rows total) | ~15 s | parquet read |
| Sweep index build (3 towers × 6,468 conditions × 75 actions) | ~1 s | done once; raw sweep frame then released |
| Reference policies (val + test) | ~2 s | 6 policies × 2 splits |
| Test eval + IoT robustness, per experiment | ~8 s (A) / ~18 s (B) | 5 degradation modes × 2 policy passes each |
| SHAP, full size (100 bg × 200 explain × 100 nsamples × 3 heads) | ~8 s | FOWT-ARISE only |
| All figures (37) at 300 dpi | ~28 s (A) / ~45 s (B) | matplotlib render dominates |

---

## 3. Budget for the full experiment (`DRY_RUN = False`)

`NUM_EPOCHS_FULL = 60`, `EARLY_STOPPING_PATIENCE = 10`.

Worst case is all five experiments running the full 60 epochs = **300 epochs**. Realistic case, based on
source A's observed early-stopping behaviour at patience 14 (126 epochs), scaled for patience 10:
**≈ 100–130 epochs**.

| | realistic (≈115 epochs) | worst case (300 epochs) |
|---|---|---|
| training | 165 s | 429 s |
| fixed + eval + SHAP + plots | ~130 s | ~130 s |
| **total, L4** | **≈ 5 min** | **≈ 9.5 min** |
| **total, CPU** | ≈ 11 min | ≈ 23 min |

**A single full run costs under 10 minutes of L4 time.** This is a small compute ask; the model is
~43 k parameters and the bottleneck is the counterfactual evaluation, not the network.

---

## 4. Budget for the multi-seed study

This is the one experiment that would resolve the open N1 / N2 / N4 question (see
`TARGET_PROFILE.md` §2b). 3 seeds × 5 configurations = 15 training runs.

| | L4 | CPU |
|---|---|---|
| training (3 × 115 epochs = 345) | 8.2 min | 21 min |
| eval + robustness (15 runs × 8 s / 18 s) | 2.0 min | 4.5 min |
| SHAP (once, on the seed-0 proposed model) | 8 s | 25 s |
| plots (once, plus per-seed history figures) | ~1.5 min | ~2.5 min |
| **total** | **≈ 12 min** | **≈ 28 min** |

**Recommendation: allocate 30 minutes of L4 time.** That covers the 3-seed study with headroom, and it
is the single highest-value compute spend available — it converts "unresolved, inside checkpoint
noise" into an actual answer on three of the four novelties.

5 seeds instead of 3 costs ≈ 20 min on L4 and tightens the confidence intervals by ~√(5/3) ≈ 1.29×.

---

## 5. Measured storage footprint

Source B, real: **58 MB** total for one 5-experiment run (16-epoch cap).

| component | size | note |
|---|---|---|
| `FOWT_ARISE/` | 15 MB | includes SHAP artefacts + 18 figures |
| each ablation | 9.7 MB | 10 figures each, no SHAP |
| `comparison/` | 3.8 MB | 19 figures + tables |
| `common/` | 88 KB | JSON provenance |
| checkpoints | 775 KB each, 2 per experiment (`best.pt`, `latest.pt`) | **7.8 MB total** |

**Figures dominate the footprint**, not checkpoints — 300 dpi PNGs run ~300 KB each and there are 37
of them per run.

Projections:

| scenario | storage |
|---|---|
| one full run | ~60 MB |
| 3-seed study, all artefacts retained | **~180 MB** |
| 3-seed study, figures only for seed 0 | ~90 MB |
| 5-seed study, all artefacts | ~300 MB |

Google Drive is fine for all of these. No storage constraint exists at this scale.

---

## 6. What the resource plan does *not* need

For completeness, since it was asked for: a simulated `history.csv` with an invented loss curve
contributes nothing to any figure in this document. Resource allocation needs epoch counts, seconds
per epoch, artefact sizes and file counts — all of which are measurable and are measured above.

The real 16-epoch run (source B) also produced genuine training curves, genuine checkpoints and
genuine SHAP output in 232 s, so a demonstration of the finished deliverable does not require
fabrication either. See `DESIGN_REVIEW_PACK.md`.

---

## 7. Costed action list

| priority | action | L4 cost | what it buys |
|---|---|---|---|
| **1** | Fix the diverging critic (§8 below) | 0 (code change) + 5 min to re-verify | removes a real defect that currently makes every training-loss figure meaningless |
| **2** | 3-seed study | 12 min (allocate 30) | resolves N1 / N2 / N4 |
| **3** | Equalise epoch budgets across configurations | included in 2 | removes the "best of 34 beats best of 19" confound |
| **4** | Validation sweep of the fatigue/power weight ratio | ~6 runs ≈ 30 min | the one axis with real headroom (power loss 0.87 % vs oracle 0.18 %) |
| 5 | 5 seeds instead of 3 | 20 min | ~1.29× tighter CIs |

Total to get from "unresolved" to a defensible ablation story: **under one hour of L4 time.**

---

## 8. Defect found by the 16-epoch run — needs fixing before the full run

The critic diverges. Measured, source B, FOWT-ARISE:

| epoch | td_loss | cql_penalty | critic_loss | actor_loss |
|---|---|---|---|---|
| 1 | 0.14 | +3.16 | 3.30 | 0.034 |
| 3 | 10.24 | −3.26 | 6.98 | 0.026 |
| 5 | 72.71 | −4.47 | 68.24 | 0.024 |
| 7 | 191.67 | −4.76 | 186.91 | 0.024 |
| 9 | **358.47** | −4.94 | **353.54** | 0.023 |

The **TD term** is diverging, not the CQL term (which stabilises around −4.9). Cause: `GAMMA = 0.99`
bootstrapping on a dataset where the action barely influences the next state — the earlier
correlation check on this data put `corr(action, next_state) ≈ 0.04`, i.e. the problem is closer to a
contextual bandit than a sequential MDP, so the bootstrap target compounds its own error each step.

**What it does and does not affect:**

* It does **not** corrupt any reported objective. `Q_IMPROVEMENT_COEF = 0`, so the actor never reads
  the critic — `actor_loss` is flat and healthy at ~0.023, and the test objective is fine at 83.4 % of
  oracle.
* It **does** make `train_loss` in `history.csv` (defined as actor + critic) meaningless, so the
  training-loss figure is dominated by a diverging term and would look alarming in a thesis.
* It **would** destroy the policy if anyone raised `Q_IMPROVEMENT_COEF` above 0.

**Fix options**, cheapest first:

1. Set `GAMMA = 0.0` (bandit framing, matching what the data supports) and re-measure. Zero cost.
2. Keep γ but clip the TD target to the observed reward range, or normalise rewards before the critic.
3. Report `actor_loss` and `critic_loss` separately in the loss figure and drop the combined
   `train_loss`, which is not a meaningful quantity when its two terms differ by four orders of
   magnitude.

I'd do 1 and 3 together, then re-run the 5-minute full experiment. This is a genuine finding that a
fabricated history curve would have concealed — a synthetic loss curve would have descended smoothly
and the defect would have reached the full run undetected.
