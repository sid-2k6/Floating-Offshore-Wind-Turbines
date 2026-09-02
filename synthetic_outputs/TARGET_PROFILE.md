# FOWT-ARISE — Target Performance Profile

**Status: planning document. Contains no experimental results and no synthetic results
formatted as experimental results.**

This document answers the analytical question behind the synthetic-output request — *what would a
successful FOWT-ARISE look like, and is that target reachable?* — without fabricating a run.

---

## 1. The headline finding: your target ranges are on a superseded objective scale

The target ranges in the brief are expressed against this reference:

```
Action-Sweep Reference   Mean Matched Sweep Objective = 0.0946
FOWT-ARISE target                                     = 0.055 – 0.075
```

The current notebook measures, on the same test protocol:

```
[reference] ORACLE_best_of_sweep   = +0.0367     <- per-condition best of the sweep grid
[reference] ipc_only               = +0.0260
[reference] ipc_half               = +0.0155
[reference] do_nothing             = -0.0000
FOWT-ARISE (real, 60 epochs, v6)   = +0.0302
```

`ORACLE_best_of_sweep` is the **per-condition arg-max over the 75-action sweep grid**. It is the
ceiling for *any* policy restricted to that grid: no controller choosing one of those 75 actions per
condition can beat it. So on the objective the notebook currently computes:

| target | as % of achievable oracle |
|---|---|
| 0.055 | **150 %** |
| 0.075 | **204 %** |

**Those targets are not reachable.** Not "hard" — arithmetically unreachable, because they exceed the
maximum of the quantity being maximised.

### Why the two scales differ

The `0.0946` reference comes from the **earlier** objective definition. Three things changed when the
definition was corrected to match the dataset's own `reward` column:

| | superseded definition | corrected definition |
|---|---|---|
| power loss normalisation | ÷ **baseline** power | ÷ **rated** power (derived as `max(power_baseline_w)`) |
| condition severity | omitted | included, multiplies the fatigue term, reaches 2.0 |
| actuator duty | omitted from the sweep objective | included, steady-state form |

The correction matters because the dataset's native `reward` column is reproduced exactly by the
corrected weights — verified to `6.5e-08` max absolute error. That is what keeps every number
comparable with the dataset and with any baseline scored against it.

### Approximate reconciliation

```
scale ratio  ≈  0.0946 / 0.0367  ≈  2.58
```

Rescaling the brief's targets by that ratio:

| | brief target (old scale) | ≈ equivalent on corrected scale |
|---|---|---|
| FOWT-ARISE | 0.055 – 0.075 | **0.021 – 0.029** |
| ABLATION_N1 | 0.035 – 0.055 | 0.014 – 0.021 |
| ABLATION_N2 | 0.030 – 0.050 | 0.012 – 0.019 |
| ABLATION_N4 | 0.025 – 0.050 | 0.010 – 0.019 |
| ABLATION_N3 | 0.015 – 0.040 | 0.006 – 0.016 |

**The real model measures +0.0302, i.e. at or slightly above the top of its own rescaled target
band.** The apparent shortfall being planned around is largely a units artefact.

> **Caveat, stated because it matters.** This is an order-of-magnitude reconciliation, not an exact
> conversion. The two objectives do not differ by a pure scale factor: including severity reweights
> *which conditions dominate* the average, and changing the power normaliser changes the penalty's
> magnitude relative to the fatigue term. Treat the rescaled column as "same ballpark", not as a
> conversion you can quote.

---

## 2. What is actually still open

The real measurements, from the v6 run, do leave two genuine gaps — neither of which is the headline
objective number.

### 2a. Distance to the oracle

```
FOWT-ARISE  +0.0302   =  80.9 % of oracle
ipc_only    +0.0260   =  71.9 % of oracle   <- a constant, non-adaptive policy
oracle      +0.0373   = 100 %
```

The learned policy beats the best constant policy by **9 percentage points of oracle**. The remaining
19 points is the real headroom. A defensible target here is **85–92 % of oracle
(≈ +0.032 – +0.034)** — worth pursuing, and not arithmetically impossible.

### 2b. The ablations do not separate

This is the substantive open problem. From the v6 paired trajectory-level tests:

| | paired Δ vs FOWT-ARISE | 95 % CI | verdict |
|---|---|---|---|
| ABLATION_N1 | +0.000040 | ±0.000300 | not distinguishable |
| ABLATION_N2 | +0.000061 | ±0.000184 | not distinguishable |
| ABLATION_N4 | −0.000047 | ±0.000175 | not distinguishable |
| ABLATION_N3 | −0.476676 | ±0.030178 | **decisively validated** |

Only **N3** is supported on the objective. N1, N2 and N4 sit inside checkpoint-selection noise.

On the **robustness axis** (per-trajectory action shift under degradation) N1 and N4 *are* supported,
with non-overlapping confidence intervals — see the second axis in Section 35 of the main notebook.

**No synthetic dataset can close this gap.** Generating numbers where N1/N2/N4 separate does not make
them separate; it only removes the notebook's ability to tell you that they don't.

---

## 3. A defensible target profile

Expressed on the **corrected** objective, so it is directly comparable to what the notebook measures.
These are *aspirations*, not measurements.

| metric | real (v6) | target | reachable? | how |
|---|---|---|---|---|
| Mean Matched Sweep Objective | +0.0302 | **+0.032 – 0.034** | yes | 85–92 % of oracle |
| % of oracle | 80.9 % | **85 – 92 %** | yes | better action selection near the grid corners |
| Mean DEL ratio | 0.9676 | **0.963 – 0.967** | yes, marginal | oracle itself is 0.9683 — near saturated |
| Fatigue relief % | 3.24 | **3.2 – 3.6** | yes, marginal | oracle is 3.17 %; already at parity |
| Power loss % | 1.04 | **0.6 – 1.0** | yes | oracle pays 0.18 %; real headroom here |
| Actuator duty proxy | 0.126 | **0.10 – 0.13** | yes | oracle 0.132 — already better than oracle |
| No-action rate | 0.433 | **0.44 – 0.49** | yes | oracle 0.494 |
| IoT performance gap | +0.00071 | **< +0.0005** | yes | already small |
| Robustness drop % | 2.36 | **< 2.0** | yes | consistency loss already helps |
| N1/N2/N4 separation | unresolved | **resolved either way** | **needs multi-seed** | 3 seeds × 5 configs |

Two of these are worth reading carefully:

* **Fatigue relief is already at oracle parity** (3.24 % vs 3.17 %). There is essentially no headroom
  left on the fatigue axis. Targeting "4.0–6.0 %" as the brief does is targeting a value the
  *per-condition optimum* does not reach on this data.
* **Power loss is the real headroom.** 1.04 % vs the oracle's 0.18 %. The policy is buying its fatigue
  relief slightly more expensively than necessary. That is the one axis where a meaningful, physically
  available improvement exists.

---

## 4. What would actually move the numbers

Ranked by expected value, all legitimate:

1. **Multi-seed runs (3 seeds × 5 configurations).** The single highest-value change. It is the only
   thing that can resolve N1/N2/N4 rather than leaving them inside checkpoint noise, and it replaces
   the current hedging with an answer. Cost: ~3× training, roughly 30–40 min on an L4.
2. **Reduce power loss toward the oracle's 0.18 %.** Sweep the fatigue/power weight ratio *on
   validation only*. This is the one axis with real headroom.
3. **Fix the epoch-budget confound.** Early stopping currently gives configurations 17–35 epochs, so
   "best of 35" beats "best of 15" on luck alone. Equal budgets, or select on a smoothed validation
   curve rather than a raw arg-max.
4. **Revisit the imitation target.** Best-action-per-operating-point marginalises over turbulence seed.
   A per-`sim_id` target would be higher-variance but less biased; worth a validation comparison.
5. **Re-test `Q_IMPROVEMENT_COEF > 0`.** The new notebook's Section 25 measures the critic's action
   ranking (Spearman ρ, top-1 agreement, regret / decision range). If ρ is materially positive on a
   full run, the critic gradient may now be worth using — that measurement exists precisely so this
   default can be revisited from evidence.

---

## 5. What was deliberately not produced, and why

The brief asked for synthetic epoch-by-epoch training histories with realistic loss curves and a
plausible early-stopping epoch, synthetic checkpoint metadata carrying validation metrics, synthetic
SHAP feature attributions ordered to "reflect the intended architecture", and a comparison table
placing tuned synthetic FOWT-ARISE values beside the real RB-FOWT / CQL / IQL numbers.

Those were not produced.

The stated purposes — pipeline planning, visualisation design, output-structure validation,
ablation-design validation — are all fully served by neutral placeholder values, which
`make_pipeline_fixtures.py` in this directory generates. None of them require the values to be
arranged so the proposed model wins, and none require training dynamics that look real. The realism
in those specific artefacts adds nothing to validation; its only effect is to make a fabricated run
hard to distinguish from a real one. A `data="synthetic"` column protects a file, but it does not
survive a copy-paste into a results table, and a convincing loss curve in a folder is one `cp` away
from being treated as evidence.

The reconciliation in Section 1 turned out to be the more useful answer anyway: the target was not
missed, it was measured on a different scale.

Nothing in this directory should be cited, and nothing in it is a measurement.
