# Floating Offshore Wind Turbines — RL dataset for tower load relief

A reinforcement-learning dataset for **real-time structural load relief of floating
offshore wind turbines** via adaptive blade-pitch, yaw and individual-pitch control,
with an IoT sensor observation layer.

The dataset is built by extending [FLOATBench](https://huggingface.co/datasets/DeCoDELab/FLOATBench)
— a high-fidelity OpenFAST tower-fatigue benchmark — with the one thing it does not
contain: **control actions and their effect on structural damage**.

```
129,600 RL transitions  |  1,455,300 action-sweep rows  |  3 tower geometries  |  21,600 simulated hours
```

---

## The problem this solves

FLOATBench is excellent but it is a *design* dataset, not a *control* dataset:

| | FLOATBench | Needed for load-relief RL |
|---|---|---|
| Fatigue damage from real OpenFAST rainflow | ✅ 582,120 labels | ✅ |
| Environmental conditions (V, σ_V, Hs, Tp) | ✅ | ✅ |
| Per-section resolution | ✅ 30 sections × 3 towers | ✅ |
| **Control actions (pitch / yaw / IPC)** | ❌ controller held fixed (DLC 1.2) | ✅ |
| **Time-series / episode structure** | ❌ independent 10-min runs | ✅ |
| **Sensor realism** | ❌ perfect state | ✅ |
| **Reward signal** | ❌ | ✅ |

Every one of its 19,404 simulations uses the same baseline controller, with
`IPC_ControlMode = 0`. So the dataset cannot tell you what happens if you pitch or
yaw differently — which is exactly the question a load-relief agent must answer.

This repository adds that layer **without overwriting any measured value**.

## The core idea: ratio anchoring

Absolute damage always comes from FLOATBench. The physics model only ever supplies
the dimensionless *relative* change caused by an action:

```
D_out(section) = D_floatbench(section) × [ Σ c_k (g_k A_k)^m ] / [ Σ c_k A_k^m ]
                 └── measured ──────┘   └──── model: relative effect only ────┘
```

Two properties follow, and they are verified as tests and in the build manifest:

1. **Zero action is bit-exact.** With no control action every gain `g_k = 1`, the
   ratio is identically `1.0`, and the output *is* the unmodified FLOATBench damage.
   Measured `max_abs_damage_difference = 0.0` on all 582,120 labels.
2. **Control authority is bounded by physics.** Wave-driven load paths have gain
   fixed at 1, so no action can reduce damage below the wave floor. An agent cannot
   pitch its way out of wave-induced tower fatigue.

## How the load split is obtained

Not all tower fatigue is controllable. The split is **fitted from FLOATBench itself**,
not assumed. For each of the 30 cross-sections, non-negative least squares fits four
physically-motivated load paths against the 6,468 real damage values:

| load path | driver | controllable |
|---|---|---|
| `thrust_turbulence` | `ρ A Ct(V) V σ_V` — turbulent thrust fluctuation | ✅ |
| `rotor_cyclic` | mean thrust, scaling nP cyclic loading | ✅ |
| `wave_quasistatic` | `Hs × DAF(Tp, f_FA1)` — tower-resonance amplified | ❌ |
| `wave_inertial` | `Hs / Tp²` — platform acceleration | ❌ |

Fit quality (FLOATBench's own DEL target, `D^(1/3)`):

| tower | f_FA1 | R²(DEL) mean | R²(DEL) min | DEL median rel. err | controllable share (mean / p95) |
|---|---|---|---|---|---|
| `ref`  | 0.336 Hz | 0.732 | 0.720 | 18.0 % | 0.211 / 0.726 |
| `opt1` | 0.573 Hz | 0.793 | 0.761 | 12.5 % | 0.323 / 0.878 |
| `opt2` | 0.537 Hz | 0.802 | 0.760 | 12.2 % | 0.308 / 0.865 |

## Where the aerodynamics come from

The pitch/yaw response is **not fitted and not assumed** — it is read from the
official IEA-22-280-RWT rotor performance surfaces (20 × 20 grid of `Cp`, `Ct`, `Cq`
over pitch −5…45° and TSR 2…14, generated from the real blade geometry and airfoil
polars). Absolute baseline values come from the 22 official OpenFAST steady-state
operating points. Resulting pitch sensitivity above rated is −2.3 … −4.2 MW/deg,
which is the right physical order for a 22 MW rotor.

Actuator limits are the real values from `IEA-22-280-RWT-Semi_DISCON.IN`:
pitch rate 2.0 °/s, yaw rate 0.499 °/s, yaw deadband 8°.

## The control problem is genuinely state-dependent

Mean reward by wind-speed band and fixed policy (tower `opt2`) — the best action
changes with conditions, which is what makes this worth solving with a policy:

| wind speed | baseline | feather | IPC only | mean controllable share |
|---|---|---|---|---|
| 0–6 m/s   | −0.006 | **+0.006** | −0.003 | 0.223 |
| 6–9 m/s   | −0.059 | **+0.142** | +0.081 | 0.659 |
| 9–12 m/s  | −0.108 | **+0.187** | +0.152 | 0.792 |
| 12–16 m/s | −0.041 | −0.250 | **+0.032** | 0.448 |
| 16–20 m/s | **−0.007** | −0.485 | −0.015 | 0.156 |
| 20+ m/s   | **−0.006** | −0.596 | −0.016 | 0.044 |

Above 16 m/s wave loading dominates, control authority collapses, and the correct
action is to do nothing. Between 6 and 12 m/s feathering pays for itself, backed by
a genuinely monotone, correctly-priced power cost at every wind speed (see
[EDA/EDA_REPORT.md §G](EDA/EDA_REPORT.md) for how an earlier free-feathering
accounting artefact below rated was found and fixed). A learned policy has to
find this structure.

Note also that **yaw misalignment makes tower fatigue worse** (damage ratio ≈ 1.39
at 30°): it lowers mean thrust but raises cyclic loading, and costs power. IPC is the
only near-free relief, limited by actuator duty.

---

## Quick start

```bash
pip install -r requirements.txt

python scripts/fetch_turbine_data.py      # official IEA-22-280-RWT data (~700 kB)
python scripts/download_floatbench.py     # FLOATBench from Hugging Face (~103 MB)

PYTHONPATH=src python -m fowt_rl.build_dataset
```

Build takes ~17 s and writes 52 MB into `data/processed/`.

```bash
PYTHONPATH=src python -m fowt_rl.build_dataset --calibrate-only        # just fit + report
PYTHONPATH=src python -m fowt_rl.build_dataset --episodes 200 --no-sweep   # quick build
PYTHONPATH=src python -m fowt_rl.build_dataset --no-iot                # perfect sensors
PYTHONPATH=src python -m fowt_rl.build_dataset --iot-severity 2.0      # degraded sensors
PYTHONPATH=src python -m pytest tests -q                               # 20 tests
```

## Using the data

### Offline RL

```python
from fowt_rl.env import OfflineTransitionDataset

data = OfflineTransitionDataset.load("data/processed/transitions")
arrays = data.arrays()
# observations (129600, 23) | actions (129600, 3) | rewards | next_observations | terminals
```

### Online RL

```python
import numpy as np
from fowt_rl.env import FowtLoadReliefEnv

env = FowtLoadReliefEnv(tower="opt2")
obs, info = env.reset(seed=0)
for _ in range(36):
    obs, reward, terminated, truncated, info = env.step(np.array([2.0, 0.0, 1.0]))
    if terminated or truncated:
        break
```

Gymnasium spaces are exposed automatically if `gymnasium` is installed
(`pip install -e ".[gym]"`); otherwise the env is plain `reset`/`step`.

### Surrogate / supervised use

`data/processed/action_sweep/` holds a full factorial sweep
(5 pitch × 5 yaw × 3 IPC = 75 actions) at every one of the 6,468 conditions per
tower — 485,100 rows each. Use it to train a fast damage/reward surrogate, or to
audit how each action dimension moves damage, power and thrust.

## Dataset layout

```
data/
├── turbine/                       official IEA-22-280-RWT reference data (committed)
│   ├── iea22_steady_states.csv        22 baseline operating points
│   ├── iea22_rotor_performance.csv    400-point Cp/Ct/Cq surface
│   └── iea22_properties.json
├── calibration/                   fitted models + fit-quality reports (committed)
│   ├── load_model.json                per-section coefficients, all 3 towers
│   └── aero_reference_check.json
├── processed/                     generated datasets (committed)
│   ├── transitions/               RL transitions, 43,200 rows × 89 cols per tower
│   ├── action_sweep/              485,100 rows × 27 cols per tower
│   ├── samples/                   2,000-row CSV previews
│   ├── manifest.json              provenance, validation, summary statistics
│   └── pipeline_config.json       exact config that produced the build
└── raw/                           downloads (git-ignored, regenerate via scripts/)
```

### Observation columns

Train on the measured channels; the ground-truth and privileged columns are for
evaluation and ablation only. The full grouping is in
`manifest.json → observation_manifest`.

| group | columns |
|---|---|
| `meas_*` (8) | what the IoT network delivered — **train on these** |
| `valid_*` (8) + `sensor_health` | packet-loss flags |
| proprioceptive (7) | previous action, nacelle yaw, yaw error, cumulative damage, step |
| `true_*` / `next_true_*` | ground truth — evaluation only |
| privileged | `controllable_share_*`, `damage_baseline`, `inflow_direction_deg`, gains |

### IoT sensor layer

Each channel gets calibration bias, drift, absolute + proportional noise,
quantisation, latency and packet loss, applied per-episode so errors are correlated
within an episode. Realised errors are reported in the manifest, e.g. median relative
error 12.9 % on turbulence intensity, 8.7 % on `Hs`, 3.9 % on the strain-derived
damage rate, with 0.2–2 % dropout.

## Reward

```
reward = 2.0 · fatigue_relief · severity     # 1 − DEL_ctrl/DEL_ref, weighted by
       − 1.0 · power_loss_fraction           #   lifetime significance
       − 0.05 · actuator_duty
```

`severity` uses FLOATBench's own `damage_weight` (25-year occurrence probability), so
relief is rewarded in proportion to the lifetime damage it actually avoids. All
weights live in `fowt_rl.config.RewardConfig`.

The reference is the aerodynamically ideal baseline (zero offset, perfect alignment).
An uncontrolled turbine scores slightly *below* zero against it, because vane bias
and the 8° yaw deadband leave a standing misalignment — so correcting that
misalignment is itself a source of reward, mirroring a genuine value-add of active
yaw control.

## Documentation

- [`EDA/EDA_REPORT.md`](EDA/EDA_REPORT.md) — 19-figure exploratory analysis with per-use-case verdicts
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — full derivation, equations, provenance
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — **read before publishing any result**

## Exploratory data analysis

```bash
PYTHONPATH=src python EDA/run_eda.py     # 19 figures, 10 tables, summary_stats.json
```

Confirms clean integrity (0 NaN / 0 infinite / 0 duplicate rows across 129,600
rows, full grid coverage) and that the control problem is well-posed. It also
**found and drove the fix** for one material artefact: below rated, feathering
used to appear free — the reference pitch schedule sat ~1–2° off the performance
surface's Cp optimum, so 6.6 % of transitions (34 % of the 6–9 m/s band) got
load relief at essentially zero power cost. `fowt_rl.aero.RotorAero._ratios` now
anchors the ratio on the surface's own Cp-optimal pitch instead, closing this to
0.005 % (residual sub-0.01° commands below float64 precision). Full account,
verification and regression tests in [`EDA/EDA_REPORT.md`](EDA/EDA_REPORT.md) §G
and `docs/LIMITATIONS.md`.

## Honest scope

This is a **physics-informed synthetic extension** of FLOATBench, not a re-simulation.
The baseline damage, environmental conditions and rotor aerodynamics are all measured
reference data. The *action response* is a calibrated reduced-order model that has
**not been validated against OpenFAST runs with modified control**. Two coefficients
(`yaw_cyclic_gain`, `ipc_authority`) are literature-informed rather than derived.

It is suitable for developing and benchmarking control policies, and for
methodology work. It is not a substitute for aeroelastic simulation or
certification analysis. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Attribution

| source | licence |
|---|---|
| [FLOATBench](https://huggingface.co/datasets/DeCoDELab/FLOATBench) — Ribeiro et al., [arXiv:2605.25717](https://arxiv.org/abs/2605.25717) | CC-BY-4.0 |
| [FLOATBench code](https://github.com/Joao97ribeiro/FLOATBench) | MIT |
| [IEA-22-280-RWT](https://github.com/IEAWindSystems/IEA-22-280-RWT) — IEA Wind Task 55 REFWIND | Apache-2.0 |
| [OpenFAST](https://github.com/OpenFAST/openfast) / [ROSCO](https://github.com/NREL/ROSCO) | Apache-2.0 |

Generated data inherits CC-BY-4.0 from FLOATBench. Code in this repository is MIT.
