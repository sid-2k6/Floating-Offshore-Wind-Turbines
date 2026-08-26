# Limitations

Read this before using results from this dataset in a report, thesis or paper.

## What this dataset is

A **physics-informed synthetic extension** of FLOATBench. Baseline fatigue
damage, environmental conditions and rotor aerodynamics are all measured
reference data. The *response to control actions* is a calibrated reduced-order
model.

## What it is not

It is **not** a re-simulation of OpenFAST with modified control, and it is not
validated against one. Nobody has run OpenFAST with the pitch/yaw/IPC actions in
this dataset and confirmed the resulting damage. That is the single most
important caveat.

---

## 1. The action response is unvalidated

The load-relief numbers rest on a chain:

```
pitch/yaw command
  → Ct ratio                    (official rotor performance surface — measured)
  → aerodynamic load amplitudes (standard relations — sound)
  → per-section damage          (NNLS fit on FLOATBench, R²(DEL) 0.73–0.80 — fitted)
  → damage ratio                (closed form — exact given the above)
```

Each link is individually defensible, but the *composition* has never been
checked against a high-fidelity run with the same control action. Reported
percentage load reductions should be read as **model-consistent estimates**, not
measurements.

**Mitigation.** Run a handful of confirmatory OpenFAST simulations at selected
(condition, action) pairs and compare. That is the single highest-value
validation step available, and it does not require a full campaign — a dozen runs
covering low/mid/high wind and low/high `Hs` would substantially firm up or
falsify the model.

## 2. Two coefficients are literature-informed, not derived

| parameter | default | what it controls |
|---|---|---|
| `yaw_cyclic_gain` | 1.5 | how much cyclic loading grows with yaw misalignment |
| `ipc_authority` | 0.30 | how much cyclic amplitude full IPC cancels |

These directly scale the control authority of yaw and IPC. They are the *only*
parameters in the pipeline that are neither measured nor fitted, and they are
load-bearing for two headline results:

- that **yaw misalignment worsens** tower fatigue (driven by `yaw_cyclic_gain`)
- that **IPC is near-free relief** (driven by `ipc_authority`)

Both are configurable in `fowt_rl.damage.ControlAuthorityConfig`. Any claim that
depends on them should be accompanied by a sensitivity sweep.

## 3. The load decomposition explains 73–80 % of variance

R²(DEL) is 0.732 / 0.793 / 0.802 for `ref` / `opt1` / `opt2`, with median
relative DEL error 12–18 %. So 20–27 % of the variance in FLOATBench damage is
**not** captured by the four-basis model.

Consequences:

- The controllable share `φ` is an estimate with real uncertainty. Where the fit
  is poor, `φ` may be biased either way, so control authority may be over- or
  under-stated for specific conditions.
- The `ref` tower fits worst (R² 0.732). Its first fore-aft frequency is 0.336 Hz
  — much closer to the 3P excitation range than the two re-designs — so it likely
  has resonance behaviour the single-DOF DAF does not represent. Treat `ref`
  results with more caution than `opt1`/`opt2`.

Missing physics that could explain the residual: platform pitch/surge natural
modes, second-order difference- and sum-frequency wave forcing, wind–wave
misalignment, aerodynamic damping of tower motion, and controller–platform
coupling (the well-known negative-damping problem for floating turbines).

## 4. Aerodynamics are steady and rigid

`fowt_rl.aero` is quasi-static: it maps `(V, λ, θ, γ)` to `Ct`/`Cp` through a
steady performance surface. It does not model:

- unsteady aerodynamics or dynamic stall
- blade flexibility and aeroelastic coupling
- rotor-speed transients (the torque controller is assumed to hold the baseline
  TSR, which is not true during large pitch excursions)
- platform-motion-induced relative inflow — a genuinely important effect for
  floating turbines, since platform pitch changes the apparent wind at the rotor
- skewed-inflow detail beyond the cosine-power laws
- tower shadow, wind shear and veer as explicit inputs

The rotor-speed assumption matters most: applying +8° pitch would in reality
change rotor speed and hence `λ`, moving the operating point on the surface. The
model holds `λ` at its baseline value.

## 5. No platform motion or mooring state

FLOATBench's release contains tower-section damage only. Platform 6-DOF motion,
mooring line tension and blade-root loads are **absent**, so this dataset covers
**tower fore-aft fatigue only**.

The original problem statement mentions structural load relief broadly. Blade-root
fatigue, mooring tension cycling and platform motion limits are outside scope. In
particular, IPC's main documented benefit is on *blade* loads, which cannot be
scored here.

FLOATBench's authors generated but did not release the underlying ~190 GB of
88-channel 10 Hz OpenFAST time series, which does contain platform motion and
mooring tensions. Requesting it is the natural route to extending scope.

## 6. Episodes are synthetic

- The random walk over the operating grid is a plausible but invented model of
  metocean evolution. It is not a hindcast and matches no real site's temporal
  statistics.
- The 600 s step comes from FLOATBench's 10-minute stationary runs. Real load
  mitigation control acts at 1–100 Hz. **This dataset cannot represent
  wave-frequency or turbulence-frequency closed-loop control** — it represents
  supervisory setpoint scheduling on a 10-minute cadence. Anything claiming
  "real-time" control at wave frequency needs a different data source.
- Successive steps are independent turbulence seeds, so there is no genuine
  dynamic state carried in the structure between steps. Tower and platform
  dynamics do not persist across a step boundary.
- Wind and waves are always aligned (FLOATBench's design). Misalignment, which
  materially affects both fatigue and the value of yaw control, is absent.

## 7. Inflow direction is invented

FLOATBench has no direction information. The Ornstein–Uhlenbeck process
(σ = 12°, τ = 2 h) and the per-episode vane bias (σ = 2.5°) are plausible but not
site-calibrated. Since the yaw action exists only relative to this process, all
yaw-related conclusions inherit its assumptions.

## 8. IoT parameters are representative, not measured

Noise, drift, quantisation, latency and dropout figures are plausible for
instrumentation-grade offshore sensors but come from no specific datasheet or
deployment. The sensor network is modelled channel-by-channel: there is **no
correlated network failure**, no clock-synchronisation error between channels,
and no burst loss. Real wireless/fieldbus networks fail in correlated bursts.

Use `--iot-severity` to bracket the sensitivity of any result to this layer.

## 9. Reward weights are a modelling choice

The defaults (`fatigue 2.0`, `power 1.0`, `duty 0.05`) are not derived from an
economic model of fatigue cost versus energy revenue. They were chosen to make
the control problem non-degenerate. A different weighting yields a different
optimal policy — this is a design knob, not a physical constant, and the optimal
policy reported anywhere should always be quoted with its weights.

## 10. Single design family, single platform

All three towers sit on the same three-column semi-submersible with the same
mooring and controller, and all are 22 MW. Nothing here transfers automatically to
spar or TLP platforms, other power ratings, or other controller designs. The
`ref → opt1 → opt2` variation is tower geometry only.

## 11. Not certification-grade

FLOATBench's own broader-impact statement notes its labels are single-slope S-N
values under DLC 1.2 only and are not a substitute for certification-grade
analysis. Everything derived here inherits that. DLC 1.2 is normal power
production — no faults, no shutdowns, no extreme events, no idling. A real
load-relief controller must be safe across all design load cases, and this
dataset says nothing about the ones that matter most for safety.

---

## Suggested wording for a report

> The dataset was constructed by extending the FLOATBench floating offshore wind
> turbine tower-fatigue benchmark with a control-action layer. Baseline fatigue
> damage, environmental conditions and rotor aerodynamic coefficients are taken
> from measured reference data (FLOATBench OpenFAST simulations and the official
> IEA-22-280-RWT definition). The response of tower fatigue to blade-pitch, yaw
> and individual-pitch actions is modelled by decomposing the measured damage
> into aerodynamic and hydrodynamic load paths via non-negative least squares
> (R²(DEL) = 0.73–0.80) and rescaling the aerodynamic paths using the official
> rotor performance surfaces. A zero control action reproduces the original
> FLOATBench damage exactly. The action response has not been validated against
> OpenFAST simulations with modified control, and two coefficients governing
> yaw-induced cyclic loading and IPC authority are literature-informed rather
> than derived; results are therefore model-consistent estimates rather than
> validated load reductions. The 600 s control step reflects FLOATBench's
> 10-minute stationary simulations, so the setting is supervisory setpoint
> scheduling rather than wave-frequency closed-loop control.

## Priority list if you have more time

1. Confirmatory OpenFAST runs at selected (condition, action) pairs — §1
2. Sensitivity sweep over `yaw_cyclic_gain` and `ipc_authority` — §2
3. Request FLOATBench's unreleased time-series for platform motion and mooring — §5
4. Add platform pitch/surge modes to the load basis to lift R², especially for
   `ref` — §3
5. Replace the synthetic metocean walk with a real hindcast time series — §6
