"""Tower fatigue load decomposition, calibrated on real FLOATBench damage.

The problem
-----------
FLOATBench gives us, for each 10-minute environmental condition and each of 30
tower cross-sections, a *real* Miner-summed fatigue damage value obtained by
rainflow counting an OpenFAST tower bending-moment time series. What it does not
give us is how that damage would change under a different control action.

Crucially, not all of the damage is controllable. Tower fore-aft fatigue on a
floating turbine is driven by several largely independent excitation paths, and
rotor control has authority over only some of them:

======================  ==========================================  ============
load path               physical driver                             controllable
======================  ==========================================  ============
turbulent thrust        wind-speed fluctuations modulating thrust    yes
rotor harmonics (nP)    shear/veer/yaw-error cyclic rotor loading   yes
quasi-static wave       wave elevation, amplified near tower FA1     no
inertial wave           platform acceleration under wave forcing      no
======================  ==========================================  ============

Estimating that split is the whole job of this module, and it is done by
regression on FLOATBench itself rather than by assumption.

Method
------
For a single-slope S-N curve with Wohler exponent m, damage accumulates in
proportion to load amplitude to the power m, and contributions from independent
load processes add. So for section s:

    D(condition, s) ~ sum_k  c_{s,k} * A_k(condition)^m

with four physically-motivated condition-level load amplitudes:

    A_turb    = rho * A_rotor * Ct(V) * V * sigma_V
                standard deviation of rotor thrust, since dT/dV = rho A Ct V.
                sigma_V is FLOATBench's own realised 10-minute wind-speed
                standard deviation, so this column is grounded in real data.

    A_cyc     = 0.5 * rho * A_rotor * Ct(V) * V^2
                mean rotor thrust, used as the scale for once-per-revolution and
                blade-passing cyclic tower loading driven by inflow asymmetry.

    A_wave_qs = Hs * DAF(Tp)
                quasi-static wave loading, amplified near the tower's first
                fore-aft mode by a single-degree-of-freedom factor
                    DAF = 1 / sqrt((1-r^2)^2 + (2*zeta*r)^2),  r = (1/Tp)/f_FA1
                with the per-tower f_FA1 from the FLOATBench paper (Table 4).

    A_wave_in = Hs / Tp^2
                acceleration-driven inertial loading from platform motion.

Per-section coefficients are fitted by *non-negative* least squares against the
6,468 real damage values for that section, with relative-error weighting so the
fit is not dominated by the highest-damage conditions (damage spans ~6 orders of
magnitude). Fit quality is reported both on damage and on FLOATBench's own
regression target, the damage-equivalent load DEL = D^(1/m).

How control actions enter
-------------------------
Each basis column carries a gain describing how a control action rescales it:

    A_turb    -> A_turb    * ct_ratio
    A_cyc     -> A_cyc     * ct_ratio * (1 + yaw_cyclic_gain*|sin(gamma)|)
                                      * (1 - ipc_authority*kappa)
    A_wave_*  -> unchanged

where `ct_ratio` is the thrust-coefficient ratio from the calibrated rotor
aerodynamics (`fowt_rl.aero`), gamma is the residual yaw misalignment and kappa
is the IPC activation level.

The asymmetry between the two aerodynamic columns is what makes the control
problem non-trivial: yawing out of the wind lowers mean thrust but *raises*
cyclic rotor loading, while IPC attacks the cyclic term only and costs actuator
duty rather than power.

Only two parameters here are not derived from data - `yaw_cyclic_gain` and
`ipc_authority` - and both are literature-informed with documented defaults; see
docs/LIMITATIONS.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .aero import RotorAero
from .floatbench import N_SECTIONS, TowerData
from .turbine import TOWER_FA1_HZ

BASIS_NAMES = ("thrust_turbulence", "rotor_cyclic", "wave_quasistatic", "wave_inertial")
N_BASIS = len(BASIS_NAMES)

# Indices of the controllable basis columns.
IDX_TURB, IDX_CYC, IDX_WAVE_QS, IDX_WAVE_IN = range(N_BASIS)


@dataclass
class LoadModelConfig:
    """Configuration for the load decomposition."""

    # Wohler / S-N slope. FLOATBench states DEL ~ D^(1/m) with m = 3, consistent
    # with the single-slope welded-steel curve of DNV-RP-C203.
    wohler_exponent: float = 3.0

    # Candidate tower fore-aft damping ratios; the best fitting value is chosen
    # per tower by grid search.
    damping_ratio_grid: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)

    # Residual weighting exponent p: rows are weighted by damage^(-p).
    #   p = 0    plain least squares in damage space (dominated by the largest
    #            damage values)
    #   p = 1-1/m delta-method weighting, i.e. least squares in DEL space
    #   p = 1    pure relative error
    # Damage spans ~6 orders of magnitude across the FLOATBench envelope, so the
    # exponent materially changes the fit; it is grid searched alongside damping.
    weight_exponent_grid: tuple[float, ...] = (0.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0)

    def __post_init__(self) -> None:
        if self.wohler_exponent <= 0:
            raise ValueError("wohler_exponent must be positive")


# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------
def aerodynamic_amplitudes(
    conditions: pd.DataFrame,
    aero: RotorAero,
    pitch_offset_deg=0.0,
    yaw_error_deg=0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turbulent-thrust and cyclic-thrust amplitudes, plus the Ct ratio.

    Returns
    -------
    (A_turb, A_cyc, ct_ratio) each of shape (n_conditions,).
    """
    mean_wind = conditions["mean_wind_speed"].to_numpy(dtype=float)
    sigma_wind = conditions["std_wind_speed"].to_numpy(dtype=float)
    gradient, response = aero.thrust_sensitivity_to_wind(
        mean_wind, pitch_offset_deg, yaw_error_deg
    )
    turbulent = np.abs(gradient) * sigma_wind
    cyclic = np.abs(response["thrust_n"])
    return turbulent, cyclic, response["ct_ratio"]


def dynamic_amplification(wave_tp, fa1_hz: float, damping_ratio: float) -> np.ndarray:
    """Single-DOF dynamic amplification factor at the wave peak frequency."""
    wave_tp = np.asarray(wave_tp, dtype=float)
    frequency_ratio = (1.0 / np.maximum(wave_tp, 1e-6)) / fa1_hz
    denominator = np.sqrt(
        (1.0 - frequency_ratio**2) ** 2 + (2.0 * damping_ratio * frequency_ratio) ** 2
    )
    return 1.0 / np.maximum(denominator, 1e-6)


def wave_load_amplitudes(
    conditions: pd.DataFrame, fa1_hz: float, damping_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Quasi-static and inertial wave load amplitudes for each condition."""
    hs = conditions["wave_hs"].to_numpy(dtype=float)
    tp = conditions["wave_tp"].to_numpy(dtype=float)
    quasistatic = hs * dynamic_amplification(tp, fa1_hz, damping_ratio)
    inertial = hs / np.maximum(tp, 1e-6) ** 2
    return quasistatic, inertial


# ---------------------------------------------------------------------------
# Calibrated model
# ---------------------------------------------------------------------------
@dataclass
class TowerLoadModel:
    """Calibrated per-section load-path decomposition for one tower variant."""

    tower: str
    fa1_hz: float
    damping_ratio: float
    wohler_exponent: float
    coefficients: np.ndarray  # (N_SECTIONS, N_BASIS), non-negative
    basis_scale: np.ndarray  # (N_BASIS,) normalisation constants
    r2_damage_per_section: np.ndarray = field(default_factory=lambda: np.zeros(N_SECTIONS))
    r2_del_per_section: np.ndarray = field(default_factory=lambda: np.zeros(N_SECTIONS))

    # -- basis --------------------------------------------------------------
    def raw_basis(
        self,
        conditions: pd.DataFrame,
        aero: RotorAero,
        pitch_offset_deg=0.0,
        yaw_error_deg=0.0,
    ) -> np.ndarray:
        """Un-normalised load amplitudes, shape (n_conditions, N_BASIS)."""
        turbulent, cyclic, _ = aerodynamic_amplitudes(
            conditions, aero, pitch_offset_deg, yaw_error_deg
        )
        quasistatic, inertial = wave_load_amplitudes(conditions, self.fa1_hz, self.damping_ratio)
        return np.column_stack([turbulent, cyclic, quasistatic, inertial])

    def design_matrix(self, raw_basis: np.ndarray, gains: np.ndarray | None = None) -> np.ndarray:
        """Normalise, apply per-column gains, and raise to the Wohler power.

        `gains` has shape (n_conditions, N_BASIS) or (N_BASIS,); defaults to 1.
        """
        scaled = np.asarray(raw_basis, dtype=float) / self.basis_scale
        if gains is not None:
            scaled = scaled * np.asarray(gains, dtype=float)
        return np.maximum(scaled, 0.0) ** self.wohler_exponent

    # -- prediction ---------------------------------------------------------
    def predict_damage(self, design: np.ndarray) -> np.ndarray:
        """Predicted damage, shape (n_conditions, N_SECTIONS)."""
        return np.asarray(design, dtype=float) @ self.coefficients.T

    def path_shares(self, design: np.ndarray) -> np.ndarray:
        """Fraction of predicted damage from each load path.

        Shape (n_conditions, N_SECTIONS, N_BASIS).
        """
        design = np.asarray(design, dtype=float)
        contributions = design[:, None, :] * self.coefficients[None, :, :]
        total = contributions.sum(axis=-1, keepdims=True)
        return np.divide(contributions, total, out=np.zeros_like(contributions), where=total > 0)

    def controllable_share(self, design: np.ndarray) -> np.ndarray:
        """Fraction of predicted damage that rotor control can act on.

        Shape (n_conditions, N_SECTIONS). This is the ceiling on achievable
        fatigue relief: an agent cannot reduce wave-driven tower fatigue by
        pitching the blades or yawing the nacelle.
        """
        shares = self.path_shares(design)
        return shares[..., IDX_TURB] + shares[..., IDX_CYC]

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "tower": self.tower,
            "fa1_hz": self.fa1_hz,
            "damping_ratio": self.damping_ratio,
            "wohler_exponent": self.wohler_exponent,
            "basis_names": list(BASIS_NAMES),
            "basis_scale": self.basis_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "r2_damage_per_section": self.r2_damage_per_section.tolist(),
            "r2_del_per_section": self.r2_del_per_section.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TowerLoadModel":
        return cls(
            tower=payload["tower"],
            fa1_hz=payload["fa1_hz"],
            damping_ratio=payload["damping_ratio"],
            wohler_exponent=payload["wohler_exponent"],
            coefficients=np.asarray(payload["coefficients"], dtype=float),
            basis_scale=np.asarray(payload["basis_scale"], dtype=float),
            r2_damage_per_section=np.asarray(payload["r2_damage_per_section"], dtype=float),
            r2_del_per_section=np.asarray(payload["r2_del_per_section"], dtype=float),
        )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _fit_nonnegative(
    design: np.ndarray,
    damage: np.ndarray,
    wohler_exponent: float,
    weight_exponent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-section weighted non-negative least squares fit.

    Rows are weighted by damage^(-weight_exponent), floored at the 5th
    percentile so that near-zero damage conditions cannot dominate the normal
    equations.
    """
    from scipy.optimize import nnls

    n_sections = damage.shape[1]
    coefficients = np.zeros((n_sections, design.shape[1]), dtype=float)
    r2_damage = np.zeros(n_sections, dtype=float)
    r2_del = np.zeros(n_sections, dtype=float)

    for section in range(n_sections):
        target = damage[:, section]
        scale = float(np.mean(np.abs(target))) or 1.0
        target_scaled = target / scale

        if weight_exponent == 0.0:
            weights = np.ones_like(target_scaled)
        else:
            floor = max(float(np.percentile(target_scaled, 5)), 1e-12)
            weights = np.maximum(target_scaled, floor) ** (-weight_exponent)

        solution, _ = nnls(design * weights[:, None], target_scaled * weights)
        coefficients[section] = solution * scale

        predicted = design @ coefficients[section]
        r2_damage[section] = _r2(target, predicted)
        r2_del[section] = _r2(
            np.power(np.maximum(target, 0.0), 1.0 / wohler_exponent),
            np.power(np.maximum(predicted, 0.0), 1.0 / wohler_exponent),
        )

    return coefficients, r2_damage, r2_del


def calibrate_tower(
    tower_data: TowerData,
    aero: RotorAero,
    config: LoadModelConfig | None = None,
) -> tuple[TowerLoadModel, dict]:
    """Fit the load-path decomposition for one tower variant.

    The tower fore-aft damping ratio is selected by grid search on the mean
    per-section coefficient of determination in DEL space (FLOATBench's own
    regression target).
    """
    config = config or LoadModelConfig()
    tower = tower_data.tower
    if tower not in TOWER_FA1_HZ:
        raise ValueError(f"no fore-aft frequency known for tower {tower!r}")
    fa1 = TOWER_FA1_HZ[tower]

    conditions = tower_data.conditions
    damage = tower_data.damage
    turbulent, cyclic, _ = aerodynamic_amplitudes(conditions, aero)

    best: tuple[float, TowerLoadModel, float] | None = None
    sweep: list[dict] = []

    for damping in config.damping_ratio_grid:
        quasistatic, inertial = wave_load_amplitudes(conditions, fa1, damping)
        raw = np.column_stack([turbulent, cyclic, quasistatic, inertial])
        scale = np.mean(np.abs(raw), axis=0)
        scale[scale == 0] = 1.0

        for weight_exponent in config.weight_exponent_grid:
            model = TowerLoadModel(
                tower=tower,
                fa1_hz=fa1,
                damping_ratio=damping,
                wohler_exponent=config.wohler_exponent,
                coefficients=np.zeros((N_SECTIONS, N_BASIS)),
                basis_scale=scale,
            )
            design = model.design_matrix(raw)
            coefficients, r2_damage, r2_del = _fit_nonnegative(
                design, damage, config.wohler_exponent, weight_exponent
            )
            model.coefficients = coefficients
            model.r2_damage_per_section = r2_damage
            model.r2_del_per_section = r2_del

            mean_r2_del = float(np.nanmean(r2_del))
            sweep.append(
                {
                    "damping_ratio": damping,
                    "weight_exponent": weight_exponent,
                    "mean_r2_del": mean_r2_del,
                    "mean_r2_damage": float(np.nanmean(r2_damage)),
                }
            )
            if best is None or mean_r2_del > best[0]:
                best = (mean_r2_del, model, weight_exponent)

    assert best is not None
    _, model, selected_weight_exponent = best

    raw = model.raw_basis(conditions, aero)
    design = model.design_matrix(raw)
    shares = model.path_shares(design)
    controllable = model.controllable_share(design)
    predicted = model.predict_damage(design)

    report = {
        "tower": tower,
        "fa1_hz": fa1,
        "selected_damping_ratio": model.damping_ratio,
        "selected_weight_exponent": selected_weight_exponent,
        "hyperparameter_sweep": sweep,
        "wohler_exponent": model.wohler_exponent,
        "n_conditions": int(len(conditions)),
        "n_sections": int(N_SECTIONS),
        "fit_quality": {
            "mean_r2_del": float(np.nanmean(model.r2_del_per_section)),
            "min_r2_del": float(np.nanmin(model.r2_del_per_section)),
            "max_r2_del": float(np.nanmax(model.r2_del_per_section)),
            "mean_r2_damage": float(np.nanmean(model.r2_damage_per_section)),
            "r2_del_section_1_base": float(model.r2_del_per_section[0]),
            "r2_del_section_30_top": float(model.r2_del_per_section[-1]),
            "del_median_abs_rel_error": float(
                np.median(
                    np.abs(
                        np.power(np.maximum(predicted, 0), 1 / model.wohler_exponent)
                        - np.power(np.maximum(damage, 0), 1 / model.wohler_exponent)
                    )
                    / np.maximum(np.power(np.maximum(damage, 0), 1 / model.wohler_exponent), 1e-30)
                )
            ),
        },
        "path_share_mean": {
            name: float(np.mean(shares[..., index])) for index, name in enumerate(BASIS_NAMES)
        },
        "controllable_share": {
            "mean": float(np.mean(controllable)),
            "p05": float(np.percentile(controllable, 5)),
            "p50": float(np.percentile(controllable, 50)),
            "p95": float(np.percentile(controllable, 95)),
            "mean_section_1_base": float(np.mean(controllable[:, 0])),
            "mean_section_30_top": float(np.mean(controllable[:, -1])),
        },
        "basis_names": list(BASIS_NAMES),
        "basis_scale": model.basis_scale.tolist(),
    }
    return model, report


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def calibration_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "calibration" / "load_model.json"


def save_models(models: dict[str, TowerLoadModel], reports: dict[str, dict]) -> Path:
    path = calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": {tower: model.to_dict() for tower, model in models.items()},
        "reports": reports,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_models() -> tuple[dict[str, TowerLoadModel], dict[str, dict]]:
    path = calibration_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m fowt_rl.build_dataset --calibrate")
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = {tower: TowerLoadModel.from_dict(entry) for tower, entry in payload["models"].items()}
    return models, payload.get("reports", {})
