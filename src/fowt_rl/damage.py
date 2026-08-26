"""Damage under control action - ratio-anchored on real FLOATBench values.

Given a calibrated `TowerLoadModel`, the damage at section s under a control
action follows in closed form. Writing c_{s,k} for the fitted per-section
coefficients, A_k for the baseline load amplitudes and g_k for the gain that the
action applies to load path k:

    D_controlled(s)      sum_k c_{s,k} (g_k A_k)^m
    ---------------  =  ---------------------------
    D_baseline(s)         sum_k c_{s,k} A_k^m

and the *absolute* damage is obtained by multiplying that ratio onto the real
FLOATBench damage value:

    D_out(s) = D_floatbench(s) * ratio(s)

Two properties follow, and they are the reason the pipeline is built this way:

1. **Zero action is exact.** With no control action every g_k = 1, the ratio is
   identically 1, and `D_out` is the unmodified FLOATBench damage. The synthetic
   layer adds information without overwriting measured data.
2. **Control authority is bounded by physics.** Because the wave gains are fixed
   at 1, no action can reduce damage below the wave-driven floor. The maximum
   achievable relief at a condition equals the controllable share estimated in
   `fowt_rl.load_model`.

Gains
-----
    g_turb    = ct_ratio
    g_cyc     = ct_ratio * (1 + yaw_cyclic_gain*|sin gamma|) * (1 - ipc_authority*kappa)
    g_wave_qs = 1
    g_wave_in = 1

`ct_ratio` comes from the official rotor performance surface and already
contains the yaw cos^2 thrust reduction. The extra `(1 + yaw_cyclic_gain*|sin
gamma|)` factor captures the opposite effect on *cyclic* loading: yawing out of
the wind reduces mean thrust but increases once-per-revolution load variation,
because each blade now sweeps through a non-axisymmetric inflow. `ipc_authority`
is the fraction of cyclic load amplitude that fully-active IPC can cancel.

These last two coefficients are the only parameters in the whole pipeline that
are neither measured nor fitted; they are literature-informed defaults and are
called out in docs/LIMITATIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .aero import RotorAero
from .load_model import IDX_CYC, IDX_TURB, N_BASIS, TowerLoadModel


@dataclass(frozen=True)
class ControlAuthorityConfig:
    """Literature-informed coefficients for the two control paths not in the data.

    yaw_cyclic_gain
        Increase in cyclic rotor load amplitude per unit |sin(yaw error)|. The
        default of 1.5 gives a 1.75x cyclic amplitude at 30 deg misalignment,
        within the range reported for blade-root cyclic load growth under large
        yaw error.
    ipc_authority
        Fraction of cyclic load amplitude cancelled by fully active IPC. The
        default of 0.30 sits at the optimistic end of reported 1P blade-load
        reductions; because IPC acts only on the cyclic column (roughly a
        quarter of total tower damage here), the net effect on tower fatigue is
        much smaller than 30%.
    """

    yaw_cyclic_gain: float = 1.5
    ipc_authority: float = 0.30

    def __post_init__(self) -> None:
        if not 0.0 <= self.ipc_authority <= 1.0:
            raise ValueError("ipc_authority must lie in [0, 1]")
        if self.yaw_cyclic_gain < 0.0:
            raise ValueError("yaw_cyclic_gain must be non-negative")


def load_path_gains(
    ct_ratio: np.ndarray,
    yaw_error_deg: np.ndarray,
    ipc_level: np.ndarray,
    config: ControlAuthorityConfig | None = None,
) -> np.ndarray:
    """Per-load-path gains for a control action, shape (n, N_BASIS)."""
    config = config or ControlAuthorityConfig()
    ct_ratio = np.asarray(ct_ratio, dtype=float)
    yaw_error_deg = np.asarray(yaw_error_deg, dtype=float)
    ipc_level = np.clip(np.asarray(ipc_level, dtype=float), 0.0, 1.0)

    cyclic_penalty = 1.0 + config.yaw_cyclic_gain * np.abs(
        np.sin(np.radians(np.clip(yaw_error_deg, -89.0, 89.0)))
    )
    ipc_relief = 1.0 - config.ipc_authority * ipc_level

    gains = np.ones((ct_ratio.size, N_BASIS), dtype=float)
    gains[:, IDX_TURB] = ct_ratio
    gains[:, IDX_CYC] = ct_ratio * cyclic_penalty * ipc_relief
    return gains


def damage_under_action(
    conditions: pd.DataFrame,
    baseline_damage: np.ndarray,
    load_model: TowerLoadModel,
    aero: RotorAero,
    pitch_offset_deg,
    yaw_error_deg,
    ipc_level,
    authority: ControlAuthorityConfig | None = None,
) -> dict:
    """Damage and rotor state under a control action.

    Parameters
    ----------
    conditions : DataFrame
        One row per environmental condition (FLOATBench condition columns).
    baseline_damage : ndarray, shape (n_conditions, n_sections)
        The real FLOATBench damage values for those conditions.
    pitch_offset_deg, yaw_error_deg, ipc_level : array-like, shape (n_conditions,)
        The control action. `yaw_error_deg` is the *residual* misalignment after
        the yaw command, not the command itself.

    Returns
    -------
    dict
        damage                : (n, n_sections) damage under the action
        damage_ratio          : (n, n_sections) damage / baseline damage
        controllable_share    : (n, n_sections) baseline controllable fraction
        power_w, thrust_n     : (n,) rotor response
        ct_ratio, cp_ratio    : (n,) coefficient ratios vs baseline
        gains                 : (n, N_BASIS) per-load-path gains applied
    """
    pitch_offset_deg = np.asarray(pitch_offset_deg, dtype=float)
    yaw_error_deg = np.asarray(yaw_error_deg, dtype=float)
    ipc_level = np.asarray(ipc_level, dtype=float)

    mean_wind = conditions["mean_wind_speed"].to_numpy(dtype=float)
    response = aero.response(mean_wind, pitch_offset_deg, yaw_error_deg)

    gains = load_path_gains(response["ct_ratio"], yaw_error_deg, ipc_level, authority)

    raw = load_model.raw_basis(conditions, aero)  # baseline amplitudes
    design_base = load_model.design_matrix(raw)
    design_act = load_model.design_matrix(raw, gains)

    predicted_base = load_model.predict_damage(design_base)
    predicted_act = load_model.predict_damage(design_act)

    ratio = np.divide(
        predicted_act,
        predicted_base,
        out=np.ones_like(predicted_act),
        where=predicted_base > 0,
    )
    damage = np.asarray(baseline_damage, dtype=float) * ratio

    return {
        "damage": damage,
        "damage_ratio": ratio,
        "controllable_share": load_model.controllable_share(design_base),
        "power_w": response["electrical_power_w"],
        "power_base_w": response["power_base_w"],
        "thrust_n": response["thrust_n"],
        "thrust_base_n": response["thrust_base_n"],
        "ct_ratio": response["ct_ratio"],
        "cp_ratio": response["cp_ratio"],
        "gains": gains,
    }


def summarise_sections(damage: np.ndarray) -> dict[str, np.ndarray]:
    """Reduce per-section damage to the scalars used by the reward and state.

    `damage_max` is the design-driving quantity: fatigue life is set by the
    worst cross-section, not the average one. `damage_base` and `damage_top`
    are reported separately because FLOATBench shows base and top sections
    behave differently, and `damage_sum` is a proxy for whole-tower damage.
    """
    damage = np.asarray(damage, dtype=float)
    return {
        "damage_max": damage.max(axis=1),
        "damage_argmax_section": damage.argmax(axis=1) + 1,
        "damage_base": damage[:, 0],
        "damage_top": damage[:, -1],
        "damage_mean": damage.mean(axis=1),
        "damage_sum": damage.sum(axis=1),
    }
