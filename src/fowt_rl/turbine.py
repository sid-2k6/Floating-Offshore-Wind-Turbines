"""IEA-22-280-RWT reference turbine properties and baseline operating schedule.

Everything in this module is *measured* reference data taken from the official
IEA Wind Task 55 turbine definition repository - no modelling assumptions are
introduced here. The modelling layer lives in `fowt_rl.aero`.

Data provenance
---------------
`data/turbine/iea22_steady_states.csv`  : OpenFAST steady-state operating points
`data/turbine/iea22_properties.json`    : windIO assembly / control properties
Both produced by `scripts/fetch_turbine_data.py` from
https://github.com/IEAWindSystems/IEA-22-280-RWT (Apache-2.0).

Tower fore-aft natural frequencies for the three FLOATBench tower variants are
taken from Table 4 of the FLOATBench paper (arXiv:2605.25717, CC-BY-4.0) and are
used by the wave-load dynamic amplification term in `fowt_rl.load_model`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# FLOATBench tower variants
# --------------------------------------------------------------------------
# First fore-aft natural frequency [Hz], FLOATBench paper Table 4.
TOWER_FA1_HZ: dict[str, float] = {
    "ref": 0.336,
    "opt1": 0.573,
    "opt2": 0.537,
}

# Tower height is identical across the three variants (FLOATBench Section 3).
TOWER_HEIGHT_M: float = 148.385

TOWERS: tuple[str, ...] = ("ref", "opt1", "opt2")


@dataclass(frozen=True)
class TurbineProperties:
    """Scalar reference-turbine properties (all from windIO)."""

    rotor_diameter_m: float
    hub_height_m: float
    rated_power_w: float
    rated_rotor_speed_rpm: float
    optimal_tsr: float
    cut_in_wind_speed_ms: float
    cut_out_wind_speed_ms: float
    air_density_kgm3: float
    rotor_area_m2: float

    @property
    def rotor_radius_m(self) -> float:
        return self.rotor_diameter_m / 2.0

    def dynamic_pressure_area(self, wind_speed):
        """0.5 * rho * A  (so that thrust = this * V^2 * Ct)."""
        return 0.5 * self.air_density_kgm3 * self.rotor_area_m2 * np.asarray(wind_speed) ** 0.0


class BaselineSchedule:
    """Baseline (uncontrolled) operating schedule as a function of wind speed.

    Wraps the 22 official OpenFAST steady-state operating points and exposes
    monotone linear interpolation, clipped to the cut-in / cut-out envelope.
    """

    _COLUMNS = (
        "rotor_speed",
        "pitch",
        "tip_speed_ratio",
        "mechanical_power",
        "electrical_power",
        "rotor_thrust",
        "ct_base",
        "cp_base",
    )

    def __init__(self, table: pd.DataFrame):
        table = table.sort_values("wind_speed").reset_index(drop=True)
        missing = set(("wind_speed",) + self._COLUMNS) - set(table.columns)
        if missing:
            raise ValueError(f"steady-state table missing columns: {sorted(missing)}")
        self.table = table
        self._v = table["wind_speed"].to_numpy(dtype=float)

    def __len__(self) -> int:
        return len(self.table)

    def _interp(self, column: str, wind_speed):
        y = self.table[column].to_numpy(dtype=float)
        return np.interp(np.asarray(wind_speed, dtype=float), self._v, y)

    def pitch_deg(self, wind_speed):
        """Baseline collective blade pitch [deg]."""
        return self._interp("pitch", wind_speed)

    def rotor_speed_rpm(self, wind_speed):
        return self._interp("rotor_speed", wind_speed)

    def tip_speed_ratio(self, wind_speed):
        return self._interp("tip_speed_ratio", wind_speed)

    def thrust_n(self, wind_speed):
        return self._interp("rotor_thrust", wind_speed)

    def mechanical_power_w(self, wind_speed):
        return self._interp("mechanical_power", wind_speed)

    def electrical_power_w(self, wind_speed):
        return self._interp("electrical_power", wind_speed)

    def ct(self, wind_speed):
        return self._interp("ct_base", wind_speed)

    def cp(self, wind_speed):
        return self._interp("cp_base", wind_speed)

    @property
    def rated_wind_speed_ms(self) -> float:
        """Lowest wind speed at which electrical power is within 0.5% of its maximum."""
        power = self.table["electrical_power"].to_numpy(dtype=float)
        target = 0.995 * power.max()
        idx = int(np.argmax(power >= target))
        return float(self._v[idx])


def _default_data_dir() -> Path:
    # src/fowt_rl/turbine.py -> repo root
    return Path(__file__).resolve().parents[2] / "data" / "turbine"


@lru_cache(maxsize=4)
def load_turbine(data_dir: str | None = None) -> tuple[TurbineProperties, BaselineSchedule]:
    """Load reference-turbine properties and the baseline operating schedule."""
    directory = Path(data_dir) if data_dir else _default_data_dir()
    props_path = directory / "iea22_properties.json"
    steady_path = directory / "iea22_steady_states.csv"
    if not props_path.exists() or not steady_path.exists():
        raise FileNotFoundError(
            f"turbine reference data not found in {directory}. "
            "Run: python scripts/fetch_turbine_data.py"
        )
    raw = json.loads(props_path.read_text(encoding="utf-8"))
    props = TurbineProperties(
        rotor_diameter_m=raw["rotor_diameter_m"],
        hub_height_m=raw["hub_height_m"],
        rated_power_w=raw["rated_power_w"],
        rated_rotor_speed_rpm=raw["rated_rotor_speed_rpm"],
        optimal_tsr=raw["optimal_tsr"],
        cut_in_wind_speed_ms=raw["cut_in_wind_speed_ms"],
        cut_out_wind_speed_ms=raw["cut_out_wind_speed_ms"],
        air_density_kgm3=raw["air_density_kgm3"],
        rotor_area_m2=raw["rotor_area_m2"],
    )
    schedule = BaselineSchedule(pd.read_csv(steady_path))
    return props, schedule
