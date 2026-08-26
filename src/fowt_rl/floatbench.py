"""FLOATBench dataset loading and reshaping.

FLOATBench (arXiv:2605.25717, CC-BY-4.0, hosted at `DeCoDELab/FLOATBench`)
ships one row per (simulation, tower cross-section):

    sim_id, wind_speed_id, wind_speed, mean_wind_speed, std_wind_speed,
    wave_hs_id, wave_hs, wave_tp_id, wave_tp, wind_seed_id,
    section_id, section_height_m, section_radius_m, section_thickness_m,
    damage_weight, damage

Each tower variant (`ref`, `opt1`, `opt2`) contributes 194,040 rows
= 6,468 ten-minute OpenFAST simulations x 30 tower sections.

An RL agent controls the *turbine*, not an individual cross-section, so this
module reshapes the data into

    conditions : one row per simulation (6,468 rows) - the environment state
    damage     : dense (n_simulations, 30) matrix of per-section fatigue damage

which is the layout the rest of the pipeline expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .turbine import TOWERS

HF_REPO_ID = "DeCoDELab/FLOATBench"

CONDITION_COLUMNS = [
    "sim_id",
    "wind_speed_id",
    "wind_speed",
    "mean_wind_speed",
    "std_wind_speed",
    "wave_hs_id",
    "wave_hs",
    "wave_tp_id",
    "wave_tp",
    "wind_seed_id",
    "damage_weight",
]

SECTION_COLUMNS = ["section_id", "section_height_m", "section_radius_m", "section_thickness_m"]

N_SECTIONS = 30


@dataclass
class TowerData:
    """FLOATBench data for one tower variant, reshaped for RL use."""

    tower: str
    conditions: pd.DataFrame  # (n_simulations, len(CONDITION_COLUMNS))
    damage: np.ndarray  # (n_simulations, N_SECTIONS)
    sections: pd.DataFrame  # (N_SECTIONS, len(SECTION_COLUMNS)) - geometry per section

    @property
    def n_simulations(self) -> int:
        return len(self.conditions)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TowerData(tower={self.tower!r}, n_simulations={self.n_simulations}, "
            f"damage={self.damage.shape})"
        )


def default_raw_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "floatbench"


def download(raw_dir: Path | None = None, towers: tuple[str, ...] = TOWERS) -> Path:
    """Download the FLOATBench `data.csv` files from Hugging Face."""
    from huggingface_hub import hf_hub_download

    raw_dir = raw_dir or default_raw_dir()
    for tower in towers:
        target = raw_dir / tower / "data.csv"
        if target.exists() and target.stat().st_size > 0:
            print(f"[cache] {target}")
            continue
        cached = hf_hub_download(HF_REPO_ID, f"{tower}/data.csv", repo_type="dataset")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(cached).read_bytes())
        print(f"[get]   {target}  ({target.stat().st_size / 1e6:.1f} MB)")
    return raw_dir


def load_tower(tower: str, raw_dir: Path | None = None) -> TowerData:
    """Load one tower variant and reshape it into conditions + damage matrix."""
    if tower not in TOWERS:
        raise ValueError(f"unknown tower {tower!r}; expected one of {TOWERS}")
    raw_dir = raw_dir or default_raw_dir()
    path = raw_dir / tower / "data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_floatbench.py"
        )

    frame = pd.read_csv(path)
    expected = set(CONDITION_COLUMNS + SECTION_COLUMNS + ["damage"])
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    frame = frame.sort_values(["sim_id", "section_id"], kind="stable")

    section_ids = np.sort(frame["section_id"].unique())
    if section_ids.size != N_SECTIONS:
        raise ValueError(f"expected {N_SECTIONS} sections, found {section_ids.size}")

    damage_wide = frame.pivot(index="sim_id", columns="section_id", values="damage")
    damage_wide = damage_wide.reindex(columns=section_ids)
    if damage_wide.isna().to_numpy().any():
        raise ValueError("damage matrix has missing entries - dataset is incomplete")

    conditions = (
        frame.drop_duplicates(subset="sim_id")[CONDITION_COLUMNS]
        .set_index("sim_id")
        .reindex(damage_wide.index)
        .reset_index()
    )

    sections = (
        frame.drop_duplicates(subset="section_id")[SECTION_COLUMNS]
        .sort_values("section_id")
        .reset_index(drop=True)
    )

    return TowerData(
        tower=tower,
        conditions=conditions,
        damage=damage_wide.to_numpy(dtype=float),
        sections=sections,
    )


def load_all(raw_dir: Path | None = None, towers: tuple[str, ...] = TOWERS) -> dict[str, TowerData]:
    return {tower: load_tower(tower, raw_dir) for tower in towers}
