"""Top-level pipeline configuration.

Composes the sub-configurations from the individual modules so that a whole
dataset build is described by one serialisable object. Every generated artefact
is written alongside a JSON dump of this config, so any dataset can be traced
back to the exact settings that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

from .actions import ActionSpace, DutyCostConfig
from .damage import ControlAuthorityConfig
from .iot import IoTConfig
from .load_model import LoadModelConfig
from .turbine import TOWERS


@dataclass
class EpisodeConfig:
    """How synthetic operating episodes are drawn from the FLOATBench grid."""

    # FLOATBench conditions are 10-minute stationary OpenFAST runs, so one
    # control step is 600 s and an episode of 36 steps spans six hours of
    # slowly evolving metocean conditions.
    step_seconds: float = 600.0
    steps_per_episode: int = 36
    episodes_per_tower: int = 1200

    # Random walk over the (wind_speed_id, wave_hs_id, wave_tp_id) grid.
    # `persistence` is the probability of staying on the current index.
    grid_persistence: float = 0.55

    # Ornstein-Uhlenbeck inflow direction process.
    direction_std_deg: float = 12.0
    direction_correlation_hours: float = 2.0
    direction_clip_deg: float = 45.0

    # Static yaw misalignment from vane miscalibration, drawn per episode.
    # A few degrees of standing misalignment is common on real turbines.
    vane_bias_std_deg: float = 2.5

    # Behaviour policy mixture used to populate the dataset. Weights need not
    # sum to one; they are normalised.
    policy_weights: dict[str, float] = field(
        default_factory=lambda: {
            "baseline": 0.20,  # zero action - the uncontrolled reference
            "random": 0.40,  # uniform exploration of the action space
            "ipc_only": 0.15,  # IPC on, no pitch or yaw offset
            "feather": 0.15,  # moderate collective feathering plus IPC
            "yaw_seeker": 0.10,  # deliberate yaw misalignment
        }
    )

    seed: int = 20260826


@dataclass
class RewardConfig:
    """Reward weights.

    Every term is dimensionless and bounded, so the weights are directly
    comparable:

        fatigue_relief      = 1 - DEL_controlled / DEL_reference     in (-inf, 1]
        severity            = lifetime fatigue significance of this
                              condition, normalised to ~1 at the p90
                              condition                              in [0, 2]
        power_loss_fraction = (P_reference - P) / P_rated             in [0, 1]
        duty_total          = normalised actuator activity            in [0, 1]

        reward = w_fatigue * fatigue_relief * severity
               - w_power   * power_loss_fraction
               - w_duty    * duty_total

    Why severity weighting
    ----------------------
    Damage varies by roughly six orders of magnitude across the FLOATBench
    envelope. Without weighting, a 10% DEL reduction in a benign sea state would
    score the same as a 10% reduction in the storm conditions that actually
    consume tower life, and the optimal policy would degenerate to a single
    fixed action. `severity` is built from FLOATBench's own `damage_weight`
    column - the 25-year probability of occurrence of each condition - so
    fatigue relief is rewarded in proportion to the lifetime damage it actually
    avoids. This is what makes the problem state-dependent, and therefore worth
    solving with a policy rather than a constant.

    The reference point
    -------------------
    The reference is the *aerodynamically ideal* baseline: zero pitch offset,
    zero IPC and perfect yaw alignment. An uncontrolled real turbine scores
    slightly below zero against it, because vane bias and the 8 deg yaw deadband
    leave a standing misalignment. Correcting that misalignment is therefore
    itself a source of positive reward, which mirrors one of the genuine
    value-adds of active yaw control.
    """

    fatigue_weight: float = 2.0
    power_weight: float = 1.0
    duty_weight: float = 0.05

    # Scale fatigue relief by the lifetime significance of the condition.
    severity_weighting: bool = True
    # Percentile of the lifetime-weighted baseline DEL that maps to severity 1.
    severity_reference_percentile: float = 90.0
    severity_clip: float = 2.0

    # Which cross-section drives the fatigue term. "max" uses the governing
    # (worst) section, which is what sets tower fatigue life.
    fatigue_section: str = "max"

    def __post_init__(self) -> None:
        allowed = {"max", "base", "top", "mean", "sum"}
        if self.fatigue_section not in allowed:
            raise ValueError(f"fatigue_section must be one of {sorted(allowed)}")


@dataclass
class SweepConfig:
    """Full-factorial action sweep used for supervised surrogate training."""

    enabled: bool = True
    n_pitch: int = 5
    n_yaw: int = 5
    n_ipc: int = 3
    # Fraction of FLOATBench conditions to include, to keep file sizes sane.
    condition_fraction: float = 1.0
    seed: int = 20260826


@dataclass
class PipelineConfig:
    """Everything needed to reproduce a dataset build."""

    towers: tuple[str, ...] = TOWERS
    load_model: LoadModelConfig = field(default_factory=LoadModelConfig)
    authority: ControlAuthorityConfig = field(default_factory=ControlAuthorityConfig)
    action_space: ActionSpace = field(default_factory=ActionSpace)
    duty: DutyCostConfig = field(default_factory=DutyCostConfig)
    episodes: EpisodeConfig = field(default_factory=EpisodeConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    iot: IoTConfig = field(default_factory=IoTConfig)

    def to_dict(self) -> dict:
        def convert(value):
            if is_dataclass(value) and not isinstance(value, type):
                return {key: convert(item) for key, item in asdict(value).items()}
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        payload = {
            "towers": list(self.towers),
            "load_model": convert(self.load_model),
            "authority": convert(self.authority),
            "action_space": convert(self.action_space),
            "duty": convert(self.duty),
            "episodes": convert(self.episodes),
            "reward": convert(self.reward),
            "sweep": convert(self.sweep),
            "iot": {
                "enabled": self.iot.enabled,
                "severity": self.iot.severity,
                "dropout_scale": self.iot.dropout_scale,
                "seed": self.iot.seed,
                "channels": self.iot.describe(),
            },
        }
        return payload

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
