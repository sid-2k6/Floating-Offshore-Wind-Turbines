"""Environments for training and evaluating load-relief policies.

Two are provided:

`FowtLoadReliefEnv`
    An *online* environment. It steps the same calibrated physics used to
    generate the published dataset, so a policy can be trained interactively
    without regenerating parquet files. Gymnasium-compatible when Gymnasium is
    installed, and duck-typed (`reset`/`step`) otherwise.

`OfflineTransitionDataset`
    A thin reader over the published transitions parquet for offline RL, with
    the observation/action/reward column groups already assembled and the
    episode boundaries exposed.

Both use the measured (IoT) observation by default, because that is what a real
controller would see. `FowtLoadReliefEnv` applies the sensor error model
step-by-step; ground truth remains available in the `info` dict for evaluation
and ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .actions import ActionSpace, DutyCostConfig, actuator_duty
from .aero import RotorAero, load_aero
from .config import EpisodeConfig, PipelineConfig, RewardConfig
from .damage import ControlAuthorityConfig, damage_under_action, summarise_sections
from .floatbench import TowerData, load_tower
from .iot import IoTConfig
from .load_model import TowerLoadModel, load_models
from .mdp import GRID_SHAPE, N_SEEDS, OBSERVATION_CHANNELS, PROPRIOCEPTIVE_COLUMNS, build_condition_index

OBSERVATION_ORDER: tuple[str, ...] = tuple(
    [f"meas_{channel}" for channel in OBSERVATION_CHANNELS]
    + [f"valid_{channel}" for channel in OBSERVATION_CHANNELS]
    + list(PROPRIOCEPTIVE_COLUMNS)
)


def _normalise(value: float, scale: float) -> float:
    return float(value / scale) if scale else float(value)


class FowtLoadReliefEnv:
    """Single-turbine tower load-relief environment.

    Observation (float32, length 23)
        8 measured channels, 8 validity flags, 7 proprioceptive states.
        Channels are scaled to roughly unit magnitude; see `observation_names`.

    Action (float32, length 3)
        [pitch_offset_deg, yaw_setpoint_deg, ipc_level], clipped and rate
        limited exactly as in the dataset builder.

    Reward
        As documented in `fowt_rl.config.RewardConfig`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        tower: str = "opt2",
        config: PipelineConfig | None = None,
        tower_data: TowerData | None = None,
        load_model: TowerLoadModel | None = None,
        aero: RotorAero | None = None,
        seed: int | None = None,
    ):
        config = config or PipelineConfig()
        self.config = config
        self.tower = tower
        self.episode_config: EpisodeConfig = config.episodes
        self.reward_config: RewardConfig = config.reward
        self.space: ActionSpace = config.action_space
        self.duty_config: DutyCostConfig = config.duty
        self.authority: ControlAuthorityConfig = config.authority
        self.iot: IoTConfig = config.iot

        self.tower_data = tower_data or load_tower(tower)
        if load_model is None:
            models, _ = load_models()
            load_model = models[tower]
        self.load_model = load_model
        self.aero = aero or load_aero(write_report=False)

        self._index = build_condition_index(self.tower_data)
        self._conditions = self.tower_data.conditions
        self._damage = self.tower_data.damage

        baseline_max = self._damage.max(axis=1)
        self._damage_scale = float(np.percentile(baseline_max, 99)) * self.episode_config.steps_per_episode or 1.0
        lifetime_del = np.power(
            np.maximum(baseline_max * self._conditions["damage_weight"].to_numpy(dtype=float), 0.0),
            1.0 / self.load_model.wohler_exponent,
        )
        self._severity_reference = (
            float(np.percentile(lifetime_del, self.reward_config.severity_reference_percentile)) or 1.0
        )

        self._rng = np.random.default_rng(seed if seed is not None else self.episode_config.seed)
        self._sensor_rng = np.random.default_rng(self.iot.seed)
        self._channels = {channel.name: channel for channel in self.iot.scaled_channels()}

        self._state: dict = {}
        self.observation_names = OBSERVATION_ORDER

        # Optional Gymnasium spaces.
        try:  # pragma: no cover - optional dependency
            import gymnasium as gym

            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(OBSERVATION_ORDER),), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=self.space.low.astype(np.float32),
                high=self.space.high.astype(np.float32),
                dtype=np.float32,
            )
        except ImportError:
            self.observation_space = None
            self.action_space = None

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.episode_config

        coordinate = np.array([self._rng.integers(0, size) for size in GRID_SHAPE], dtype=np.int64)
        direction = float(self._rng.normal(0.0, cfg.direction_std_deg))
        self._state = {
            "step": 0,
            "coordinate": coordinate,
            "direction": np.clip(direction, -cfg.direction_clip_deg, cfg.direction_clip_deg),
            "vane_bias": float(self._rng.normal(0.0, cfg.vane_bias_std_deg)),
            "previous_action": np.zeros(3, dtype=float),
            "nacelle_yaw": direction,
            "cumulative_damage": 0.0,
            "last_true": None,
            "held": {},
        }
        observation, info = self._observe(action_result=None)
        return observation, info

    # ------------------------------------------------------------------
    def step(self, action):
        if not self._state:
            raise RuntimeError("call reset() before step()")
        cfg = self.episode_config
        state = self._state

        requested = self.space.clip(np.asarray(action, dtype=float).reshape(3))
        applied = self.space.apply_rate_limits(state["previous_action"], requested)

        inflow = state["direction"]
        measured_direction = inflow + state["vane_bias"]

        target = measured_direction - applied[1]
        delta = target - state["nacelle_yaw"]
        if abs(delta) > self.space.limits.yaw_deadband_deg:
            travel = np.sign(delta) * min(
                abs(delta), self.space.limits.yaw_rate_deg_s * cfg.step_seconds
            )
        else:
            travel = 0.0
        nacelle_yaw = state["nacelle_yaw"] + travel
        yaw_error = inflow - nacelle_yaw

        row = int(
            self._index[
                state["coordinate"][0], state["coordinate"][1], state["coordinate"][2],
                int(self._rng.integers(0, N_SEEDS)),
            ]
        )
        conditions = self._conditions.iloc[[row]].reset_index(drop=True)
        damage_row = self._damage[[row]]

        controlled = damage_under_action(
            conditions, damage_row, self.load_model, self.aero,
            np.array([applied[0]]), np.array([yaw_error]), np.array([applied[2]]), self.authority,
        )
        baseline = damage_under_action(
            conditions, damage_row, self.load_model, self.aero,
            np.zeros(1), np.zeros(1), np.zeros(1), self.authority,
        )
        key = f"damage_{self.reward_config.fatigue_section}"
        damage_controlled = float(summarise_sections(controlled["damage"])[key][0])
        damage_baseline = float(summarise_sections(baseline["damage"])[key][0])

        del_ratio = (
            (damage_controlled / damage_baseline) ** (1.0 / self.load_model.wohler_exponent)
            if damage_baseline > 0
            else 1.0
        )
        fatigue_relief = 1.0 - del_ratio

        if self.reward_config.severity_weighting:
            weight = float(conditions["damage_weight"].iloc[0])
            lifetime_del = max(damage_baseline * weight, 0.0) ** (1.0 / self.load_model.wohler_exponent)
            severity = float(
                np.clip(lifetime_del / self._severity_reference, 0.0, self.reward_config.severity_clip)
            )
        else:
            severity = 1.0

        power = float(controlled["power_w"][0])
        power_base = float(controlled["power_base_w"][0])
        power_loss_fraction = float(
            np.clip((power_base - power) / self.aero.properties.rated_power_w, 0.0, 1.0)
        )
        duty = actuator_duty(applied, state["previous_action"], self.space, cfg.step_seconds, self.duty_config)
        duty_total = float(duty["duty_total"][0])

        reward = (
            self.reward_config.fatigue_weight * fatigue_relief * severity
            - self.reward_config.power_weight * power_loss_fraction
            - self.reward_config.duty_weight * duty_total
        )

        mean_wind = float(conditions["mean_wind_speed"].iloc[0])
        turbulence_std = float(conditions["std_wind_speed"].iloc[0])
        true_channels = {
            "wind_speed": mean_wind,
            "turbulence_std": turbulence_std,
            "wind_direction": measured_direction,
            "wave_hs": float(conditions["wave_hs"].iloc[0]),
            "wave_tp": float(conditions["wave_tp"].iloc[0]),
            "thrust": float(controlled["thrust_n"][0]),
            "power": power,
            "tower_damage_rate": damage_controlled,
        }

        # advance the environment
        state["cumulative_damage"] += damage_controlled
        state["previous_action"] = applied
        state["nacelle_yaw"] = nacelle_yaw
        state["last_yaw_error"] = yaw_error
        state["last_true"] = true_channels
        state["step"] += 1

        decay = float(np.exp(-(cfg.step_seconds / 3600.0) / max(cfg.direction_correlation_hours, 1e-6)))
        innovation = cfg.direction_std_deg * np.sqrt(max(1.0 - decay**2, 0.0))
        state["direction"] = float(
            np.clip(
                decay * inflow + self._rng.normal(0.0, innovation),
                -cfg.direction_clip_deg,
                cfg.direction_clip_deg,
            )
        )
        increment = self._rng.choice(
            [-1, 0, 1],
            size=3,
            p=[(1 - cfg.grid_persistence) / 2, cfg.grid_persistence, (1 - cfg.grid_persistence) / 2],
        )
        state["coordinate"] = np.clip(
            state["coordinate"] + increment, 0, np.array(GRID_SHAPE) - 1
        )

        terminated = False
        truncated = state["step"] >= cfg.steps_per_episode
        observation, info = self._observe(
            action_result={
                "applied_action": applied,
                "requested_action": requested,
                "yaw_error_deg": yaw_error,
                "damage_controlled": damage_controlled,
                "damage_baseline": damage_baseline,
                "del_ratio": del_ratio,
                "fatigue_relief": fatigue_relief,
                "severity": severity,
                "power_w": power,
                "power_baseline_w": power_base,
                "power_loss_fraction": power_loss_fraction,
                "duty_total": duty_total,
                "controllable_share": float(controlled["controllable_share"].max()),
                "true_channels": true_channels,
            }
        )
        return observation, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    def _measure(self, name: str, value: float) -> tuple[float, int]:
        """Apply the sensor error model to one scalar reading."""
        channel = self._channels.get(name)
        if channel is None or not self.iot.enabled:
            return float(value), 1
        rng = self._sensor_rng
        noise_scale = float(
            np.sqrt(channel.noise_std**2 + (channel.noise_relative * abs(value)) ** 2)
        )
        measured = value + (rng.normal(0.0, noise_scale) if noise_scale > 0 else 0.0)
        if channel.bias_std > 0:
            measured += rng.normal(0.0, channel.bias_std)
        if channel.quantisation > 0:
            measured = round(measured / channel.quantisation) * channel.quantisation
        if channel.non_negative:
            measured = max(measured, 0.0)
        valid = 1
        if channel.dropout_probability > 0 and rng.random() < channel.dropout_probability:
            valid = 0
            measured = self._state["held"].get(name, measured)
        self._state["held"][name] = measured
        return float(measured), valid

    def _observe(self, action_result: dict | None):
        state = self._state
        truth = state.get("last_true")
        if truth is None:
            # Before the first step the plant has not been evaluated yet; report
            # the environmental channels only, with zeros for rotor state.
            truth = {
                "wind_speed": 0.0,
                "turbulence_std": 0.0,
                "wind_direction": state["direction"] + state["vane_bias"],
                "wave_hs": 0.0,
                "wave_tp": 0.0,
                "thrust": 0.0,
                "power": 0.0,
                "tower_damage_rate": 0.0,
            }

        scales = {
            "wind_speed": 25.0,
            "turbulence_std": 4.0,
            "wind_direction": 45.0,
            "wave_hs": 9.0,
            "wave_tp": 17.0,
            "thrust": 3.0e6,
            "power": self.aero.properties.rated_power_w,
            "tower_damage_rate": max(self._damage.max(), 1e-30),
        }

        values: list[float] = []
        validity: list[float] = []
        measured_report: dict[str, float] = {}
        for channel in OBSERVATION_CHANNELS:
            measured, valid = self._measure(channel, truth[channel])
            measured_report[channel] = measured
            values.append(_normalise(measured, scales[channel]))
            validity.append(float(valid))

        previous = state["previous_action"]
        proprioceptive = [
            previous[0] / max(self.space.pitch_offset_bounds[1], 1e-9),
            previous[1] / max(self.space.yaw_setpoint_bounds[1], 1e-9),
            previous[2],
            state["nacelle_yaw"] / 45.0,
            state.get("last_yaw_error", 0.0) / 45.0,
            state["cumulative_damage"] / self._damage_scale,
            state["step"] / max(self.episode_config.steps_per_episode - 1, 1),
        ]

        observation = np.asarray(values + validity + proprioceptive, dtype=np.float32)
        info = {
            "step": state["step"],
            "true_channels": truth,
            "measured_channels": measured_report,
            "sensor_health": float(np.mean(validity)),
            "tower": self.tower,
        }
        if action_result is not None:
            info.update(action_result)
        return observation, info


# ---------------------------------------------------------------------------
@dataclass
class OfflineTransitionDataset:
    """Reader over the published transitions parquet, for offline RL."""

    frame: pd.DataFrame

    @classmethod
    def load(cls, path: Path | str) -> "OfflineTransitionDataset":
        path = Path(path)
        if path.is_dir():
            parts = sorted(path.glob("transitions_*.parquet"))
            if not parts:
                raise FileNotFoundError(f"no transitions parquet files in {path}")
            frame = pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True)
        else:
            frame = pd.read_parquet(path)
        return cls(frame=frame.sort_values(["tower", "episode_id", "step"]).reset_index(drop=True))

    @property
    def observation_columns(self) -> list[str]:
        return list(OBSERVATION_ORDER)

    @property
    def action_columns(self) -> list[str]:
        return ["action_pitch_offset_deg", "action_yaw_setpoint_deg", "action_ipc_level"]

    def arrays(self) -> dict[str, np.ndarray]:
        """Return observation / action / reward / done arrays for offline RL.

        Next observations are the following row within the same episode. The
        terminal step of each episode has no successor, so it repeats its own
        observation and is flagged in `terminals` - the standard convention,
        since the bootstrapped value of a terminal state is masked out anyway.
        """
        frame = self.frame
        observations = frame[self.observation_columns].to_numpy(dtype=np.float32)
        actions = frame[self.action_columns].to_numpy(dtype=np.float32)
        rewards = frame["reward"].to_numpy(dtype=np.float32)
        terminals = frame["done"].to_numpy(dtype=np.int8)

        # Row i's successor is row i+1 unless row i ends an episode.
        successor = np.arange(len(frame)) + 1
        successor[-1] = len(frame) - 1
        is_last = terminals.astype(bool)
        successor[is_last] = np.flatnonzero(is_last)
        next_observations = observations[np.clip(successor, 0, len(frame) - 1)]

        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "terminals": terminals,
        }
