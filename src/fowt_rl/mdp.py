"""Markov decision process construction over the FLOATBench operating envelope.

FLOATBench rows are independent 10-minute stationary simulations - there is no
time axis, so there is no MDP. This module builds one.

Episodes
--------
Each FLOATBench condition is identified by a grid coordinate
`(wind_speed_id, wave_hs_id, wave_tp_id)` on the 22 x 7 x 7 envelope, with six
turbulence seeds per coordinate. An episode is a correlated random walk over
that grid: at each step every coordinate either holds or moves by one index,
which reproduces the way real metocean conditions evolve - slowly, and with wind
and wave states that drift together because FLOATBench sampled them from a joint
distribution. A fresh turbulence seed is drawn each step, so consecutive steps
are different realisations of a slowly changing sea state.

With a 600 s step and 36 steps, one episode covers six hours of operation.

Inflow direction
----------------
FLOATBench simulates aligned wind and waves and carries no direction
information, so inflow direction is introduced here as an Ornstein-Uhlenbeck
process (mean-reverting, hour-scale correlation). This is what gives the yaw
action something to do: the agent can track the wind, or deliberately misalign.
A per-episode static vane bias is added on top, since standing yaw misalignment
from vane miscalibration is common on operating turbines.

Plant
-----
The nacelle is a rate-limited, deadband-limited first-order tracker of the
commanded heading, using the real `Y_Rate` and `Y_ErrThresh` values from the
official ROSCO controller file. Residual yaw error feeds both the aerodynamic
model (thrust and power loss) and the cyclic load term (fatigue penalty).

Reward
------
See `fowt_rl.config.RewardConfig`. The uncontrolled baseline scores exactly
zero, so the sign of a return says whether a policy bought more fatigue relief
than it paid for in lost energy and actuator duty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .actions import ActionSpace, DutyCostConfig, actuator_duty
from .aero import RotorAero
from .config import EpisodeConfig, RewardConfig
from .damage import ControlAuthorityConfig, damage_under_action, summarise_sections
from .floatbench import TowerData
from .load_model import TowerLoadModel

# Shape of the FLOATBench operating envelope.
GRID_SHAPE = (22, 7, 7)
N_SEEDS = 6

# Columns the agent may condition on. `meas_*` counterparts are produced by the
# IoT layer; policies should train on those and not on the `true_*` columns.
OBSERVATION_CHANNELS = (
    "wind_speed",
    "turbulence_std",
    "wind_direction",
    "wave_hs",
    "wave_tp",
    "thrust",
    "power",
    "tower_damage_rate",
)

# Plant/actuator state that is known exactly by the controller (no sensing
# uncertainty): its own previous commands and the yaw encoder.
PROPRIOCEPTIVE_COLUMNS = (
    "prev_pitch_offset_deg",
    "prev_yaw_setpoint_deg",
    "prev_ipc_level",
    "nacelle_yaw_deg",
    "yaw_error_deg",
    "cumulative_damage_fraction",
    "step_fraction",
)

ACTION_COLUMNS = ("action_pitch_offset_deg", "action_yaw_setpoint_deg", "action_ipc_level")


# ---------------------------------------------------------------------------
# Condition indexing
# ---------------------------------------------------------------------------
def build_condition_index(tower_data: TowerData) -> np.ndarray:
    """Map grid coordinates to condition row indices.

    Returns an int array of shape (22, 7, 7, 6) holding the row index into
    `tower_data.conditions`, or -1 where a combination is absent.
    """
    conditions = tower_data.conditions
    index = np.full(GRID_SHAPE + (N_SEEDS,), -1, dtype=np.int64)
    index[
        conditions["wind_speed_id"].to_numpy() - 1,
        conditions["wave_hs_id"].to_numpy() - 1,
        conditions["wave_tp_id"].to_numpy() - 1,
        conditions["wind_seed_id"].to_numpy() - 1,
    ] = np.arange(len(conditions), dtype=np.int64)
    if (index < 0).any():
        missing = int((index < 0).sum())
        raise ValueError(f"condition grid is incomplete: {missing} missing combinations")
    return index


def _random_walk(
    n_episodes: int,
    n_steps: int,
    config: EpisodeConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Correlated random walk over the grid, shape (n_episodes, n_steps, 3)."""
    stay = float(np.clip(config.grid_persistence, 0.0, 1.0))
    move = (1.0 - stay) / 2.0
    probabilities = [move, stay, move]

    coordinates = np.empty((n_episodes, n_steps, 3), dtype=np.int64)
    start = np.column_stack([rng.integers(0, size, n_episodes) for size in GRID_SHAPE])
    coordinates[:, 0, :] = start

    for step in range(1, n_steps):
        increment = rng.choice([-1, 0, 1], size=(n_episodes, 3), p=probabilities)
        candidate = coordinates[:, step - 1, :] + increment
        for axis, size in enumerate(GRID_SHAPE):
            candidate[:, axis] = np.clip(candidate[:, axis], 0, size - 1)
        coordinates[:, step, :] = candidate
    return coordinates


def _direction_process(
    n_episodes: int,
    n_steps: int,
    config: EpisodeConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Ornstein-Uhlenbeck inflow direction, shape (n_episodes, n_steps)."""
    dt_hours = config.step_seconds / 3600.0
    decay = float(np.exp(-dt_hours / max(config.direction_correlation_hours, 1e-6)))
    innovation = config.direction_std_deg * np.sqrt(max(1.0 - decay**2, 0.0))

    direction = np.empty((n_episodes, n_steps), dtype=float)
    direction[:, 0] = rng.normal(0.0, config.direction_std_deg, n_episodes)
    for step in range(1, n_steps):
        direction[:, step] = decay * direction[:, step - 1] + rng.normal(
            0.0, innovation, n_episodes
        )
    return np.clip(direction, -config.direction_clip_deg, config.direction_clip_deg)


# ---------------------------------------------------------------------------
# Behaviour policies
# ---------------------------------------------------------------------------
def _behaviour_action(
    policy: np.ndarray,
    space: ActionSpace,
    measured_direction: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Requested (pre-rate-limit) action for each episode at one step.

    `policy` holds a policy label per episode. The mixture deliberately includes
    the zero-action baseline and a pure-IPC policy so that a downstream offline
    RL algorithm sees both the uncontrolled reference and a known-good
    low-cost strategy, alongside uniform exploration.
    """
    n = policy.size
    action = np.zeros((n, 3), dtype=float)

    random_mask = policy == "random"
    if random_mask.any():
        action[random_mask] = space.sample(int(random_mask.sum()), rng)

    ipc_mask = policy == "ipc_only"
    if ipc_mask.any():
        action[ipc_mask, 2] = rng.uniform(0.5, 1.0, int(ipc_mask.sum()))

    feather_mask = policy == "feather"
    if feather_mask.any():
        count = int(feather_mask.sum())
        action[feather_mask, 0] = rng.uniform(1.0, 5.0, count)
        action[feather_mask, 2] = rng.uniform(0.0, 1.0, count)

    yaw_mask = policy == "yaw_seeker"
    if yaw_mask.any():
        count = int(yaw_mask.sum())
        # Command a misalignment on top of tracking the measured inflow.
        action[yaw_mask, 1] = measured_direction[yaw_mask] + rng.uniform(-25.0, 25.0, count)
        action[yaw_mask, 2] = rng.uniform(0.0, 0.5, count)

    # "baseline" leaves the action at exactly zero.
    return space.clip(action)


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------
def build_transitions(
    tower_data: TowerData,
    load_model: TowerLoadModel,
    aero: RotorAero,
    space: ActionSpace,
    episode_config: EpisodeConfig,
    reward_config: RewardConfig,
    duty_config: DutyCostConfig | None = None,
    authority: ControlAuthorityConfig | None = None,
) -> pd.DataFrame:
    """Roll out episodes and return one row per transition.

    Rows are emitted in (episode_id, step) order and carry:

    * `true_*`   ground-truth observation channels at the start of the step
    * `action_*` the action actually applied after rate limiting
    * `reward` and its decomposed terms
    * `next_true_*` the same channels one step later
    * `done` episode-termination flag
    * diagnostic columns (damage per key section, controllable share, gains)
    """
    duty_config = duty_config or DutyCostConfig()
    rng = np.random.default_rng(episode_config.seed + abs(hash(tower_data.tower)) % 10_000)

    n_episodes = int(episode_config.episodes_per_tower)
    n_steps = int(episode_config.steps_per_episode)
    if n_episodes <= 0 or n_steps <= 1:
        raise ValueError("need at least one episode of two or more steps")

    index = build_condition_index(tower_data)
    conditions = tower_data.conditions
    damage_matrix = tower_data.damage

    # Pre-compute the whole trajectory of environmental conditions.
    coordinates = _random_walk(n_episodes, n_steps, episode_config, rng)
    seeds = rng.integers(0, N_SEEDS, size=(n_episodes, n_steps))
    rows = index[coordinates[..., 0], coordinates[..., 1], coordinates[..., 2], seeds]

    direction = _direction_process(n_episodes, n_steps, episode_config, rng)
    vane_bias = rng.normal(0.0, episode_config.vane_bias_std_deg, n_episodes)

    names = list(episode_config.policy_weights)
    weights = np.array([episode_config.policy_weights[name] for name in names], dtype=float)
    weights = weights / weights.sum()
    policy = rng.choice(names, size=n_episodes, p=weights)

    # A per-tower damage scale used to normalise the cumulative-damage state.
    baseline_max = damage_matrix.max(axis=1)
    damage_scale = float(np.percentile(baseline_max, 99)) * n_steps
    damage_scale = damage_scale if damage_scale > 0 else 1.0

    # Severity reference: the lifetime-weighted baseline DEL at the configured
    # percentile over the whole tower envelope. Computed once so that severity
    # is comparable across every episode and step of this tower.
    lifetime_del = np.power(
        np.maximum(baseline_max * conditions["damage_weight"].to_numpy(dtype=float), 0.0),
        1.0 / load_model.wohler_exponent,
    )
    severity_reference = float(
        np.percentile(lifetime_del, reward_config.severity_reference_percentile)
    )
    severity_reference = severity_reference if severity_reference > 0 else 1.0

    # Rolling plant state.
    previous_action = np.zeros((n_episodes, 3), dtype=float)
    nacelle_yaw = direction[:, 0].copy()  # start aligned with the inflow
    cumulative_damage = np.zeros(n_episodes, dtype=float)

    episode_ids = np.arange(n_episodes, dtype=np.int64)
    records: list[dict] = []
    # Per-step cache so step t can look up the state it produced for step t+1.
    step_state: list[dict] = []

    for step in range(n_steps):
        row_index = rows[:, step]
        step_conditions = conditions.iloc[row_index].reset_index(drop=True)
        step_damage = damage_matrix[row_index]

        inflow = direction[:, step]
        measured_direction = inflow + vane_bias

        # ---- action -----------------------------------------------------
        requested = _behaviour_action(policy, space, measured_direction, rng)
        action = space.apply_rate_limits(previous_action, requested)

        # ---- nacelle tracking (deadband + rate limit) -------------------
        target_heading = measured_direction - action[:, 1]
        delta = target_heading - nacelle_yaw
        engaged = np.abs(delta) > space.limits.yaw_deadband_deg
        max_travel = space.limits.yaw_rate_deg_s * episode_config.step_seconds
        travel = np.where(engaged, np.sign(delta) * np.minimum(np.abs(delta), max_travel), 0.0)
        new_nacelle_yaw = nacelle_yaw + travel
        yaw_error = inflow - new_nacelle_yaw

        # ---- plant response --------------------------------------------
        controlled = damage_under_action(
            step_conditions,
            step_damage,
            load_model,
            aero,
            action[:, 0],
            yaw_error,
            action[:, 2],
            authority,
        )
        baseline = damage_under_action(
            step_conditions,
            step_damage,
            load_model,
            aero,
            np.zeros(n_episodes),
            np.zeros(n_episodes),
            np.zeros(n_episodes),
            authority,
        )

        controlled_summary = summarise_sections(controlled["damage"])
        baseline_summary = summarise_sections(baseline["damage"])

        key = f"damage_{reward_config.fatigue_section}"
        damage_controlled = controlled_summary[key]
        damage_baseline = baseline_summary[key]

        # ---- reward -----------------------------------------------------
        del_ratio = np.power(
            np.divide(
                damage_controlled,
                damage_baseline,
                out=np.ones_like(damage_controlled),
                where=damage_baseline > 0,
            ),
            1.0 / load_model.wohler_exponent,
        )
        fatigue_relief = 1.0 - del_ratio

        # Lifetime significance of this condition, so that relief is worth more
        # where tower life is actually being consumed.
        if reward_config.severity_weighting:
            step_lifetime_del = np.power(
                np.maximum(
                    damage_baseline * step_conditions["damage_weight"].to_numpy(dtype=float), 0.0
                ),
                1.0 / load_model.wohler_exponent,
            )
            severity = np.clip(
                step_lifetime_del / severity_reference, 0.0, reward_config.severity_clip
            )
        else:
            severity = np.ones_like(fatigue_relief)

        power_loss_fraction = np.clip(
            (controlled["power_base_w"] - controlled["power_w"]) / aero.properties.rated_power_w,
            0.0,
            1.0,
        )

        duty = actuator_duty(action, previous_action, space, episode_config.step_seconds, duty_config)

        reward = (
            reward_config.fatigue_weight * fatigue_relief * severity
            - reward_config.power_weight * power_loss_fraction
            - reward_config.duty_weight * duty["duty_total"]
        )

        turbulence_std = step_conditions["std_wind_speed"].to_numpy(dtype=float)
        mean_wind = step_conditions["mean_wind_speed"].to_numpy(dtype=float)

        state = {
            "true_wind_speed": mean_wind,
            "true_turbulence_std": turbulence_std,
            "true_wind_direction": measured_direction,
            "true_wave_hs": step_conditions["wave_hs"].to_numpy(dtype=float),
            "true_wave_tp": step_conditions["wave_tp"].to_numpy(dtype=float),
            "true_thrust": controlled["thrust_n"],
            "true_power": controlled["power_w"],
            "true_tower_damage_rate": damage_controlled,
        }
        step_state.append(state)

        cumulative_before = cumulative_damage.copy()
        cumulative_damage = cumulative_damage + damage_controlled

        records.append(
            {
                "episode_id": episode_ids,
                "step": np.full(n_episodes, step, dtype=np.int32),
                "behaviour_policy": policy,
                "sim_id": step_conditions["sim_id"].to_numpy(),
                "wind_speed_id": step_conditions["wind_speed_id"].to_numpy(),
                "wave_hs_id": step_conditions["wave_hs_id"].to_numpy(),
                "wave_tp_id": step_conditions["wave_tp_id"].to_numpy(),
                "wind_seed_id": step_conditions["wind_seed_id"].to_numpy(),
                "wind_speed_setpoint": step_conditions["wind_speed"].to_numpy(dtype=float),
                "damage_weight": step_conditions["damage_weight"].to_numpy(dtype=float),
                # observation - ground truth
                **state,
                "true_turbulence_intensity": turbulence_std / np.maximum(mean_wind, 1e-6),
                # proprioceptive state
                "prev_pitch_offset_deg": previous_action[:, 0].copy(),
                "prev_yaw_setpoint_deg": previous_action[:, 1].copy(),
                "prev_ipc_level": previous_action[:, 2].copy(),
                "nacelle_yaw_deg": new_nacelle_yaw.copy(),
                "yaw_error_deg": yaw_error,
                "inflow_direction_deg": inflow,
                "vane_bias_deg": vane_bias,
                "cumulative_damage_fraction": cumulative_before / damage_scale,
                "step_fraction": np.full(n_episodes, step / (n_steps - 1), dtype=float),
                # action
                "action_pitch_offset_deg": action[:, 0],
                "action_yaw_setpoint_deg": action[:, 1],
                "action_ipc_level": action[:, 2],
                "action_requested_pitch_offset_deg": requested[:, 0],
                "action_requested_yaw_setpoint_deg": requested[:, 1],
                "action_requested_ipc_level": requested[:, 2],
                # reward and terms
                "reward": reward,
                "reward_fatigue_relief": fatigue_relief,
                "reward_severity": severity,
                "reward_fatigue_term": reward_config.fatigue_weight * fatigue_relief * severity,
                "reward_power_loss_fraction": power_loss_fraction,
                "reward_duty_total": duty["duty_total"],
                "duty_pitch_travel": duty["duty_pitch_travel"],
                "duty_yaw_engagement": duty["duty_yaw_engagement"],
                "duty_yaw_seconds": duty["duty_yaw_seconds"],
                "duty_ipc": duty["duty_ipc"],
                # damage diagnostics
                "damage_controlled": damage_controlled,
                "damage_baseline": damage_baseline,
                "damage_ratio": np.divide(
                    damage_controlled,
                    damage_baseline,
                    out=np.ones_like(damage_controlled),
                    where=damage_baseline > 0,
                ),
                "del_ratio": del_ratio,
                "damage_controlled_base_section": controlled_summary["damage_base"],
                "damage_controlled_top_section": controlled_summary["damage_top"],
                "damage_controlled_max_section_id": controlled_summary["damage_argmax_section"],
                "damage_baseline_base_section": baseline_summary["damage_base"],
                "damage_baseline_top_section": baseline_summary["damage_top"],
                "controllable_share_max_section": controlled["controllable_share"].max(axis=1),
                "controllable_share_mean": controlled["controllable_share"].mean(axis=1),
                # rotor state
                "ct_ratio": controlled["ct_ratio"],
                "cp_ratio": controlled["cp_ratio"],
                "power_baseline_w": controlled["power_base_w"],
                "thrust_baseline_n": controlled["thrust_base_n"],
                "gain_thrust_turbulence": controlled["gains"][:, 0],
                "gain_rotor_cyclic": controlled["gains"][:, 1],
            }
        )

        previous_action = action
        nacelle_yaw = new_nacelle_yaw

    # ---- assemble, attaching next-state columns -------------------------
    frame = pd.DataFrame({key: np.concatenate([r[key] for r in records]) for key in records[0]})

    next_state = {}
    for channel in OBSERVATION_CHANNELS:
        column = f"true_{channel}"
        stacked = np.concatenate(
            [
                step_state[min(step + 1, n_steps - 1)][column]
                for step in range(n_steps)
            ]
        )
        next_state[f"next_true_{channel}"] = stacked
    frame = frame.assign(**next_state)

    frame["done"] = (frame["step"] == n_steps - 1).astype(np.int8)
    frame["tower"] = tower_data.tower
    frame["step_seconds"] = np.float32(episode_config.step_seconds)
    return frame.sort_values(["episode_id", "step"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Action sweep (supervised surrogate training set)
# ---------------------------------------------------------------------------
def build_action_sweep(
    tower_data: TowerData,
    load_model: TowerLoadModel,
    aero: RotorAero,
    space: ActionSpace,
    n_pitch: int,
    n_yaw: int,
    n_ipc: int,
    condition_fraction: float = 1.0,
    seed: int = 0,
    authority: ControlAuthorityConfig | None = None,
) -> pd.DataFrame:
    """Evaluate a full factorial action grid at every FLOATBench condition.

    This is the static counterpart to `build_transitions`: no episodes, no
    sensor noise, one row per (condition, action). It is the dataset to use for
    training a fast reward/damage surrogate, and for auditing how damage,
    power and thrust respond to each action dimension.

    Here `action_yaw_setpoint_deg` is interpreted directly as the residual yaw
    misalignment, since there is no episode context and therefore no inflow
    direction to track.
    """
    rng = np.random.default_rng(seed)
    conditions = tower_data.conditions
    damage_matrix = tower_data.damage

    if condition_fraction < 1.0:
        keep = rng.random(len(conditions)) < condition_fraction
        keep[0] = True
        conditions = conditions.loc[keep].reset_index(drop=True)
        damage_matrix = damage_matrix[keep]

    grid = space.grid(n_pitch=n_pitch, n_yaw=n_yaw, n_ipc=n_ipc)
    frames: list[pd.DataFrame] = []

    for pitch_offset, yaw_error, ipc_level in grid:
        n = len(conditions)
        result = damage_under_action(
            conditions,
            damage_matrix,
            load_model,
            aero,
            np.full(n, pitch_offset),
            np.full(n, yaw_error),
            np.full(n, ipc_level),
            authority,
        )
        controlled = summarise_sections(result["damage"])
        frames.append(
            pd.DataFrame(
                {
                    "tower": tower_data.tower,
                    "sim_id": conditions["sim_id"].to_numpy(),
                    "wind_speed_id": conditions["wind_speed_id"].to_numpy(),
                    "wave_hs_id": conditions["wave_hs_id"].to_numpy(),
                    "wave_tp_id": conditions["wave_tp_id"].to_numpy(),
                    "wind_seed_id": conditions["wind_seed_id"].to_numpy(),
                    "mean_wind_speed": conditions["mean_wind_speed"].to_numpy(dtype=float),
                    "std_wind_speed": conditions["std_wind_speed"].to_numpy(dtype=float),
                    "wave_hs": conditions["wave_hs"].to_numpy(dtype=float),
                    "wave_tp": conditions["wave_tp"].to_numpy(dtype=float),
                    "damage_weight": conditions["damage_weight"].to_numpy(dtype=float),
                    "action_pitch_offset_deg": np.float32(pitch_offset),
                    "action_yaw_error_deg": np.float32(yaw_error),
                    "action_ipc_level": np.float32(ipc_level),
                    "damage_max": controlled["damage_max"],
                    "damage_base_section": controlled["damage_base"],
                    "damage_top_section": controlled["damage_top"],
                    "damage_mean_section": controlled["damage_mean"],
                    "damage_max_section_id": controlled["damage_argmax_section"].astype(np.int16),
                    "damage_ratio_max": result["damage_ratio"].max(axis=1),
                    "controllable_share_max": result["controllable_share"].max(axis=1),
                    "power_w": result["power_w"],
                    "power_baseline_w": result["power_base_w"],
                    "thrust_n": result["thrust_n"],
                    "thrust_baseline_n": result["thrust_base_n"],
                    "ct_ratio": result["ct_ratio"],
                    "cp_ratio": result["cp_ratio"],
                }
            )
        )

    return pd.concat(frames, ignore_index=True)
