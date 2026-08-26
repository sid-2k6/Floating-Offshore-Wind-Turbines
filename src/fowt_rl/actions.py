"""Control action space, actuator limits and actuator duty cost.

The agent commands three quantities every control step:

==================  ==============  =========================================
action              range           physical meaning
==================  ==============  =========================================
pitch_offset_deg    [0, 8] deg      collective blade-pitch offset added to the
                                    baseline schedule, towards feather. Sheds
                                    thrust and therefore tower load, at the
                                    cost of power.
yaw_setpoint_deg    [-30, 30] deg   commanded nacelle yaw relative to the
                                    measured inflow direction. Zero means
                                    "track the wind"; non-zero means deliberate
                                    misalignment.
ipc_level           [0, 1]          individual pitch control activation. Acts
                                    on cyclic rotor loading only, costs
                                    actuator duty rather than power.
==================  ==============  =========================================

Only feathering pitch offsets are allowed: negative offsets move the blades
towards stall, which increases loads and risks stall-induced vibration, and the
official pitch limit is PC_MinPit = -0.07 rad = -4.0 deg, so there is very
little margin in that direction anyway.

Actuator limits are taken from the official IEA-22-280-RWT semi-submersible
ROSCO controller file (`IEA-22-280-RWT-Semi_DISCON.IN`):

    PC_MaxRat    =  0.03491 rad/s  ->  2.00 deg/s   pitch rate limit
    Y_Rate       =  0.00870 rad/s  ->  0.499 deg/s  yaw rate
    Y_ErrThresh  =  4 / 8 deg                       yaw deadband
    IPC_ControlMode = 0                             IPC off in the baseline

That last line is worth noting: IPC is disabled in the reference controller, so
every FLOATBench run has zero IPC activity. The IPC action is therefore a
genuine extension of the dataset rather than something latent in it.

A 10-minute control step is long compared with actuator dynamics, so the rate
limits do not bind on a single step. They are instead used to convert commanded
setpoint changes into *actuator duty* - yaw travel time, pitch travel, IPC
cyclic activity - which is what the reward penalises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ACTION_NAMES = ("pitch_offset_deg", "yaw_setpoint_deg", "ipc_level")
N_ACTIONS = len(ACTION_NAMES)


@dataclass(frozen=True)
class ActuatorLimits:
    """Physical actuator limits from the official ROSCO controller file."""

    pitch_rate_deg_s: float = 2.0  # PC_MaxRat = 0.03491 rad/s
    yaw_rate_deg_s: float = 0.4985  # Y_Rate = 0.00870 rad/s
    yaw_deadband_deg: float = 8.0  # Y_ErrThresh (upper value)
    min_pitch_deg: float = -4.01  # PC_MinPit = -0.07 rad
    max_pitch_deg: float = 90.0  # PC_MaxPit = 1.57 rad


@dataclass(frozen=True)
class ActionSpace:
    """Bounded, rate-limited three-dimensional continuous action space."""

    pitch_offset_bounds: tuple[float, float] = (0.0, 8.0)
    yaw_setpoint_bounds: tuple[float, float] = (-30.0, 30.0)
    ipc_level_bounds: tuple[float, float] = (0.0, 1.0)

    # Per-step setpoint change limits. These keep commanded trajectories
    # physically smooth over a 10-minute step; they are operational choices
    # rather than hard actuator limits (see module docstring).
    max_pitch_offset_change_deg: float = 4.0
    max_yaw_setpoint_change_deg: float = 15.0
    max_ipc_level_change: float = 0.5

    limits: ActuatorLimits = field(default_factory=ActuatorLimits)

    # ------------------------------------------------------------------
    @property
    def low(self) -> np.ndarray:
        return np.array(
            [self.pitch_offset_bounds[0], self.yaw_setpoint_bounds[0], self.ipc_level_bounds[0]],
            dtype=float,
        )

    @property
    def high(self) -> np.ndarray:
        return np.array(
            [self.pitch_offset_bounds[1], self.yaw_setpoint_bounds[1], self.ipc_level_bounds[1]],
            dtype=float,
        )

    def clip(self, action: np.ndarray) -> np.ndarray:
        """Clip an action (..., 3) into the bounds."""
        return np.clip(np.asarray(action, dtype=float), self.low, self.high)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform random actions, shape (n, 3).

        A quarter of samples are forced to the exact zero-action baseline so
        that any downstream model sees the uncontrolled reference often enough
        to anchor on it.
        """
        action = self.low + rng.random((n, N_ACTIONS)) * (self.high - self.low)
        baseline = rng.random(n) < 0.25
        action[baseline] = np.array([0.0, 0.0, 0.0])
        return action

    def grid(
        self,
        n_pitch: int = 5,
        n_yaw: int = 5,
        n_ipc: int = 3,
    ) -> np.ndarray:
        """Full factorial action grid, shape (n_pitch*n_yaw*n_ipc, 3)."""
        pitch = np.linspace(*self.pitch_offset_bounds, n_pitch)
        yaw = np.linspace(*self.yaw_setpoint_bounds, n_yaw)
        ipc = np.linspace(*self.ipc_level_bounds, n_ipc)
        mesh = np.meshgrid(pitch, yaw, ipc, indexing="ij")
        return np.column_stack([component.ravel() for component in mesh])

    def apply_rate_limits(self, previous: np.ndarray, requested: np.ndarray) -> np.ndarray:
        """Limit how far an action may move from the previous one, then clip."""
        previous = np.asarray(previous, dtype=float)
        requested = self.clip(requested)
        max_change = np.array(
            [
                self.max_pitch_offset_change_deg,
                self.max_yaw_setpoint_change_deg,
                self.max_ipc_level_change,
            ],
            dtype=float,
        )
        delta = np.clip(requested - previous, -max_change, max_change)
        return self.clip(previous + delta)


@dataclass(frozen=True)
class DutyCostConfig:
    """Weights converting actuator activity into a dimensionless duty cost."""

    # Pitch bearing travel, normalised by the full commanded pitch range.
    pitch_travel_weight: float = 1.0
    # Yaw drive engagement, normalised by step duration (fraction of the step
    # spent actively yawing at the rated yaw rate).
    yaw_engagement_weight: float = 1.0
    # Continuous cyclic pitch activity while IPC is enabled. IPC pitches every
    # blade once or twice per revolution for the whole step, so it dominates
    # bearing duty even at modest amplitude.
    ipc_duty_weight: float = 1.0
    # Standing pitch offset: holding a feathered offset is a static bearing
    # position, so it is charged much less than motion.
    pitch_hold_weight: float = 0.1


def actuator_duty(
    action: np.ndarray,
    previous_action: np.ndarray,
    space: ActionSpace,
    step_seconds: float,
    config: DutyCostConfig | None = None,
) -> dict[str, np.ndarray]:
    """Decompose actuator activity for one step.

    Returns individual duty terms (each roughly normalised to [0, 1]) plus the
    weighted total.
    """
    config = config or DutyCostConfig()
    action = np.atleast_2d(np.asarray(action, dtype=float))
    previous_action = np.atleast_2d(np.asarray(previous_action, dtype=float))

    pitch, yaw, ipc = action[:, 0], action[:, 1], action[:, 2]
    prev_pitch, prev_yaw, prev_ipc = (
        previous_action[:, 0],
        previous_action[:, 1],
        previous_action[:, 2],
    )

    pitch_span = space.pitch_offset_bounds[1] - space.pitch_offset_bounds[0]
    yaw_span = space.yaw_setpoint_bounds[1] - space.yaw_setpoint_bounds[0]

    pitch_travel = np.abs(pitch - prev_pitch) / max(pitch_span, 1e-9)
    pitch_hold = np.abs(pitch) / max(pitch_span, 1e-9)

    # Time needed to slew the yaw setpoint, as a fraction of the step. The yaw
    # deadband means small commanded changes do not engage the drive at all.
    yaw_change = np.abs(yaw - prev_yaw)
    engaged = yaw_change > space.limits.yaw_deadband_deg
    yaw_seconds = np.where(engaged, yaw_change / max(space.limits.yaw_rate_deg_s, 1e-9), 0.0)
    yaw_engagement = np.clip(yaw_seconds / max(step_seconds, 1e-9), 0.0, 1.0)

    ipc_duty = np.clip(ipc, 0.0, 1.0)

    total = (
        config.pitch_travel_weight * pitch_travel
        + config.pitch_hold_weight * pitch_hold
        + config.yaw_engagement_weight * yaw_engagement
        + config.ipc_duty_weight * ipc_duty
    )
    normaliser = (
        config.pitch_travel_weight
        + config.pitch_hold_weight
        + config.yaw_engagement_weight
        + config.ipc_duty_weight
    )

    return {
        "duty_pitch_travel": pitch_travel,
        "duty_pitch_hold": pitch_hold,
        "duty_yaw_engagement": yaw_engagement,
        "duty_ipc": ipc_duty,
        "duty_yaw_seconds": yaw_seconds,
        "duty_total": total / max(normaliser, 1e-9),
        "unused_previous_ipc": prev_ipc,
    }
