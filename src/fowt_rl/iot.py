"""IoT sensor layer: turn simulation ground truth into realistic measurements.

Why this matters
----------------
OpenFAST - and therefore FLOATBench - reports perfect state: noiseless,
unbiased, perfectly synchronised, never missing. A controller deployed on a real
floating turbine sees nothing of the kind. It reads a distributed network of
instruments over a wireless/fieldbus link, each with its own noise floor,
calibration drift, ADC resolution, update rate and packet-loss behaviour.

A policy trained on perfect state and deployed on noisy telemetry will
underperform, sometimes badly, because it has learned to rely on precision that
does not exist. So the pipeline emits *both* representations for every step:

    true_*   the ground-truth channel value (from FLOATBench + the physics layer)
    meas_*   what the IoT network actually delivered
    valid_*  whether a fresh sample arrived this step (0 = packet lost / stale)

Training on `meas_*` and evaluating against `true_*` is the intended usage, and
keeping both lets you quantify exactly how much the sensor layer costs.

Effects modelled, in application order
--------------------------------------
1. calibration bias   - per-episode constant offset (instrument not re-zeroed)
2. drift              - slow random walk over the episode (thermal, ageing)
3. additive noise     - absolute floor plus a term proportional to the reading
4. quantisation       - finite ADC / telemetry resolution
5. latency            - channel reports a value from `latency_steps` ago
6. dropout            - packet lost; the last good value is held and valid_* = 0

Channel parameters are representative instrumentation-grade figures for offshore
wind installations, not values from a specific datasheet, and every one is
overridable. They are listed in docs/LIMITATIONS.md as an explicit modelling
assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SensorChannel:
    """Error model for one measured channel."""

    name: str
    instrument: str
    noise_std: float = 0.0  # absolute noise standard deviation
    noise_relative: float = 0.0  # noise proportional to |value|
    bias_std: float = 0.0  # per-episode constant calibration offset
    drift_std_per_step: float = 0.0  # random-walk drift increment per step
    quantisation: float = 0.0  # telemetry resolution (0 = none)
    dropout_probability: float = 0.0  # per-step packet-loss probability
    latency_steps: int = 0  # reporting delay in control steps
    non_negative: bool = False  # clamp at zero (physically bounded channels)


# Default sensor suite. Channel names refer to columns produced by the MDP
# builder; any channel whose column is absent is silently skipped.
DEFAULT_CHANNELS: tuple[SensorChannel, ...] = (
    SensorChannel(
        name="wind_speed",
        instrument="nacelle cup anemometer / met mast",
        noise_std=0.4,
        noise_relative=0.03,
        bias_std=0.20,
        drift_std_per_step=0.01,
        quantisation=0.1,
        dropout_probability=0.005,
        non_negative=True,
    ),
    SensorChannel(
        name="turbulence_std",
        instrument="derived from anemometer time series",
        noise_std=0.15,
        noise_relative=0.08,
        bias_std=0.05,
        quantisation=0.05,
        dropout_probability=0.01,
        latency_steps=1,
        non_negative=True,
    ),
    SensorChannel(
        name="wind_direction",
        instrument="nacelle wind vane",
        noise_std=3.0,
        bias_std=2.0,
        drift_std_per_step=0.05,
        quantisation=1.0,
        dropout_probability=0.005,
    ),
    SensorChannel(
        name="wave_hs",
        instrument="wave radar / nearby buoy",
        noise_std=0.15,
        noise_relative=0.05,
        bias_std=0.08,
        quantisation=0.05,
        dropout_probability=0.02,
        latency_steps=1,
        non_negative=True,
    ),
    SensorChannel(
        name="wave_tp",
        instrument="wave radar spectral estimate",
        noise_std=0.5,
        noise_relative=0.03,
        bias_std=0.20,
        quantisation=0.1,
        dropout_probability=0.02,
        latency_steps=1,
        non_negative=True,
    ),
    SensorChannel(
        name="tower_damage_rate",
        instrument="tower-base strain gauge rosette",
        noise_relative=0.06,
        bias_std=0.0,
        drift_std_per_step=0.0,
        dropout_probability=0.01,
        non_negative=True,
    ),
    SensorChannel(
        name="thrust",
        instrument="estimated from strain gauges and rotor state",
        noise_relative=0.04,
        noise_std=1.0e4,
        dropout_probability=0.005,
        non_negative=True,
    ),
    SensorChannel(
        name="power",
        instrument="SCADA power transducer",
        noise_relative=0.005,
        quantisation=1.0e4,
        dropout_probability=0.002,
        non_negative=True,
    ),
)


@dataclass
class IoTConfig:
    """Configuration of the IoT sensor layer."""

    channels: tuple[SensorChannel, ...] = DEFAULT_CHANNELS
    enabled: bool = True
    # Global multiplier on every noise / bias / drift term. 0 reproduces perfect
    # sensing; 1 is the nominal suite; >1 stress-tests robustness.
    severity: float = 1.0
    # Global multiplier on dropout probabilities, for degraded-network studies.
    dropout_scale: float = 1.0
    seed: int = 20260826

    def scaled_channels(self) -> tuple[SensorChannel, ...]:
        """Apply `severity` and `dropout_scale` to every channel."""
        if self.severity == 1.0 and self.dropout_scale == 1.0:
            return self.channels
        scaled = []
        for channel in self.channels:
            scaled.append(
                replace(
                    channel,
                    noise_std=channel.noise_std * self.severity,
                    noise_relative=channel.noise_relative * self.severity,
                    bias_std=channel.bias_std * self.severity,
                    drift_std_per_step=channel.drift_std_per_step * self.severity,
                    dropout_probability=float(
                        np.clip(channel.dropout_probability * self.dropout_scale, 0.0, 1.0)
                    ),
                )
            )
        return tuple(scaled)

    def describe(self) -> list[dict]:
        return [
            {
                "channel": channel.name,
                "instrument": channel.instrument,
                "noise_std": channel.noise_std,
                "noise_relative": channel.noise_relative,
                "bias_std": channel.bias_std,
                "drift_std_per_step": channel.drift_std_per_step,
                "quantisation": channel.quantisation,
                "dropout_probability": channel.dropout_probability,
                "latency_steps": channel.latency_steps,
            }
            for channel in self.scaled_channels()
        ]


def _apply_to_series(
    values: np.ndarray,
    channel: SensorChannel,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the full sensor error chain to one episode's worth of one channel.

    `values` must be ordered in time. Returns (measured, valid_flag).
    """
    n = values.size
    measured = values.astype(float, copy=True)

    # 1. per-episode calibration bias
    if channel.bias_std > 0:
        measured = measured + rng.normal(0.0, channel.bias_std)

    # 2. slow drift as a random walk
    if channel.drift_std_per_step > 0:
        measured = measured + np.cumsum(rng.normal(0.0, channel.drift_std_per_step, n))

    # 3. additive noise, absolute plus proportional
    scale = np.sqrt(
        channel.noise_std**2 + (channel.noise_relative * np.abs(values)) ** 2
    )
    if np.any(scale > 0):
        measured = measured + rng.normal(0.0, 1.0, n) * scale

    # 4. quantisation
    if channel.quantisation > 0:
        measured = np.round(measured / channel.quantisation) * channel.quantisation

    if channel.non_negative:
        measured = np.maximum(measured, 0.0)

    # 5. latency - the channel reports an older sample; the first steps repeat
    #    the earliest available reading.
    if channel.latency_steps > 0:
        lag = min(channel.latency_steps, n)
        measured = np.concatenate([np.repeat(measured[:1], lag), measured[:-lag]])

    # 6. dropout - hold the last good value and flag the sample invalid
    valid = np.ones(n, dtype=np.int8)
    if channel.dropout_probability > 0:
        lost = rng.random(n) < channel.dropout_probability
        lost[0] = False  # a session always starts with a fresh reading
        if lost.any():
            valid[lost] = 0
            held = measured.copy()
            last_good = held[0]
            for index in range(n):
                if lost[index]:
                    held[index] = last_good
                else:
                    last_good = held[index]
            measured = held

    return measured, valid


def apply_sensor_layer(
    frame: pd.DataFrame,
    config: IoTConfig | None = None,
    episode_column: str = "episode_id",
    step_column: str = "step",
) -> pd.DataFrame:
    """Add `meas_*` and `valid_*` columns for every configured channel.

    Expects a column `true_<channel>` for each channel; channels without a
    matching column are skipped. Errors are applied independently per episode so
    that bias, drift and latency are correlated within an episode but not across
    episodes - which is how real instrumentation behaves between maintenance
    visits.
    """
    config = config or IoTConfig()
    frame = frame.sort_values([episode_column, step_column], kind="stable").reset_index(drop=True)

    channels = [
        channel
        for channel in config.scaled_channels()
        if f"true_{channel.name}" in frame.columns
    ]
    if not channels:
        return frame

    if not config.enabled:
        for channel in channels:
            frame[f"meas_{channel.name}"] = frame[f"true_{channel.name}"].to_numpy(dtype=float)
            frame[f"valid_{channel.name}"] = np.int8(1)
        frame["sensor_health"] = np.float32(1.0)
        return frame

    rng = np.random.default_rng(config.seed)
    episode_ids = frame[episode_column].to_numpy()
    boundaries = np.flatnonzero(np.r_[True, episode_ids[1:] != episode_ids[:-1]])
    slices = [
        slice(start, end)
        for start, end in zip(boundaries, np.r_[boundaries[1:], episode_ids.size])
    ]

    for channel in channels:
        truth = frame[f"true_{channel.name}"].to_numpy(dtype=float)
        measured = np.empty_like(truth)
        valid = np.empty(truth.size, dtype=np.int8)
        for window in slices:
            measured[window], valid[window] = _apply_to_series(truth[window], channel, rng)
        frame[f"meas_{channel.name}"] = measured
        frame[f"valid_{channel.name}"] = valid

    valid_columns = [f"valid_{channel.name}" for channel in channels]
    frame["sensor_health"] = frame[valid_columns].to_numpy(dtype=float).mean(axis=1).astype("float32")
    return frame


def measurement_error_report(frame: pd.DataFrame, config: IoTConfig | None = None) -> dict:
    """Summarise realised measurement error, for validating the sensor layer."""
    config = config or IoTConfig()
    report: dict[str, dict] = {}
    for channel in config.scaled_channels():
        true_column = f"true_{channel.name}"
        meas_column = f"meas_{channel.name}"
        if true_column not in frame.columns or meas_column not in frame.columns:
            continue
        truth = frame[true_column].to_numpy(dtype=float)
        measured = frame[meas_column].to_numpy(dtype=float)
        denominator = np.maximum(np.abs(truth), 1e-12)
        report[channel.name] = {
            "instrument": channel.instrument,
            "bias": float(np.mean(measured - truth)),
            "rmse": float(np.sqrt(np.mean((measured - truth) ** 2))),
            "median_abs_rel_error": float(np.median(np.abs(measured - truth) / denominator)),
            "dropout_rate": float(1.0 - frame[f"valid_{channel.name}"].mean()),
        }
    return report
