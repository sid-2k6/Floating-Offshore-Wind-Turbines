"""End-to-end dataset build.

Pipeline
--------
    1. load official IEA-22-280-RWT reference aerodynamics
    2. for each FLOATBench tower variant
         a. load and reshape the benchmark data
         b. calibrate the aero/wave damage decomposition on the real damage
         c. roll out control episodes -> RL transitions
         d. apply the IoT sensor layer
         e. evaluate a full-factorial action sweep -> supervised surrogate set
    3. validate the outputs
    4. write parquet datasets, CSV samples, calibration reports and a manifest

Usage
-----
    python -m fowt_rl.build_dataset                      # full build
    python -m fowt_rl.build_dataset --calibrate-only     # just fit + report
    python -m fowt_rl.build_dataset --episodes 200 --no-sweep   # quick build
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .aero import load_aero
from .config import PipelineConfig
from .floatbench import load_tower
from .iot import apply_sensor_layer, measurement_error_report
from .load_model import calibrate_tower, save_models
from .mdp import (
    ACTION_COLUMNS,
    OBSERVATION_CHANNELS,
    PROPRIOCEPTIVE_COLUMNS,
    build_action_sweep,
    build_transitions,
)
from .turbine import TOWERS


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_dir() -> Path:
    return repo_root() / "data" / "processed"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_transitions(frame: pd.DataFrame, config: PipelineConfig) -> dict:
    """Structural and physical checks on a transitions table."""
    checks: dict[str, object] = {}
    failures: list[str] = []

    # No missing values in the numeric payload.
    numeric = frame.select_dtypes(include=[np.number])
    n_nan = int(numeric.isna().to_numpy().sum())
    checks["n_nan_numeric"] = n_nan
    if n_nan:
        failures.append(f"{n_nan} NaN values in numeric columns")

    # Every episode must terminate exactly once, on its last step.
    done_per_episode = frame.groupby("episode_id")["done"].sum()
    checks["episodes_with_single_done"] = bool((done_per_episode == 1).all())
    if not checks["episodes_with_single_done"]:
        failures.append("some episodes do not terminate exactly once")

    # Reward must equal its decomposition.
    reconstructed = (
        frame["reward_fatigue_term"]
        - config.reward.power_weight * frame["reward_power_loss_fraction"]
        - config.reward.duty_weight * frame["reward_duty_total"]
    )
    max_reward_error = float(np.max(np.abs(reconstructed - frame["reward"])))
    checks["max_reward_decomposition_error"] = max_reward_error
    if max_reward_error > 1e-9:
        failures.append(f"reward decomposition mismatch up to {max_reward_error:.2e}")

    # Actions must respect the declared bounds.
    space = config.action_space
    bounds = {
        "action_pitch_offset_deg": space.pitch_offset_bounds,
        "action_yaw_setpoint_deg": space.yaw_setpoint_bounds,
        "action_ipc_level": space.ipc_level_bounds,
    }
    for column, (low, high) in bounds.items():
        within = bool(frame[column].between(low - 1e-9, high + 1e-9).all())
        checks[f"{column}_within_bounds"] = within
        if not within:
            failures.append(f"{column} outside {low}..{high}")

    # Damage relief cannot exceed the controllable share of the load.
    # damage_ratio >= 1 - controllable_share, allowing a small numerical margin.
    floor = 1.0 - frame["controllable_share_max_section"]
    violation = float(np.max(np.clip(floor - frame["damage_ratio"], 0.0, None)))
    checks["max_controllable_share_violation"] = violation
    if violation > 1e-6:
        failures.append(f"damage reduced below the wave-driven floor by {violation:.2e}")

    # The IoT layer must have produced a measurement for every channel.
    missing = [
        channel
        for channel in OBSERVATION_CHANNELS
        if f"meas_{channel}" not in frame.columns
    ]
    checks["missing_measured_channels"] = missing
    if missing:
        failures.append(f"missing measured channels: {missing}")

    checks["passed"] = not failures
    checks["failures"] = failures
    return checks


def validate_zero_action_exactness(config: PipelineConfig) -> dict:
    """Confirm that a zero control action reproduces FLOATBench damage exactly.

    This is the load-bearing property of the whole approach: the synthetic layer
    must never overwrite measured data.
    """
    from .damage import damage_under_action
    from .load_model import load_models

    aero = load_aero(write_report=False)
    models, _ = load_models()
    results = {}
    for tower in config.towers:
        tower_data = load_tower(tower)
        n = len(tower_data.conditions)
        zeros = np.zeros(n)
        outcome = damage_under_action(
            tower_data.conditions,
            tower_data.damage,
            models[tower],
            aero,
            zeros,
            zeros,
            zeros,
            config.authority,
        )
        results[tower] = {
            "max_abs_damage_difference": float(
                np.max(np.abs(outcome["damage"] - tower_data.damage))
            ),
            "max_abs_ratio_deviation": float(np.max(np.abs(outcome["damage_ratio"] - 1.0))),
            "n_conditions": int(n),
            "n_sections": int(tower_data.damage.shape[1]),
        }
    results["passed"] = all(
        entry["max_abs_damage_difference"] == 0.0
        for key, entry in results.items()
        if isinstance(entry, dict)
    )
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _downcast(frame: pd.DataFrame) -> pd.DataFrame:
    """Store floats as float32 to keep the published parquet files compact."""
    out = frame.copy()
    for column in out.select_dtypes(include=["float64"]).columns:
        out[column] = out[column].astype("float32")
    for column in out.select_dtypes(include=["int64"]).columns:
        values = out[column]
        if values.abs().max() < np.iinfo(np.int32).max:
            out[column] = values.astype("int32")
    return out


def _display_path(path: Path) -> str:
    """Path relative to the repo root when possible, else absolute."""
    try:
        return str(path.resolve().relative_to(repo_root()))
    except ValueError:
        return str(path.resolve())


def _write_parquet(frame: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    return {
        "path": _display_path(path),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "bytes": int(path.stat().st_size),
    }


def _write_sample(frame: pd.DataFrame, path: Path, n_rows: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = frame.head(n_rows)
    sample.to_csv(path, index=False)
    return {
        "path": _display_path(path),
        "rows": int(len(sample)),
        "bytes": int(path.stat().st_size),
    }


def observation_manifest() -> dict:
    """Declare which columns a policy may read, and which are privileged."""
    return {
        "measured_observation": [f"meas_{channel}" for channel in OBSERVATION_CHANNELS],
        "measurement_validity": [f"valid_{channel}" for channel in OBSERVATION_CHANNELS]
        + ["sensor_health"],
        "proprioceptive_observation": list(PROPRIOCEPTIVE_COLUMNS),
        "ground_truth_observation": [f"true_{channel}" for channel in OBSERVATION_CHANNELS],
        "next_ground_truth_observation": [
            f"next_true_{channel}" for channel in OBSERVATION_CHANNELS
        ],
        "actions": list(ACTION_COLUMNS),
        "reward": ["reward"],
        "reward_terms": [
            "reward_fatigue_relief",
            "reward_severity",
            "reward_fatigue_term",
            "reward_power_loss_fraction",
            "reward_duty_total",
        ],
        "termination": ["done"],
        "privileged_diagnostics": [
            "damage_controlled",
            "damage_baseline",
            "damage_ratio",
            "del_ratio",
            "controllable_share_max_section",
            "controllable_share_mean",
            "inflow_direction_deg",
            "vane_bias_deg",
            "gain_thrust_turbulence",
            "gain_rotor_cyclic",
        ],
        "note": (
            "Train policies on measured_observation + measurement_validity + "
            "proprioceptive_observation. The ground_truth and privileged columns "
            "exist for evaluation and ablation only; using them at training time "
            "leaks information no real controller would have."
        ),
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(
    config: PipelineConfig,
    output_dir: Path | None = None,
    calibrate_only: bool = False,
    sample_rows: int = 2000,
) -> dict:
    output_dir = output_dir or default_output_dir()
    started = time.time()

    print("[1/4] loading reference turbine aerodynamics")
    aero = load_aero()
    aero_check = aero.cross_validate()
    print(
        "      rotor performance surface vs steady states: "
        f"Ct r={aero_check['ct_pearson_r']:.4f}, Cp r={aero_check['cp_pearson_r']:.4f}"
    )

    models = {}
    reports = {}
    print("[2/4] calibrating load decomposition on FLOATBench damage")
    tower_data_cache = {}
    for tower in config.towers:
        tower_data = load_tower(tower)
        tower_data_cache[tower] = tower_data
        model, report = calibrate_tower(tower_data, aero, config.load_model)
        models[tower] = model
        reports[tower] = report
        quality = report["fit_quality"]
        share = report["controllable_share"]
        print(
            f"      {tower:<5} R2(DEL)={quality['mean_r2_del']:.3f} "
            f"(min {quality['min_r2_del']:.3f})  "
            f"DEL median rel.err={quality['del_median_abs_rel_error']*100:.1f}%  "
            f"controllable share mean={share['mean']:.3f} p95={share['p95']:.3f}"
        )
    calibration_path = save_models(models, reports)
    print(f"      wrote {_display_path(calibration_path)}")

    manifest: dict = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package_version": __import__("fowt_rl").__version__,
        "sources": {
            "floatbench": {
                "dataset": "DeCoDELab/FLOATBench (Hugging Face)",
                "paper": "arXiv:2605.25717",
                "license": "CC-BY-4.0",
            },
            "reference_turbine": {
                "repository": "IEAWindSystems/IEA-22-280-RWT",
                "license": "Apache-2.0",
                "artefacts": [
                    "OpenFAST steady-state operating points",
                    "Cp/Ct/Cq rotor performance surfaces",
                    "IEA-22-280-RWT-Semi_DISCON.IN actuator limits",
                ],
            },
        },
        "aero_reference_check": aero_check,
        "aero_pitch_sensitivity": aero.pitch_sensitivity_report(),
        "calibration": reports,
        "observation_manifest": observation_manifest(),
        "outputs": {"transitions": {}, "action_sweep": {}, "samples": {}},
    }

    if calibrate_only:
        manifest["zero_action_exactness"] = validate_zero_action_exactness(config)
        _finalise(manifest, config, output_dir, started)
        return manifest

    print("[3/4] rolling out control episodes and evaluating action sweeps")
    transition_frames = []
    sweep_frames = []
    for tower in config.towers:
        tower_data = tower_data_cache[tower]
        model = models[tower]

        frame = build_transitions(
            tower_data,
            model,
            aero,
            config.action_space,
            config.episodes,
            config.reward,
            config.duty,
            config.authority,
        )
        frame = apply_sensor_layer(frame, config.iot)
        checks = validate_transitions(frame, config)
        if not checks["passed"]:
            raise RuntimeError(f"{tower}: transition validation failed: {checks['failures']}")

        frame = _downcast(frame)
        info = _write_parquet(frame, output_dir / "transitions" / f"transitions_{tower}.parquet")
        info["checks"] = checks
        info["sensor_error"] = measurement_error_report(frame, config.iot)
        info["mean_reward_by_policy"] = (
            frame.groupby("behaviour_policy")["reward"].mean().round(6).to_dict()
        )
        manifest["outputs"]["transitions"][tower] = info
        transition_frames.append(frame)
        print(
            f"      {tower:<5} transitions rows={info['rows']:,} "
            f"cols={info['columns']} size={info['bytes']/1e6:.1f} MB"
        )

        if config.sweep.enabled:
            sweep = build_action_sweep(
                tower_data,
                model,
                aero,
                config.action_space,
                n_pitch=config.sweep.n_pitch,
                n_yaw=config.sweep.n_yaw,
                n_ipc=config.sweep.n_ipc,
                condition_fraction=config.sweep.condition_fraction,
                seed=config.sweep.seed,
                authority=config.authority,
            )
            sweep = _downcast(sweep)
            sweep_info = _write_parquet(
                sweep, output_dir / "action_sweep" / f"action_sweep_{tower}.parquet"
            )
            manifest["outputs"]["action_sweep"][tower] = sweep_info
            sweep_frames.append(sweep)
            print(
                f"      {tower:<5} action sweep rows={sweep_info['rows']:,} "
                f"cols={sweep_info['columns']} size={sweep_info['bytes']/1e6:.1f} MB"
            )

    print("[4/4] writing samples, validation and manifest")
    combined_transitions = pd.concat(transition_frames, ignore_index=True)
    manifest["outputs"]["samples"]["transitions"] = _write_sample(
        combined_transitions.sample(frac=1.0, random_state=0).sort_values(
            ["tower", "episode_id", "step"]
        ),
        output_dir / "samples" / "transitions_sample.csv",
        sample_rows,
    )
    if sweep_frames:
        combined_sweep = pd.concat(sweep_frames, ignore_index=True)
        manifest["outputs"]["samples"]["action_sweep"] = _write_sample(
            combined_sweep,
            output_dir / "samples" / "action_sweep_sample.csv",
            sample_rows,
        )

    manifest["zero_action_exactness"] = validate_zero_action_exactness(config)
    manifest["dataset_summary"] = _summarise(combined_transitions)
    _finalise(manifest, config, output_dir, started)
    return manifest


def _summarise(frame: pd.DataFrame) -> dict:
    return {
        "total_transitions": int(len(frame)),
        "towers": sorted(frame["tower"].unique().tolist()),
        "episodes": int(frame.groupby(["tower", "episode_id"]).ngroups),
        "steps_per_episode": int(frame["step"].max() + 1),
        "step_seconds": float(frame["step_seconds"].iloc[0]),
        "simulated_hours": float(
            frame.groupby(["tower", "episode_id"]).ngroups
            * (frame["step"].max() + 1)
            * frame["step_seconds"].iloc[0]
            / 3600.0
        ),
        "reward": {
            "mean": float(frame["reward"].mean()),
            "std": float(frame["reward"].std()),
            "min": float(frame["reward"].min()),
            "max": float(frame["reward"].max()),
            "fraction_positive": float((frame["reward"] > 0).mean()),
        },
        "del_ratio": {
            "min": float(frame["del_ratio"].min()),
            "p05": float(frame["del_ratio"].quantile(0.05)),
            "median": float(frame["del_ratio"].median()),
            "max": float(frame["del_ratio"].max()),
        },
        "controllable_share_max_section": {
            "mean": float(frame["controllable_share_max_section"].mean()),
            "p05": float(frame["controllable_share_max_section"].quantile(0.05)),
            "p95": float(frame["controllable_share_max_section"].quantile(0.95)),
        },
        "behaviour_policy_counts": frame["behaviour_policy"].value_counts().to_dict(),
    }


def _finalise(manifest: dict, config: PipelineConfig, output_dir: Path, started: float) -> None:
    manifest["build_seconds"] = round(time.time() - started, 2)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "pipeline_config.json")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(f"      manifest -> {_display_path(output_dir / 'manifest.json')}")
    print(f"      done in {manifest['build_seconds']:.1f} s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--towers", nargs="+", default=list(TOWERS), choices=list(TOWERS))
    parser.add_argument("--episodes", type=int, default=None, help="episodes per tower")
    parser.add_argument("--steps", type=int, default=None, help="steps per episode")
    parser.add_argument("--no-sweep", action="store_true", help="skip the action sweep")
    parser.add_argument("--no-iot", action="store_true", help="disable the sensor layer")
    parser.add_argument("--iot-severity", type=float, default=None, help="sensor noise multiplier")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample-rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    config = PipelineConfig(towers=tuple(args.towers))
    if args.episodes is not None:
        config.episodes = replace(config.episodes, episodes_per_tower=args.episodes)
    if args.steps is not None:
        config.episodes = replace(config.episodes, steps_per_episode=args.steps)
    if args.seed is not None:
        config.episodes = replace(config.episodes, seed=args.seed)
    if args.no_sweep:
        config.sweep = replace(config.sweep, enabled=False)
    if args.no_iot:
        config.iot.enabled = False
    if args.iot_severity is not None:
        config.iot.severity = args.iot_severity

    build(
        config,
        output_dir=args.out_dir,
        calibrate_only=args.calibrate_only,
        sample_rows=args.sample_rows,
    )


if __name__ == "__main__":
    main()
