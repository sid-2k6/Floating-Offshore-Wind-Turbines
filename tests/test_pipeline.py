"""Tests for the load-relief dataset pipeline.

The tests that matter most are the *invariants*: a zero control action must
reproduce FLOATBench damage exactly, and no action may reduce damage below the
wave-driven floor. Those two properties are what let the dataset claim to be
anchored on measured data rather than replacing it.

Tests that need the FLOATBench download are skipped automatically when it is
absent, so the aerodynamic tests still run in a bare checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

from fowt_rl.actions import ActionSpace, actuator_duty
from fowt_rl.aero import load_aero
from fowt_rl.config import EpisodeConfig, PipelineConfig, RewardConfig
from fowt_rl.damage import damage_under_action, load_path_gains, summarise_sections
from fowt_rl.floatbench import default_raw_dir, load_tower
from fowt_rl.load_model import calibrate_tower
from fowt_rl.mdp import build_condition_index, build_transitions
from fowt_rl.turbine import TOWERS, load_turbine

TOWER = "opt2"

floatbench_available = (default_raw_dir() / TOWER / "data.csv").exists()
requires_floatbench = pytest.mark.skipif(
    not floatbench_available,
    reason="FLOATBench not downloaded; run python scripts/download_floatbench.py",
)


# ---------------------------------------------------------------------------
# Reference turbine data
# ---------------------------------------------------------------------------
def test_reference_turbine_properties():
    properties, schedule = load_turbine()
    assert properties.rotor_diameter_m == pytest.approx(284.0)
    assert properties.hub_height_m == pytest.approx(170.0)
    assert properties.rated_power_w == pytest.approx(22e6)
    assert len(schedule) == 22
    # Electrical power must be monotone non-decreasing then flat at rated.
    power = schedule.table["electrical_power"].to_numpy()
    assert power.max() <= properties.rated_power_w * 1.005
    assert 9.0 < schedule.rated_wind_speed_ms < 14.0


def test_performance_surface_is_consistent_with_steady_states():
    aero = load_aero(write_report=False)
    check = aero.cross_validate()
    # The two official artefacts are produced by different solvers, so we only
    # require strong rank agreement - ratio anchoring removes any level offset.
    assert check["ct_pearson_r"] > 0.95
    assert check["cp_pearson_r"] > 0.95


# ---------------------------------------------------------------------------
# Aerodynamic action response
# ---------------------------------------------------------------------------
def test_zero_action_reproduces_reference_operating_point():
    aero = load_aero(write_report=False)
    wind = np.array([5.0, 8.0, 11.0, 15.0, 20.0, 24.0])
    response = aero.response(wind, 0.0, 0.0)
    assert np.allclose(response["ct_ratio"], 1.0)
    assert np.allclose(response["thrust_n"], aero.schedule.thrust_n(wind))


def test_feathering_never_increases_thrust_or_power():
    aero = load_aero(write_report=False)
    wind = np.linspace(4.0, 24.0, 21)
    baseline = aero.response(wind, 0.0, 0.0)
    for offset in (1.0, 2.0, 4.0, 8.0):
        perturbed = aero.response(wind, offset, 0.0)
        assert np.all(perturbed["thrust_n"] <= baseline["thrust_n"] + 1e-6)
        assert np.all(perturbed["electrical_power_w"] <= baseline["electrical_power_w"] + 1e-6)


def test_thrust_decreases_monotonically_with_pitch_offset():
    aero = load_aero(write_report=False)
    offsets = np.linspace(0.0, 8.0, 17)
    for wind in (7.0, 11.0, 16.0, 22.0):
        thrust = aero.response(np.full_like(offsets, wind), offsets, 0.0)["thrust_n"]
        assert np.all(np.diff(thrust) <= 1e-6), f"thrust not monotone at V={wind}"


def test_yaw_follows_cosine_power_laws():
    aero = load_aero(write_report=False)
    wind = np.full(4, 12.0)
    yaw = np.array([0.0, 10.0, 20.0, 30.0])
    response = aero.response(wind, 0.0, yaw)
    expected = np.cos(np.radians(yaw)) ** 2.0
    assert np.allclose(response["ct_ratio"], expected, rtol=1e-6)


def test_pitch_sensitivity_is_physically_plausible():
    """Above rated, dP/dtheta should be of order -1 MW/deg for a 22 MW rotor."""
    aero = load_aero(write_report=False)
    report = aero.pitch_sensitivity_report((14.0, 18.0, 22.0))
    for entry in report.values():
        assert -8.0 < entry["dP_dpitch_MW_per_deg"] < -0.5


# ---------------------------------------------------------------------------
# Action space and duty
# ---------------------------------------------------------------------------
def test_action_space_rate_limiting_and_bounds():
    space = ActionSpace()
    previous = np.array([[0.0, 0.0, 0.0]])
    requested = np.array([[100.0, 100.0, 100.0]])
    limited = space.apply_rate_limits(previous, requested)
    assert limited[0, 0] == pytest.approx(space.max_pitch_offset_change_deg)
    assert limited[0, 1] == pytest.approx(space.max_yaw_setpoint_change_deg)
    assert limited[0, 2] == pytest.approx(space.max_ipc_level_change)

    rng = np.random.default_rng(0)
    sampled = space.sample(500, rng)
    assert np.all(sampled >= space.low - 1e-9)
    assert np.all(sampled <= space.high + 1e-9)
    # The sampler must include the exact zero-action baseline.
    assert np.any(np.all(np.isclose(sampled, 0.0), axis=1))


def test_yaw_deadband_suppresses_small_commands():
    space = ActionSpace()
    previous = np.array([[0.0, 0.0, 0.0]])
    small = np.array([[0.0, space.limits.yaw_deadband_deg - 1.0, 0.0]])
    large = np.array([[0.0, space.limits.yaw_deadband_deg + 6.0, 0.0]])
    assert actuator_duty(small, previous, space, 600.0)["duty_yaw_engagement"][0] == 0.0
    assert actuator_duty(large, previous, space, 600.0)["duty_yaw_engagement"][0] > 0.0


def test_zero_action_has_zero_duty():
    space = ActionSpace()
    zero = np.zeros((1, 3))
    duty = actuator_duty(zero, zero, space, 600.0)
    assert duty["duty_total"][0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Load-path gains
# ---------------------------------------------------------------------------
def test_load_path_gains_leave_wave_columns_untouched():
    gains = load_path_gains(
        np.array([0.5, 1.0]), np.array([0.0, 20.0]), np.array([0.0, 1.0])
    )
    assert np.allclose(gains[:, 2], 1.0)
    assert np.allclose(gains[:, 3], 1.0)


def test_yaw_raises_cyclic_gain_while_ipc_lowers_it():
    aligned = load_path_gains(np.array([1.0]), np.array([0.0]), np.array([0.0]))
    yawed = load_path_gains(np.array([1.0]), np.array([25.0]), np.array([0.0]))
    with_ipc = load_path_gains(np.array([1.0]), np.array([0.0]), np.array([1.0]))
    assert yawed[0, 1] > aligned[0, 1]
    assert with_ipc[0, 1] < aligned[0, 1]


# ---------------------------------------------------------------------------
# Calibration and damage (need FLOATBench)
# ---------------------------------------------------------------------------
@requires_floatbench
def test_floatbench_reshaping():
    tower_data = load_tower(TOWER)
    assert tower_data.damage.shape == (6468, 30)
    assert len(tower_data.conditions) == 6468
    assert tower_data.conditions["sim_id"].is_unique
    assert np.all(tower_data.damage > 0)


@requires_floatbench
def test_condition_grid_is_complete():
    index = build_condition_index(load_tower(TOWER))
    assert index.shape == (22, 7, 7, 6)
    assert index.min() >= 0
    assert np.unique(index).size == index.size


@requires_floatbench
def test_load_model_fit_quality_and_shares():
    aero = load_aero(write_report=False)
    model, report = calibrate_tower(load_tower(TOWER), aero)
    assert report["fit_quality"]["mean_r2_del"] > 0.6
    assert np.all(model.coefficients >= 0.0)
    share = report["controllable_share"]
    # Control authority must be partial: waves are not controllable, and the
    # aerodynamic path is not negligible either.
    assert 0.05 < share["mean"] < 0.9
    assert share["p95"] > share["p05"]


@requires_floatbench
def test_zero_action_reproduces_floatbench_damage_exactly():
    """The load-bearing invariant: measured damage is never overwritten."""
    aero = load_aero(write_report=False)
    tower_data = load_tower(TOWER)
    model, _ = calibrate_tower(tower_data, aero)
    zeros = np.zeros(len(tower_data.conditions))
    result = damage_under_action(
        tower_data.conditions, tower_data.damage, model, aero, zeros, zeros, zeros
    )
    assert np.array_equal(result["damage"], tower_data.damage)
    assert np.all(result["damage_ratio"] == 1.0)


@requires_floatbench
def test_damage_never_falls_below_the_wave_floor():
    """No control action may reduce damage past the uncontrollable share."""
    aero = load_aero(write_report=False)
    tower_data = load_tower(TOWER)
    model, _ = calibrate_tower(tower_data, aero)
    n = len(tower_data.conditions)
    result = damage_under_action(
        tower_data.conditions,
        tower_data.damage,
        model,
        aero,
        np.full(n, 8.0),  # maximum feathering
        np.zeros(n),
        np.ones(n),  # full IPC
    )
    floor = 1.0 - result["controllable_share"]
    assert np.all(result["damage_ratio"] >= floor - 1e-9)


@requires_floatbench
def test_section_summary_consistency():
    tower_data = load_tower(TOWER)
    summary = summarise_sections(tower_data.damage)
    assert np.all(summary["damage_max"] >= summary["damage_mean"])
    assert np.all(summary["damage_sum"] >= summary["damage_max"])
    assert np.all((summary["damage_argmax_section"] >= 1) & (summary["damage_argmax_section"] <= 30))


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------
@requires_floatbench
def test_transitions_structure_and_reward_decomposition():
    aero = load_aero(write_report=False)
    tower_data = load_tower(TOWER)
    model, _ = calibrate_tower(tower_data, aero)
    config = PipelineConfig()
    episodes = EpisodeConfig(episodes_per_tower=25, steps_per_episode=12)
    reward_config = RewardConfig()

    frame = build_transitions(
        tower_data, model, aero, config.action_space, episodes, reward_config
    )

    assert len(frame) == 25 * 12
    assert frame.groupby("episode_id")["done"].sum().eq(1).all()
    assert frame.select_dtypes(include=[np.number]).notna().to_numpy().all()

    reconstructed = (
        frame["reward_fatigue_term"]
        - reward_config.power_weight * frame["reward_power_loss_fraction"]
        - reward_config.duty_weight * frame["reward_duty_total"]
    )
    assert np.allclose(reconstructed, frame["reward"], atol=1e-12)

    # The baseline behaviour policy must take literally no action.
    baseline = frame[frame["behaviour_policy"] == "baseline"]
    if len(baseline):
        assert np.allclose(baseline["action_pitch_offset_deg"], 0.0)
        assert np.allclose(baseline["action_ipc_level"], 0.0)


@requires_floatbench
def test_all_towers_calibrate():
    aero = load_aero(write_report=False)
    for tower in TOWERS:
        if not (default_raw_dir() / tower / "data.csv").exists():
            pytest.skip(f"{tower} not downloaded")
        _, report = calibrate_tower(load_tower(tower), aero)
        assert report["fit_quality"]["mean_r2_del"] > 0.6
