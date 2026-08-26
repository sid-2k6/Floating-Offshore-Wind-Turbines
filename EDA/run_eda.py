"""End-to-end exploratory data analysis of the FOWT load-relief RL dataset.

Produces every figure and table referenced by EDA/EDA_REPORT.md:

    EDA/figures/*.png    18 figures, grouped into six analysis sections
    EDA/tables/*.csv     supporting numeric tables
    EDA/summary_stats.json   machine-readable summary of every headline number

Run from the repository root:

    PYTHONPATH=src python EDA/run_eda.py

Sections
--------
    A  source data          what FLOATBench actually contains
    B  reference aerodynamics   the measured basis of the action response
    C  calibration            how well the load decomposition fits
    D  action response        the action sweep - does control behave sensibly
    E  RL dataset             reward structure, coverage, episode dynamics
    F  IoT layer + quality    sensor realism and data integrity
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm

from fowt_rl.aero import load_aero
from fowt_rl.floatbench import load_tower
from fowt_rl.load_model import BASIS_NAMES, load_models
from fowt_rl.turbine import TOWER_FA1_HZ, TOWERS

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
EDA = ROOT / "EDA"
FIG = EDA / "figures"
TAB = EDA / "tables"

TOWER_COLOURS = {"ref": "#c0392b", "opt1": "#2471a3", "opt2": "#1e8449"}
POLICY_COLOURS = {
    "baseline": "#5d6d7e",
    "ipc_only": "#1e8449",
    "feather": "#b9770e",
    "random": "#7d3c98",
    "yaw_seeker": "#c0392b",
}
WIND_BINS = [0, 6, 9, 12, 16, 20, 30]
WIND_LABELS = ["3-6", "6-9", "9-12", "12-16", "16-20", "20-25"]

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

STATS: dict = {}


def save(fig: plt.Figure, name: str) -> None:
    path = FIG / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure  {path.relative_to(ROOT)}")


def table(frame: pd.DataFrame, name: str, index: bool = True) -> None:
    path = TAB / name
    frame.to_csv(path, index=index)
    print(f"  table   {path.relative_to(ROOT)}")


def load_all_data() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    towers = {tower: load_tower(tower) for tower in TOWERS}
    transitions = pd.concat(
        [
            pd.read_parquet(ROOT / "data/processed/transitions" / f"transitions_{tower}.parquet")
            for tower in TOWERS
        ],
        ignore_index=True,
    )
    sweep = pd.concat(
        [
            pd.read_parquet(ROOT / "data/processed/action_sweep" / f"action_sweep_{tower}.parquet")
            for tower in TOWERS
        ],
        ignore_index=True,
    )
    return towers, transitions, sweep


# ===========================================================================
# SECTION A - source data
# ===========================================================================
def section_a(towers: dict) -> None:
    print("[A] source data")

    # --- A1 damage distribution and section profile ---------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    for tower, data in towers.items():
        values = data.damage.ravel()
        axes[0].hist(
            np.log10(values),
            bins=90,
            histtype="step",
            linewidth=1.6,
            label=f"{tower} (n={values.size:,})",
            color=TOWER_COLOURS[tower],
        )
    axes[0].set(
        xlabel="log10 fatigue damage per 10-min run",
        ylabel="count",
        title="A1a  Damage spans ~6 orders of magnitude",
    )
    axes[0].legend()

    for tower, data in towers.items():
        height = data.sections["section_height_m"].to_numpy()
        weighted = (data.damage * data.conditions["damage_weight"].to_numpy()[:, None]).sum(axis=0)
        axes[1].semilogx(weighted, height, color=TOWER_COLOURS[tower], linewidth=1.8, label=tower)
    axes[1].set(
        xlabel="lifetime-weighted damage (25 yr)",
        ylabel="section height above tower base [m]",
        title="A1b  Lifetime damage profile",
    )
    axes[1].legend()

    for tower, data in towers.items():
        height = data.sections["section_height_m"].to_numpy()
        axes[2].plot(
            data.sections["section_thickness_m"] * 1000,
            height,
            color=TOWER_COLOURS[tower],
            linewidth=1.8,
            label=f"{tower} (FA1={TOWER_FA1_HZ[tower]} Hz)",
        )
    axes[2].set(
        xlabel="wall thickness [mm]",
        ylabel="section height [m]",
        title="A1c  Tower geometry differs by design",
    )
    axes[2].legend()
    fig.suptitle("A1  FLOATBench source data: damage and tower geometry", y=1.04, fontsize=11)
    save(fig, "A1_damage_and_geometry.png")

    # --- A2 operating envelope ------------------------------------------
    data = towers["opt2"]
    conditions = data.conditions
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))

    hb = axes[0].hexbin(
        conditions["mean_wind_speed"], conditions["wave_hs"], gridsize=32, cmap="viridis", mincnt=1
    )
    fig.colorbar(hb, ax=axes[0], label="conditions")
    axes[0].set(xlabel="mean wind speed [m/s]", ylabel="Hs [m]", title="A2a  Wind-wave envelope")

    hb = axes[1].hexbin(
        conditions["wave_tp"], conditions["wave_hs"], gridsize=32, cmap="viridis", mincnt=1
    )
    fig.colorbar(hb, ax=axes[1], label="conditions")
    axes[1].set(xlabel="Tp [s]", ylabel="Hs [m]", title="A2b  Sea-state space")

    intensity = conditions["std_wind_speed"] / conditions["mean_wind_speed"]
    axes[2].scatter(
        conditions["mean_wind_speed"], intensity, s=3, alpha=0.15, color="#2471a3", edgecolors="none"
    )
    axes[2].set(
        xlabel="mean wind speed [m/s]",
        ylabel="turbulence intensity [-]",
        title="A2c  TI decays with wind speed",
        ylim=(0, 0.45),
    )

    axes[3].scatter(
        conditions["mean_wind_speed"],
        conditions["damage_weight"],
        s=3,
        alpha=0.2,
        color="#b9770e",
        edgecolors="none",
    )
    axes[3].set(
        xlabel="mean wind speed [m/s]",
        ylabel="damage_weight (25-yr occurrence)",
        title="A2d  Severe states are rare",
    )
    fig.suptitle(
        "A2  Operating envelope: 22 wind x 7 Hs x 7 Tp, wind and waves aligned",
        y=1.05,
        fontsize=11,
    )
    save(fig, "A2_operating_envelope.png")

    rows = []
    for tower, data in towers.items():
        values = data.damage
        weighted = (values * data.conditions["damage_weight"].to_numpy()[:, None]).sum(axis=0)
        rows.append(
            {
                "tower": tower,
                "fa1_hz": TOWER_FA1_HZ[tower],
                "n_conditions": len(data.conditions),
                "n_labels": values.size,
                "damage_min": values.min(),
                "damage_median": np.median(values),
                "damage_max": values.max(),
                "damage_orders_of_magnitude": np.log10(values.max() / values.min()),
                "lifetime_damage_base": weighted[0],
                "lifetime_damage_top": weighted[-1],
                "governing_section": int(np.argmax(weighted) + 1),
            }
        )
    summary = pd.DataFrame(rows).set_index("tower")
    table(summary, "A_source_data_summary.csv")
    STATS["A_source_data"] = summary.to_dict(orient="index")

    envelope = towers["opt2"].conditions
    STATS["A_envelope"] = {
        "wind_speed_range": [float(envelope.mean_wind_speed.min()), float(envelope.mean_wind_speed.max())],
        "hs_range": [float(envelope.wave_hs.min()), float(envelope.wave_hs.max())],
        "tp_range": [float(envelope.wave_tp.min()), float(envelope.wave_tp.max())],
        "turbulence_intensity_p01_p50_p99": np.percentile(
            envelope.std_wind_speed / envelope.mean_wind_speed, [1, 50, 99]
        ).tolist(),
    }


# ===========================================================================
# SECTION B - reference aerodynamics
# ===========================================================================
def section_b() -> None:
    print("[B] reference aerodynamics")
    aero = load_aero(write_report=False)
    schedule = aero.schedule.table
    surface = aero.surface

    # --- B1 baseline schedule -------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2))
    plots = [
        ("pitch", "baseline blade pitch [deg]", "B1a  Pitch schedule"),
        ("rotor_speed", "rotor speed [rpm]", "B1b  Rotor speed"),
        ("tip_speed_ratio", "tip-speed ratio [-]", "B1c  TSR"),
        ("rotor_thrust", "rotor thrust [MN]", "B1d  Thrust peaks near rated"),
        ("electrical_power", "electrical power [MW]", "B1e  Power curve"),
        ("ct_base", "Ct [-]", "B1f  Thrust coefficient collapses"),
    ]
    for ax, (column, ylabel, title) in zip(axes.ravel(), plots):
        values = schedule[column]
        if column == "rotor_thrust":
            values = values / 1e6
        if column == "electrical_power":
            values = values / 1e6
        ax.plot(schedule["wind_speed"], values, "o-", color="#1a5276", markersize=3.5, linewidth=1.6)
        ax.axvline(aero.schedule.rated_wind_speed_ms, color="#c0392b", linestyle="--", linewidth=1)
        ax.set(xlabel="wind speed [m/s]", ylabel=ylabel, title=title)
    axes[0, 0].text(
        aero.schedule.rated_wind_speed_ms + 0.4,
        axes[0, 0].get_ylim()[1] * 0.75,
        "rated",
        color="#c0392b",
        fontsize=8,
    )
    fig.suptitle(
        "B1  Official IEA-22-280-RWT baseline operating schedule (22 OpenFAST steady states)",
        y=1.01,
        fontsize=11,
    )
    save(fig, "B1_baseline_schedule.png")

    # --- B2 rotor performance surfaces ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    pitch_grid, tsr_grid = np.meshgrid(surface.pitch_deg, surface.tsr, indexing="ij")
    for ax, (grid, name, levels) in zip(
        axes,
        [
            (surface.ct, "Ct", np.linspace(0, 1.4, 15)),
            (surface.cp, "Cp", np.linspace(-0.2, 0.55, 16)),
        ],
    ):
        contour = ax.contourf(pitch_grid, tsr_grid, grid, levels=levels, cmap="RdYlBu_r", extend="both")
        fig.colorbar(contour, ax=ax, label=name)
        ax.contour(pitch_grid, tsr_grid, grid, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
        ax.plot(
            schedule["pitch"],
            schedule["tip_speed_ratio"],
            "k-o",
            markersize=3,
            linewidth=1.8,
            label="baseline operating curve",
        )
        ax.set(xlabel="blade pitch [deg]", ylabel="tip-speed ratio [-]", title=f"B2  {name} surface")
        ax.legend(loc="upper right")
    fig.suptitle(
        "B2  Official 20x20 rotor performance surfaces - the measured basis of the pitch response",
        y=1.03,
        fontsize=11,
    )
    save(fig, "B2_rotor_performance_surface.png")

    # --- B3 action response ---------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    offsets = np.linspace(0, 8, 41)
    winds = [7.0, 9.0, 11.0, 14.0, 18.0, 22.0]
    cmap = plt.get_cmap("viridis")
    for i, wind in enumerate(winds):
        colour = cmap(i / (len(winds) - 1))
        response = aero.response(np.full_like(offsets, wind), offsets, 0.0)
        base = aero.response(np.array([wind]), 0.0, 0.0)
        axes[0].plot(
            offsets,
            response["thrust_n"] / base["thrust_n"][0],
            color=colour,
            linewidth=1.7,
            label=f"{wind:g} m/s",
        )
        axes[1].plot(
            offsets,
            response["electrical_power_w"] / max(base["electrical_power_w"][0], 1),
            color=colour,
            linewidth=1.7,
            label=f"{wind:g} m/s",
        )
    axes[0].set(
        xlabel="collective pitch offset [deg]",
        ylabel="thrust / baseline thrust",
        title="B3a  Feathering sheds thrust",
    )
    axes[1].set(
        xlabel="collective pitch offset [deg]",
        ylabel="power / baseline power",
        title="B3b  ...and costs power",
    )
    axes[0].legend(ncol=2, title="wind speed")

    yaw = np.linspace(0, 40, 81)
    axes[2].plot(yaw, np.cos(np.radians(yaw)) ** 2.0, linewidth=1.9, color="#c0392b", label="thrust ~ cos^2")
    axes[2].plot(
        yaw, np.cos(np.radians(yaw)) ** 1.88, linewidth=1.9, color="#1a5276", label="power ~ cos^1.88"
    )
    axes[2].axvspan(0, 8, color="#7d3c98", alpha=0.12)
    axes[2].text(1.0, 0.55, "yaw deadband", rotation=90, fontsize=7.5, color="#7d3c98")
    axes[2].set(
        xlabel="yaw misalignment [deg]",
        ylabel="ratio to aligned",
        title="B3c  Yaw cosine-power laws",
    )
    axes[2].legend()
    fig.suptitle("B3  Control action response of the rotor (ratio-anchored)", y=1.04, fontsize=11)
    save(fig, "B3_pitch_yaw_response.png")

    sensitivity = pd.DataFrame(aero.pitch_sensitivity_report((7, 9, 11, 14, 18, 22))).T
    table(sensitivity, "B_pitch_sensitivity.csv")
    STATS["B_aero"] = {
        "surface_vs_steady_state": aero.cross_validate(),
        "pitch_sensitivity": aero.pitch_sensitivity_report((7, 9, 11, 14, 18, 22)),
        "rated_wind_speed_ms": aero.schedule.rated_wind_speed_ms,
        "ct_range": [float(schedule.ct_base.min()), float(schedule.ct_base.max())],
        "cp_max": float(schedule.cp_base.max()),
    }



# ===========================================================================
# SECTION C - calibration quality
# ===========================================================================
def section_c(towers: dict) -> None:
    print("[C] load-model calibration")
    aero = load_aero(write_report=False)
    models, reports = load_models()

    # --- C1 fit quality --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    for tower, data in towers.items():
        model = models[tower]
        raw = model.raw_basis(data.conditions, aero)
        predicted = model.predict_damage(model.design_matrix(raw))
        observed = data.damage
        exponent = 1.0 / model.wohler_exponent
        sample = np.random.default_rng(0).choice(observed.size, 40_000, replace=False)
        axes[0].scatter(
            (observed.ravel()[sample]) ** exponent,
            (np.maximum(predicted.ravel()[sample], 0)) ** exponent,
            s=1.5,
            alpha=0.06,
            color=TOWER_COLOURS[tower],
            edgecolors="none",
            label=tower,
        )
    limits = axes[0].get_xlim()
    axes[0].plot(limits, limits, "k--", linewidth=1)
    axes[0].set(
        xlabel="observed DEL  (D^(1/3))",
        ylabel="predicted DEL",
        title="C1a  Predicted vs observed DEL",
    )
    legend = axes[0].legend(markerscale=6)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    for tower in TOWERS:
        model = models[tower]
        height = towers[tower].sections["section_height_m"]
        axes[1].plot(
            model.r2_del_per_section, height, "o-", markersize=3, color=TOWER_COLOURS[tower], label=tower
        )
    axes[1].set(
        xlabel="R^2 in DEL space",
        ylabel="section height [m]",
        title="C1b  Fit quality per section",
        xlim=(0.6, 0.9),
    )
    axes[1].legend()

    sweep_rows = []
    for tower in TOWERS:
        for entry in reports[tower]["hyperparameter_sweep"]:
            sweep_rows.append({"tower": tower, **entry})
    sweep_frame = pd.DataFrame(sweep_rows)
    for tower in TOWERS:
        subset = sweep_frame[sweep_frame.tower == tower].groupby("weight_exponent").mean_r2_del.max()
        axes[2].plot(
            subset.index, subset.values, "o-", markersize=4, color=TOWER_COLOURS[tower], label=tower
        )
    axes[2].set(
        xlabel="residual weighting exponent p",
        ylabel="mean R^2 (DEL)",
        title="C1c  p = 1/3 wins for all towers",
    )
    axes[2].legend()
    fig.suptitle(
        "C1  Load decomposition fitted on FLOATBench damage by non-negative least squares",
        y=1.03,
        fontsize=11,
    )
    save(fig, "C1_calibration_fit.png")

    # --- C2 load-path shares --------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    palette = ["#e67e22", "#c0392b", "#2471a3", "#5dade2"]
    for ax, tower in zip(axes, TOWERS):
        data = towers[tower]
        model = models[tower]
        design = model.design_matrix(model.raw_basis(data.conditions, aero))
        shares = model.path_shares(design).mean(axis=0)  # (30, 4)
        height = data.sections["section_height_m"].to_numpy()
        ax.stackplot(
            height,
            shares.T,
            labels=[name.replace("_", " ") for name in BASIS_NAMES],
            colors=palette,
            alpha=0.9,
        )
        controllable = shares[:, 0] + shares[:, 1]
        ax.plot(height, controllable, "k--", linewidth=1.4, label="controllable total")
        ax.set(
            xlabel="section height [m]",
            title=f"C2  {tower}  (FA1 = {TOWER_FA1_HZ[tower]} Hz)",
            ylim=(0, 1),
        )
    axes[0].set_ylabel("mean share of predicted damage")
    axes[0].legend(loc="upper center", ncol=2, fontsize=7.5)
    fig.suptitle(
        "C2  Which load paths drive tower damage - only the orange/red bands are controllable",
        y=1.03,
        fontsize=11,
    )
    save(fig, "C2_load_path_shares.png")

    # --- C3 controllable share across the envelope -----------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    data = towers["opt2"]
    model = models["opt2"]
    design = model.design_matrix(model.raw_basis(data.conditions, aero))
    controllable = model.controllable_share(design).max(axis=1)
    conditions = data.conditions

    grid = (
        pd.DataFrame(
            {
                "wind": conditions.wind_speed_id,
                "hs": conditions.wave_hs_id,
                "share": controllable,
            }
        )
        .pivot_table(index="hs", columns="wind", values="share", aggfunc="mean")
    )
    image = axes[0].imshow(
        grid.values, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
        extent=[0.5, 22.5, 0.5, 7.5],
    )
    fig.colorbar(image, ax=axes[0], label="controllable share")
    axes[0].set(
        xlabel="wind speed index (1 = 3.5 m/s ... 22 = 24.5 m/s)",
        ylabel="Hs index (1 = mildest)",
        title="C3a  Control authority over the envelope",
    )
    axes[0].grid(False)

    for tower in TOWERS:
        data_t = towers[tower]
        model_t = models[tower]
        design_t = model_t.design_matrix(model_t.raw_basis(data_t.conditions, aero))
        share_t = model_t.controllable_share(design_t).max(axis=1)
        binned = pd.Series(share_t).groupby(
            pd.cut(data_t.conditions.mean_wind_speed, WIND_BINS, labels=WIND_LABELS)
        , observed=True).mean()
        axes[1].plot(range(len(binned)), binned.values, "o-", color=TOWER_COLOURS[tower], label=tower)
    axes[1].set_xticks(range(len(WIND_LABELS)))
    axes[1].set_xticklabels(WIND_LABELS)
    axes[1].set(
        xlabel="wind speed band [m/s]",
        ylabel="mean controllable share",
        title="C3b  Authority peaks at 9-12 m/s",
    )
    axes[1].legend()

    for tower in TOWERS:
        data_t = towers[tower]
        model_t = models[tower]
        design_t = model_t.design_matrix(model_t.raw_basis(data_t.conditions, aero))
        share_t = model_t.controllable_share(design_t).max(axis=1)
        axes[2].hist(
            share_t, bins=60, histtype="step", linewidth=1.6, color=TOWER_COLOURS[tower], label=tower
        )
    axes[2].set(
        xlabel="controllable share (governing section)",
        ylabel="conditions",
        title="C3c  Highly bimodal: often near zero",
    )
    axes[2].legend()
    fig.suptitle(
        "C3  The controllable share bounds achievable load relief and varies strongly with state",
        y=1.04,
        fontsize=11,
    )
    save(fig, "C3_controllable_share.png")

    per_section = []
    for tower in TOWERS:
        model = models[tower]
        for index in range(len(model.r2_del_per_section)):
            per_section.append(
                {
                    "tower": tower,
                    "section_id": index + 1,
                    "section_height_m": towers[tower].sections.section_height_m.iloc[index],
                    "r2_del": model.r2_del_per_section[index],
                    "r2_damage": model.r2_damage_per_section[index],
                    **{
                        f"coef_{name}": model.coefficients[index, position]
                        for position, name in enumerate(BASIS_NAMES)
                    },
                }
            )
    table(pd.DataFrame(per_section), "C_calibration_per_section.csv", index=False)
    table(sweep_frame, "C_hyperparameter_sweep.csv", index=False)

    STATS["C_calibration"] = {
        tower: {
            "selected_damping_ratio": reports[tower]["selected_damping_ratio"],
            "selected_weight_exponent": reports[tower]["selected_weight_exponent"],
            "fit_quality": reports[tower]["fit_quality"],
            "path_share_mean": reports[tower]["path_share_mean"],
            "controllable_share": reports[tower]["controllable_share"],
        }
        for tower in TOWERS
    }


# ===========================================================================
# SECTION D - action response (sweep)
# ===========================================================================
def section_d(sweep: pd.DataFrame) -> None:
    print("[D] action response")
    subset = sweep[sweep.tower == "opt2"].copy()
    subset["wind_band"] = pd.cut(subset.mean_wind_speed, WIND_BINS, labels=WIND_LABELS)

    # --- D1 marginal action effects --------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    specs = [
        ("action_pitch_offset_deg", "collective pitch offset [deg]", "D1a  Pitch"),
        ("action_yaw_error_deg", "yaw misalignment [deg]", "D1b  Yaw"),
        ("action_ipc_level", "IPC level [-]", "D1c  IPC"),
    ]
    cmap = plt.get_cmap("viridis")
    for ax, (column, xlabel, title) in zip(axes, specs):
        others = [c for c, _, _ in specs if c != column]
        neutral = subset[(subset[others[0]] == 0) & (subset[others[1]] == 0)]
        for i, band in enumerate(WIND_LABELS):
            band_data = neutral[neutral.wind_band == band]
            if band_data.empty:
                continue
            curve = band_data.groupby(column).damage_ratio_max.mean()
            ax.plot(
                curve.index, curve.values, "o-", markersize=3.5,
                color=cmap(i / (len(WIND_LABELS) - 1)), linewidth=1.6, label=band,
            )
        ax.axhline(1.0, color="k", linestyle=":", linewidth=1)
        ax.set(xlabel=xlabel, ylabel="damage ratio vs baseline", title=title)
    axes[0].legend(title="wind [m/s]", ncol=2)
    fig.suptitle(
        "D1  Marginal effect of each action on governing-section damage (other actions held at zero)",
        y=1.05,
        fontsize=11,
    )
    save(fig, "D1_action_marginal_effects.png")

    # --- D2 damage-power trade-off ---------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    aligned = subset[subset.action_yaw_error_deg == 0]
    grouped = aligned.groupby(
        ["wind_band", "action_pitch_offset_deg", "action_ipc_level"], observed=True
    ).agg(
        damage_reduction=("damage_ratio_max", lambda s: 1 - s.mean()),
        power_loss=("power_w", "mean"),
        power_base=("power_baseline_w", "mean"),
    ).reset_index()
    grouped["power_loss_fraction"] = (grouped.power_base - grouped.power_loss) / 22e6

    for i, band in enumerate(WIND_LABELS):
        band_data = grouped[grouped.wind_band == band]
        if band_data.empty:
            continue
        axes[0].scatter(
            band_data.power_loss_fraction,
            band_data.damage_reduction,
            s=34,
            color=cmap(i / (len(WIND_LABELS) - 1)),
            label=band,
            edgecolors="white",
            linewidths=0.4,
        )
    axes[0].set(
        xlabel="power loss (fraction of rated)",
        ylabel="damage reduction [-]",
        title="D2a  The load-relief / energy trade-off",
    )
    axes[0].legend(title="wind [m/s]", ncol=2)

    ipc_only = subset[
        (subset.action_pitch_offset_deg == 0) & (subset.action_yaw_error_deg == 0)
    ]
    pivot = ipc_only.pivot_table(
        index="wind_band", columns="action_ipc_level", values="damage_ratio_max",
        aggfunc="mean", observed=True,
    )
    for column in pivot.columns:
        axes[1].plot(range(len(pivot)), pivot[column], "o-", markersize=4, label=f"IPC={column:g}")
    axes[1].set_xticks(range(len(pivot)))
    axes[1].set_xticklabels(pivot.index)
    axes[1].axhline(1.0, color="k", linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="wind speed band [m/s]",
        ylabel="damage ratio",
        title="D2b  IPC relief at zero power cost",
    )
    axes[1].legend()

    both_zero = subset[(subset.action_pitch_offset_deg == 0) & (subset.action_ipc_level == 0)]
    pivot_yaw = both_zero.pivot_table(
        index="wind_band", columns="action_yaw_error_deg", values="damage_ratio_max",
        aggfunc="mean", observed=True,
    )
    for column in pivot_yaw.columns:
        axes[2].plot(range(len(pivot_yaw)), pivot_yaw[column], "o-", markersize=4, label=f"{column:g} deg")
    axes[2].set_xticks(range(len(pivot_yaw)))
    axes[2].set_xticklabels(pivot_yaw.index)
    axes[2].axhline(1.0, color="k", linestyle=":", linewidth=1)
    axes[2].set(
        xlabel="wind speed band [m/s]",
        ylabel="damage ratio",
        title="D2c  Yaw misalignment HURTS the tower",
    )
    axes[2].legend(ncol=2, fontsize=7.5)
    fig.suptitle("D2  Action economics: what each control channel buys and costs", y=1.04, fontsize=11)
    save(fig, "D2_action_tradeoffs.png")

    # --- D3 best action map ----------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    subset["surrogate_reward"] = (
        2.0 * (1 - subset.damage_ratio_max ** (1 / 3))
        - (subset.power_baseline_w - subset.power_w) / 22e6
    )
    best = subset.loc[
        subset.groupby(["wind_speed_id", "wave_hs_id"], observed=True).surrogate_reward.idxmax()
    ]
    for ax, (column, title, cmap_name, vmax) in zip(
        axes,
        [
            ("action_pitch_offset_deg", "D3a  Optimal pitch offset [deg]", "YlOrRd", 8),
            ("action_ipc_level", "D3b  Optimal IPC level [-]", "YlGn", 1),
            ("surrogate_reward", "D3c  Achievable reward", "coolwarm", None),
        ],
    ):
        grid = best.pivot_table(index="wave_hs_id", columns="wind_speed_id", values=column)
        kwargs = {"vmin": 0, "vmax": vmax} if vmax else {}
        if column == "surrogate_reward":
            kwargs = {"norm": TwoSlopeNorm(vcenter=0.0)}
        image = ax.imshow(
            grid.values, origin="lower", aspect="auto", cmap=cmap_name,
            extent=[0.5, 22.5, 0.5, 7.5], **kwargs,
        )
        fig.colorbar(image, ax=ax)
        ax.set(xlabel="wind speed index", ylabel="Hs index", title=title)
        ax.grid(False)
    fig.suptitle(
        "D3  The optimal action is state-dependent - this is why a policy is needed",
        y=1.04,
        fontsize=11,
    )
    save(fig, "D3_best_action_map.png")

    marginals = []
    for column, _, _ in specs:
        curve = subset.groupby(column).agg(
            mean_damage_ratio=("damage_ratio_max", "mean"),
            mean_power_w=("power_w", "mean"),
            mean_thrust_n=("thrust_n", "mean"),
        )
        curve.index.name = "action_value"
        curve["action"] = column
        marginals.append(curve.reset_index())
    table(pd.concat(marginals), "D_action_marginals.csv", index=False)

    STATS["D_action_response"] = {
        "pitch_8deg_mean_damage_ratio": float(
            subset[(subset.action_pitch_offset_deg == 8)
                   & (subset.action_yaw_error_deg == 0)
                   & (subset.action_ipc_level == 0)].damage_ratio_max.mean()
        ),
        "ipc_full_mean_damage_ratio": float(
            subset[(subset.action_pitch_offset_deg == 0)
                   & (subset.action_yaw_error_deg == 0)
                   & (subset.action_ipc_level == 1)].damage_ratio_max.mean()
        ),
        "yaw_30deg_mean_damage_ratio": float(
            subset[(subset.action_pitch_offset_deg == 0)
                   & (subset.action_yaw_error_deg == 30)
                   & (subset.action_ipc_level == 0)].damage_ratio_max.mean()
        ),
        "damage_ratio_min": float(subset.damage_ratio_max.min()),
        "damage_ratio_max": float(subset.damage_ratio_max.max()),
        "n_sweep_rows": int(len(sweep)),
        "n_actions": int(sweep.groupby(
            ["action_pitch_offset_deg", "action_yaw_error_deg", "action_ipc_level"]
        ).ngroups),
    }



# ===========================================================================
# SECTION E - RL dataset
# ===========================================================================
def section_e(transitions: pd.DataFrame) -> None:
    print("[E] RL dataset structure")
    frame = transitions.copy()
    frame["wind_band"] = pd.cut(frame.true_wind_speed, WIND_BINS, labels=WIND_LABELS)

    # --- E1 reward distribution and components ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for policy, colour in POLICY_COLOURS.items():
        values = frame.loc[frame.behaviour_policy == policy, "reward"]
        if values.empty:
            continue
        axes[0].hist(values, bins=80, histtype="step", linewidth=1.6, color=colour, label=policy, density=True)
    axes[0].axvline(0, color="k", linestyle=":", linewidth=1)
    axes[0].set(xlabel="reward", ylabel="density", title="E1a  Reward by behaviour policy")
    axes[0].legend()

    components = frame.groupby("behaviour_policy")[
        ["reward_fatigue_term", "reward_power_loss_fraction", "reward_duty_total", "reward"]
    ].mean()
    components["power_penalty"] = -1.0 * components.reward_power_loss_fraction
    components["duty_penalty"] = -0.05 * components.reward_duty_total
    order = components.reward.sort_values(ascending=False).index
    positions = np.arange(len(order))
    axes[1].bar(positions - 0.22, components.loc[order, "reward_fatigue_term"], width=0.22,
                label="fatigue term (+)", color="#1e8449")
    axes[1].bar(positions, components.loc[order, "power_penalty"], width=0.22,
                label="power penalty (-)", color="#c0392b")
    axes[1].bar(positions + 0.22, components.loc[order, "duty_penalty"], width=0.22,
                label="duty penalty (-)", color="#b9770e")
    axes[1].plot(positions, components.loc[order, "reward"], "ko-", markersize=5, label="net reward")
    axes[1].axhline(0, color="k", linewidth=0.8)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(order, rotation=20, ha="right")
    axes[1].set(ylabel="mean contribution", title="E1b  Reward decomposition")
    axes[1].legend(fontsize=7.5)

    for policy, colour in POLICY_COLOURS.items():
        subset = frame[frame.behaviour_policy == policy]
        if subset.empty:
            continue
        curve = subset.groupby("wind_band", observed=True).reward.mean()
        axes[2].plot(range(len(curve)), curve.values, "o-", markersize=4, color=colour, label=policy)
    axes[2].axhline(0, color="k", linestyle=":", linewidth=1)
    axes[2].set_xticks(range(len(WIND_LABELS)))
    axes[2].set_xticklabels(WIND_LABELS)
    axes[2].set(
        xlabel="wind speed band [m/s]",
        ylabel="mean reward",
        title="E1c  Best policy changes with wind speed",
    )
    axes[2].legend()
    fig.suptitle("E1  Reward structure of the RL dataset", y=1.04, fontsize=11)
    save(fig, "E1_reward_structure.png")

    # --- E2 action coverage -----------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    action_specs = [
        ("action_pitch_offset_deg", "pitch offset [deg]"),
        ("action_yaw_setpoint_deg", "yaw setpoint [deg]"),
        ("action_ipc_level", "IPC level [-]"),
    ]
    for ax, (column, xlabel) in zip(axes, action_specs):
        ax.hist(frame[column], bins=60, color="#2471a3", alpha=0.85)
        ax.set(xlabel=xlabel, ylabel="transitions", title=f"E2  {xlabel} coverage")
    hb = axes[3].hexbin(
        frame.action_pitch_offset_deg, frame.action_ipc_level, gridsize=28, cmap="viridis",
        mincnt=1, norm=LogNorm(),
    )
    fig.colorbar(hb, ax=axes[3], label="transitions")
    axes[3].set(xlabel="pitch offset [deg]", ylabel="IPC level [-]", title="E2d  Joint coverage")
    fig.suptitle(
        "E2  Action-space coverage from the behaviour-policy mixture (spike at zero = baseline policy)",
        y=1.05,
        fontsize=11,
    )
    save(fig, "E2_action_coverage.png")

    # --- E3 example episodes ---------------------------------------------
    rng = np.random.default_rng(3)
    picks = []
    for policy in ["baseline", "ipc_only", "feather", "random"]:
        candidates = frame[(frame.tower == "opt2") & (frame.behaviour_policy == policy)].episode_id.unique()
        if candidates.size:
            picks.append((policy, int(rng.choice(candidates))))

    fig, axes = plt.subplots(4, len(picks), figsize=(3.6 * len(picks), 8.2), sharex=True)
    for column, (policy, episode) in enumerate(picks):
        episode_data = frame[
            (frame.tower == "opt2") & (frame.behaviour_policy == policy) & (frame.episode_id == episode)
        ].sort_values("step")
        steps = episode_data.step

        axis = axes[0, column]
        axis.plot(steps, episode_data.true_wind_speed, color="#1a5276", label="V [m/s]")
        axis.plot(steps, episode_data.true_wave_hs, color="#7d3c98", label="Hs [m]")
        axis.set_title(f"{policy}\nepisode {episode}", fontsize=9)
        if column == 0:
            axis.set_ylabel("environment")
        axis.legend(fontsize=7)

        axis = axes[1, column]
        axis.plot(steps, episode_data.action_pitch_offset_deg, color="#b9770e", label="pitch off [deg]")
        axis.plot(steps, episode_data.action_ipc_level * 8, color="#1e8449", label="IPC x8")
        if column == 0:
            axis.set_ylabel("action")
        axis.legend(fontsize=7)

        axis = axes[2, column]
        axis.plot(steps, episode_data.yaw_error_deg, color="#c0392b", label="yaw error [deg]")
        axis.axhline(0, color="k", linewidth=0.6, linestyle=":")
        if column == 0:
            axis.set_ylabel("yaw")
        axis.legend(fontsize=7)

        axis = axes[3, column]
        axis.plot(steps, episode_data.del_ratio, color="#5d6d7e", label="DEL ratio")
        twin = axis.twinx()
        twin.plot(steps, episode_data.reward, color="#148f77", linewidth=1.4, label="reward")
        twin.axhline(0, color="#148f77", linewidth=0.6, linestyle=":")
        twin.grid(False)
        axis.set_xlabel("step (10 min each)")
        if column == 0:
            axis.set_ylabel("DEL ratio")
        if column == len(picks) - 1:
            twin.set_ylabel("reward", color="#148f77")
        axis.legend(fontsize=7, loc="upper left")
    fig.suptitle("E3  Example six-hour episodes: environment, action, yaw and outcome", y=1.0, fontsize=11)
    save(fig, "E3_example_episodes.png")

    # --- E4 state dependence ---------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    pivot = frame.pivot_table(
        index="behaviour_policy", columns="wind_band", values="reward", aggfunc="mean", observed=True
    )
    image = axes[0].imshow(pivot.values, cmap="RdYlGn", aspect="auto", norm=TwoSlopeNorm(vcenter=0.0))
    fig.colorbar(image, ax=axes[0], label="mean reward")
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels(pivot.columns)
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels(pivot.index)
    axes[0].set(xlabel="wind speed band [m/s]", title="E4a  Policy x wind-speed reward")
    axes[0].grid(False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            axes[0].text(j, i, f"{pivot.values[i, j]:+.2f}", ha="center", va="center", fontsize=6.5)

    quartiles = pd.qcut(frame.controllable_share_max_section, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    pivot2 = frame.pivot_table(
        index="behaviour_policy", columns=quartiles, values="reward", aggfunc="mean", observed=True
    )
    image = axes[1].imshow(pivot2.values, cmap="RdYlGn", aspect="auto", norm=TwoSlopeNorm(vcenter=0.0))
    fig.colorbar(image, ax=axes[1], label="mean reward")
    axes[1].set_xticks(range(len(pivot2.columns)))
    axes[1].set_xticklabels([f"{c}\n(low->high)" if c == "Q1" else str(c) for c in pivot2.columns])
    axes[1].set_yticks(range(len(pivot2.index)))
    axes[1].set_yticklabels(pivot2.index)
    axes[1].set(xlabel="controllable-share quartile", title="E4b  Policy x control authority")
    axes[1].grid(False)
    for i in range(pivot2.shape[0]):
        for j in range(pivot2.shape[1]):
            axes[1].text(j, i, f"{pivot2.values[i, j]:+.2f}", ha="center", va="center", fontsize=6.5)

    for policy, colour in POLICY_COLOURS.items():
        subset = frame[frame.behaviour_policy == policy]
        if subset.empty:
            continue
        axes[2].hist(subset.del_ratio, bins=70, histtype="step", linewidth=1.5, color=colour,
                     label=policy, density=True)
    axes[2].axvline(1.0, color="k", linestyle=":", linewidth=1)
    axes[2].set(
        xlabel="DEL ratio vs aligned baseline",
        ylabel="density",
        title="E4c  Achieved load relief (<1 is better)",
    )
    axes[2].legend()
    fig.suptitle("E4  State-dependence: no single action dominates the envelope", y=1.04, fontsize=11)
    save(fig, "E4_state_dependence.png")

    policy_summary = frame.groupby("behaviour_policy").agg(
        n=("reward", "size"),
        mean_reward=("reward", "mean"),
        std_reward=("reward", "std"),
        fraction_positive=("reward", lambda s: (s > 0).mean()),
        mean_del_ratio=("del_ratio", "mean"),
        mean_fatigue_relief=("reward_fatigue_relief", "mean"),
        mean_power_loss=("reward_power_loss_fraction", "mean"),
        mean_duty=("reward_duty_total", "mean"),
    )
    table(policy_summary, "E_reward_by_policy.csv")
    table(
        frame.pivot_table(index="behaviour_policy", columns="wind_band", values="reward",
                          aggfunc="mean", observed=True),
        "E_reward_by_policy_and_wind.csv",
    )

    STATS["E_rl_dataset"] = {
        "n_transitions": int(len(frame)),
        "n_episodes": int(frame.groupby(["tower", "episode_id"]).ngroups),
        "steps_per_episode": int(frame.step.max() + 1),
        "simulated_hours": float(
            frame.groupby(["tower", "episode_id"]).ngroups * (frame.step.max() + 1) * 600 / 3600
        ),
        "reward_by_policy": policy_summary.to_dict(orient="index"),
        "reward_overall": {
            "mean": float(frame.reward.mean()),
            "std": float(frame.reward.std()),
            "min": float(frame.reward.min()),
            "max": float(frame.reward.max()),
            "fraction_positive": float((frame.reward > 0).mean()),
        },
        "del_ratio": {
            "min": float(frame.del_ratio.min()),
            "p05": float(frame.del_ratio.quantile(0.05)),
            "median": float(frame.del_ratio.median()),
            "max": float(frame.del_ratio.max()),
        },
        "best_policy_by_wind_band": pivot.idxmax(axis=0).to_dict(),
        "yaw_error_abs_mean_baseline_policy": float(
            frame[frame.behaviour_policy == "baseline"].yaw_error_deg.abs().mean()
        ),
    }


# ===========================================================================
# SECTION F - IoT layer and data quality
# ===========================================================================
def section_f(transitions: pd.DataFrame) -> None:
    print("[F] IoT layer and data quality")
    frame = transitions
    channels = [
        ("wind_speed", "wind speed [m/s]"),
        ("wave_hs", "Hs [m]"),
        ("turbulence_std", "wind speed std [m/s]"),
        ("tower_damage_rate", "damage rate [-]"),
    ]

    # --- F1 measurement error --------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4))
    sample = frame.sample(12_000, random_state=0)
    for column, (channel, label) in enumerate(channels):
        truth = sample[f"true_{channel}"]
        measured = sample[f"meas_{channel}"]

        axis = axes[0, column]
        axis.scatter(truth, measured, s=2, alpha=0.12, color="#2471a3", edgecolors="none")
        limits = [min(truth.min(), measured.min()), max(truth.max(), measured.max())]
        axis.plot(limits, limits, "r--", linewidth=1)
        if channel == "tower_damage_rate":
            axis.set_xscale("log")
            axis.set_yscale("log")
        axis.set(xlabel=f"true {label}", ylabel=f"measured {label}", title=f"F1  {channel}")

        axis = axes[1, column]
        denominator = np.maximum(np.abs(frame[f"true_{channel}"]), 1e-12)
        error = (frame[f"meas_{channel}"] - frame[f"true_{channel}"]) / denominator * 100
        axis.hist(np.clip(error, -60, 60), bins=90, color="#b9770e", alpha=0.85)
        axis.axvline(0, color="k", linestyle=":", linewidth=1)
        axis.set(
            xlabel="relative error [%]",
            ylabel="transitions",
            title=f"median |err| = {np.median(np.abs(error)):.1f}%",
        )
    fig.suptitle(
        "F1  IoT sensor layer: measured vs ground truth, and realised relative error",
        y=1.01,
        fontsize=11,
    )
    save(fig, "F1_iot_measurement_error.png")

    # --- F2 dropout and health -------------------------------------------
    all_channels = [
        "wind_speed", "turbulence_std", "wind_direction", "wave_hs",
        "wave_tp", "tower_damage_rate", "thrust", "power",
    ]
    rows = []
    for channel in all_channels:
        truth = frame[f"true_{channel}"]
        measured = frame[f"meas_{channel}"]
        denominator = np.maximum(np.abs(truth), 1e-12)
        rows.append(
            {
                "channel": channel,
                "dropout_rate_pct": (1 - frame[f"valid_{channel}"].mean()) * 100,
                "median_abs_rel_error_pct": float(np.median(np.abs(measured - truth) / denominator) * 100),
                "bias": float((measured - truth).mean()),
                "rmse": float(np.sqrt(((measured - truth) ** 2).mean())),
            }
        )
    error_frame = pd.DataFrame(rows).set_index("channel")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7))
    axes[0].barh(error_frame.index, error_frame.dropout_rate_pct, color="#c0392b", alpha=0.85)
    axes[0].set(xlabel="packet-loss rate [%]", title="F2a  Dropout per channel")
    axes[1].barh(error_frame.index, error_frame.median_abs_rel_error_pct, color="#1a5276", alpha=0.85)
    axes[1].set(xlabel="median |relative error| [%]", title="F2b  Measurement error per channel")
    health = frame.sensor_health.value_counts().sort_index()
    axes[2].bar(health.index.astype(float), health.values, width=0.1, color="#1e8449", alpha=0.9)
    axes[2].set_yscale("log")
    axes[2].set(
        xlabel="sensor_health (fraction of channels fresh)",
        ylabel="transitions (log)",
        title=f"F2c  {(frame.sensor_health < 1).mean()*100:.1f}% of steps degraded",
    )
    fig.suptitle("F2  Sensor-network reliability as delivered to the policy", y=1.04, fontsize=11)
    save(fig, "F2_iot_reliability.png")

    # --- F3 observation correlation + integrity --------------------------
    observation_columns = (
        [f"meas_{channel}" for channel in all_channels]
        + [
            "prev_pitch_offset_deg", "prev_yaw_setpoint_deg", "prev_ipc_level",
            "nacelle_yaw_deg", "yaw_error_deg", "cumulative_damage_fraction", "step_fraction",
        ]
    )
    correlation = frame[observation_columns].corr()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    image = axes[0].imshow(correlation.values, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(image, ax=axes[0], label="Pearson r")
    labels = [c.replace("meas_", "").replace("_deg", "").replace("prev_", "prev ") for c in observation_columns]
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_title("F3a  Observation correlation structure")
    axes[0].grid(False)

    numeric = frame.select_dtypes(include=[np.number])
    integrity = pd.Series(
        {
            "rows": len(frame),
            "numeric columns": numeric.shape[1],
            "NaN values": int(numeric.isna().to_numpy().sum()),
            "infinite values": int(np.isinf(numeric.to_numpy()).sum()),
            "duplicate rows": int(frame.duplicated().sum()),
            "episodes": frame.groupby(["tower", "episode_id"]).ngroups,
            "episodes with 1 done": int((frame.groupby(["tower", "episode_id"]).done.sum() == 1).sum()),
        }
    )
    axes[1].axis("off")
    text = "\n".join(f"{key:<22} {value:>12,}" for key, value in integrity.items())
    axes[1].text(
        0.02, 0.95, "Data integrity\n" + "-" * 36 + "\n" + text,
        family="monospace", fontsize=9.5, va="top", transform=axes[1].transAxes,
    )
    coverage = (
        f"\n\nCoverage\n" + "-" * 36 + "\n"
        f"{'unique FLOATBench sims used':<30} {frame.sim_id.nunique():>6,}\n"
        f"{'of available per tower':<30} {6468:>6,}\n"
        f"{'wind ids visited':<30} {frame.wind_speed_id.nunique():>6}/22\n"
        f"{'Hs ids visited':<30} {frame.wave_hs_id.nunique():>6}/7\n"
        f"{'Tp ids visited':<30} {frame.wave_tp_id.nunique():>6}/7\n"
        f"{'turbulence seeds visited':<30} {frame.wind_seed_id.nunique():>6}/6"
    )
    axes[1].text(0.02, 0.42, coverage, family="monospace", fontsize=9.5, va="top",
                 transform=axes[1].transAxes)
    fig.suptitle("F3  Observation structure and dataset integrity", y=1.0, fontsize=11)
    save(fig, "F3_integrity_and_correlation.png")

    table(error_frame, "F_iot_error_report.csv")
    table(frame[observation_columns].describe().T, "F_observation_stats.csv")

    STATS["F_iot_and_quality"] = {
        "sensor_error": error_frame.to_dict(orient="index"),
        "degraded_step_fraction": float((frame.sensor_health < 1).mean()),
        "integrity": {key: int(value) for key, value in integrity.items()},
        "coverage": {
            "unique_sims_used": int(frame.sim_id.nunique()),
            "available_per_tower": 6468,
            "wind_ids": int(frame.wind_speed_id.nunique()),
            "hs_ids": int(frame.wave_hs_id.nunique()),
            "tp_ids": int(frame.wave_tp_id.nunique()),
            "seeds": int(frame.wind_seed_id.nunique()),
        },
    }


# ===========================================================================
# SECTION G - diagnostics: issues found by this analysis
# ===========================================================================
def section_g(transitions: pd.DataFrame) -> None:
    """Quantify the below-rated monotonicity-guard artefact.

    Below rated, the reference pitch schedule sits 1-2 deg below the Cp optimum
    of the performance surface at the same TSR. A positive pitch offset
    therefore moves *towards* the surface optimum, and the raw Cp ratio exceeds
    1. The monotonicity guard clamps that to 1, which is correct in that it
    prevents free extra power - but it also makes small feathering offsets look
    entirely free below rated, when in reality feathering always costs power.
    """
    print("[G] diagnostics")
    aero = load_aero(write_report=False)
    frame = transitions

    winds = np.linspace(3, 25, 221)
    tsr = aero.schedule.tip_speed_ratio(winds)
    pitch_base = aero.schedule.pitch_deg(winds)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))

    cmap = plt.get_cmap("plasma")
    offsets = [1, 2, 4, 6, 8]
    for i, offset in enumerate(offsets):
        raw = aero.surface.power_coefficient(
            pitch_base + offset, tsr
        ) / aero.surface.power_coefficient(pitch_base, tsr)
        axes[0].plot(winds, raw, color=cmap(i / (len(offsets) - 1)), linewidth=1.7,
                     label=f"+{offset} deg")
    axes[0].axhline(1.0, color="k", linestyle="--", linewidth=1.2)
    axes[0].fill_between(winds, 1.0, 1.06, where=winds <= 9.6, color="#c0392b", alpha=0.12)
    axes[0].text(4.2, 1.021, "guard clamps\nthis region", fontsize=7.5, color="#c0392b")
    axes[0].set(
        xlabel="wind speed [m/s]",
        ylabel="raw Cp ratio from surface",
        title="G1a  Raw Cp ratio exceeds 1 below rated",
        ylim=(0, 1.15),
    )
    axes[0].legend(fontsize=7.5, ncol=2)

    axes[1].plot(winds, pitch_base, color="#1a5276", linewidth=1.9, label="reference schedule pitch")
    optimum = np.array(
        [
            aero.surface.pitch_deg[
                int(np.argmax(aero.surface.power_coefficient(aero.surface.pitch_deg, np.full(20, t))))
            ]
            for t in tsr
        ]
    )
    axes[1].plot(winds, optimum, color="#c0392b", linewidth=1.9, linestyle="--",
                 label="surface Cp-optimal pitch")
    axes[1].set(
        xlabel="wind speed [m/s]",
        ylabel="blade pitch [deg]",
        title="G1b  Root cause: the two artefacts disagree below rated",
        ylim=(-6, 25),
    )
    axes[1].legend(fontsize=8)

    raw_action = aero.surface.power_coefficient(
        aero.schedule.pitch_deg(frame.true_wind_speed.values) + frame.action_pitch_offset_deg.values,
        aero.schedule.tip_speed_ratio(frame.true_wind_speed.values),
    ) / aero.surface.power_coefficient(
        aero.schedule.pitch_deg(frame.true_wind_speed.values),
        aero.schedule.tip_speed_ratio(frame.true_wind_speed.values),
    )
    affected = (raw_action > 1.0) & (frame.action_pitch_offset_deg.values > 0)
    bands = pd.cut(frame.true_wind_speed, WIND_BINS, labels=WIND_LABELS)
    share = pd.Series(affected).groupby(bands, observed=True).mean() * 100
    axes[2].bar(range(len(share)), share.values, color="#c0392b", alpha=0.85)
    axes[2].set_xticks(range(len(share)))
    axes[2].set_xticklabels(share.index)
    axes[2].set(
        xlabel="wind speed band [m/s]",
        ylabel="% of transitions affected",
        title=f"G1c  {affected.mean()*100:.1f}% of dataset affected overall",
    )
    for i, value in enumerate(share.values):
        axes[2].text(i, value + 0.6, f"{value:.0f}%", ha="center", fontsize=8)
    fig.suptitle(
        "G1  DIAGNOSTIC: below-rated feathering appears free - an artefact, not physics",
        y=1.04,
        fontsize=11,
        color="#8b0000",
    )
    save(fig, "G1_diagnostic_guard_artefact.png")

    feather = frame[frame.behaviour_policy == "feather"]
    feather_band = feather.true_wind_speed.between(6, 12)
    feather_raw = aero.surface.power_coefficient(
        aero.schedule.pitch_deg(feather.true_wind_speed.values)
        + feather.action_pitch_offset_deg.values,
        aero.schedule.tip_speed_ratio(feather.true_wind_speed.values),
    ) / aero.surface.power_coefficient(
        aero.schedule.pitch_deg(feather.true_wind_speed.values),
        aero.schedule.tip_speed_ratio(feather.true_wind_speed.values),
    )
    feather_affected = (feather_raw > 1.0) & (feather.action_pitch_offset_deg.values > 0)

    diagnostic = {
        "affected_transitions": int(affected.sum()),
        "affected_fraction": float(affected.mean()),
        "guard_binds_below_wind_speed_ms": 9.6,
        "max_raw_cp_ratio": float(
            max(
                (
                    aero.surface.power_coefficient(pitch_base + offset, tsr)
                    / aero.surface.power_coefficient(pitch_base, tsr)
                ).max()
                for offset in offsets
            )
        ),
        "mean_reward_affected": float(frame.reward.values[affected].mean()),
        "mean_reward_unaffected": float(frame.reward.values[~affected].mean()),
        "mean_power_loss_affected": float(frame.reward_power_loss_fraction.values[affected].mean()),
        "mean_fatigue_relief_affected": float(frame.reward_fatigue_relief.values[affected].mean()),
        "feather_policy_6_12ms_affected_fraction": float(
            (feather_affected & feather_band.values).sum() / max(feather_band.sum(), 1)
        ),
        "feather_policy_6_12ms_mean_reward": float(feather.reward[feather_band].mean()),
        "affected_share_by_wind_band_pct": share.to_dict(),
    }
    table(pd.DataFrame([diagnostic]).T.rename(columns={0: "value"}), "G_diagnostic_guard_artefact.csv")
    STATS["G_diagnostics"] = diagnostic


# ===========================================================================
def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    print("loading data ...")
    towers, transitions, sweep = load_all_data()
    print(f"  transitions {transitions.shape}  sweep {sweep.shape}")

    section_a(towers)
    section_b()
    section_c(towers)
    section_d(sweep)
    section_e(transitions)
    section_f(transitions)
    section_g(transitions)

    path = EDA / "summary_stats.json"
    path.write_text(json.dumps(STATS, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    print(f"\nsummary -> {path.relative_to(ROOT)}")
    print(f"figures: {len(list(FIG.glob('*.png')))}  tables: {len(list(TAB.glob('*.csv')))}")


if __name__ == "__main__":
    main()
