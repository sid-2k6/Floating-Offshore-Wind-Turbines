#!/usr/bin/env python3
"""
FOWT-ARISE pipeline fixtures -- SYNTHETIC, PLANNING/VALIDATION ONLY.

PURPOSE
    Exercise the output-tree layout, the aggregation code and the figure design without a
    30-minute training run. That is the legitimate use of synthetic data here: a test fixture.

WHAT THIS DELIBERATELY DOES NOT DO
    * It does not order the placeholder values so the proposed model wins. Values are drawn from one
      neutral band shared by all five configurations, and the resulting order is printed -- whatever
      it happens to be. Ordering fixtures to favour a hypothesis is how a fixture becomes a claim.
    * It does not simulate training dynamics. `history.csv` carries the correct COLUMNS (so plotting
      and schema code is exercised) but its values are a clean monotone ramp with no noise, no
      plateau and no early stopping. A glance at the curve tells you it is a placeholder.
    * It does not fabricate checkpoint validation metrics or SHAP attributions. Those are claims
      about a specific trained model; there is no trained model here.

    None of the stated goals -- pipeline planning, visualisation design, output-structure
    validation, ablation-design validation -- require any of the above.

USAGE
    python3 make_pipeline_fixtures.py --out ./fixtures [--real-output-root /path/to/real/outputs]

Every CSV carries  data = "synthetic".
Every figure is accompanied by a caption BELOW the image in FIGURE_INDEX.md.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
FONT_SIZE = 20
DPI = 300
CAPTION = "Synthetically generated image — planning/reference only"
SYNTH_STATUS = "SYNTHETIC — PLANNING ONLY"

MODELS = ["FOWT_ARISE", "ABLATION_N1", "ABLATION_N2", "ABLATION_N3", "ABLATION_N4"]
N_EPOCHS = 48
N_EPISODES = 120
N_STEPS = 36

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
})

# Real baseline reference values, reproduced verbatim from the project brief. These are the ONLY
# non-synthetic numbers in this script and they are never modified, never recomputed, and never
# placed in the same table as a placeholder value.
REAL_BASELINES = {
    "RB-FOWT": {"Mean Matched Sweep Objective": 0.0507, "Mean DEL Ratio": 0.9612,
                "Median DEL Ratio": 0.9909, "Fatigue Relief %": 3.8845, "Power Loss %": 5.8512,
                "Actuator Duty Proxy": 0.9367, "Mean Action Magnitude": 0.9367,
                "Mean |Yaw Action|": 0.0000, "Action-Sweep Coverage %": 100.0,
                "Clean Performance": 0.0506, "IoT-Degraded Performance": 0.0519,
                "IoT Performance Gap": -0.0013},
    "CQL": {"Mean Matched Sweep Objective": -0.0397, "Mean DEL Ratio": 0.9688,
            "Median DEL Ratio": 0.9908, "Fatigue Relief %": 3.1165, "Power Loss %": 11.6170,
            "Actuator Duty Proxy": 1.3410, "Mean Action Magnitude": 1.3410,
            "Mean |Yaw Action|": 1.4497, "Action-Sweep Coverage %": 100.0,
            "Clean Performance": -0.0397, "IoT-Degraded Performance": -0.0394,
            "IoT Performance Gap": -0.0003},
    "IQL": {"Mean Matched Sweep Objective": -0.0978, "Mean DEL Ratio": 0.9987,
            "Median DEL Ratio": 1.0000, "Fatigue Relief %": 0.1268, "Power Loss %": 10.0461,
            "Actuator Duty Proxy": 0.8490, "Mean Action Magnitude": 0.8490,
            "Mean |Yaw Action|": 1.7251, "Action-Sweep Coverage %": 100.0,
            "Clean Performance": -0.0982, "IoT-Degraded Performance": -0.0924,
            "IoT Performance Gap": -0.0058},
}
ACTION_SWEEP_REFERENCE = {
    "Mean Matched Sweep Objective": 0.0946, "Mean DEL Ratio": 0.9493,
    "Fatigue Relief %": 5.0748, "Power Loss %": 1.6984,
}

COMPARISON_COLUMNS = [
    "Method", "Mean Matched Sweep Objective", "Mean DEL Ratio", "Median DEL Ratio",
    "Fatigue Relief %", "Power Loss %", "Actuator Duty Proxy", "Mean Action Magnitude",
    "Mean |Yaw Action|", "No-Action Rate", "Action-Sweep Coverage %", "Clean Performance",
    "IoT-Degraded Performance", "IoT Performance Gap", "Robustness Drop %",
    "Parameter Count", "Best Validation Epoch",
]

MANIFEST: list[dict] = []


# --------------------------------------------------------------------------------------------
# safety
# --------------------------------------------------------------------------------------------
def assert_separated(synthetic_root: Path, real_root: Path | None) -> None:
    """Refuse to run if the fixture tree could touch a real experiment tree."""
    s = synthetic_root.resolve()
    if real_root is None:
        print(f"[SAFETY] no --real-output-root given; nothing to collide with.")
        print(f"[SAFETY] fixtures -> {s}")
        return
    r = real_root.resolve()
    if s == r:
        sys.exit(f"[SAFETY] ABORT: fixture root == real output root ({s}).")
    if r in s.parents:
        sys.exit(f"[SAFETY] ABORT: fixture root {s} is INSIDE the real output root {r}.")
    if s in r.parents:
        sys.exit(f"[SAFETY] ABORT: real output root {r} is INSIDE the fixture root {s}.")
    print(f"[SAFETY] OK -- disjoint.\n[SAFETY]   real      : {r}\n[SAFETY]   synthetic : {s}")


def save_csv(df: pd.DataFrame, path: Path, model: str, description: str) -> None:
    """Write a CSV with the mandatory synthetic marker, and record it in the manifest."""
    out = df.copy()
    out["data"] = "synthetic"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    MANIFEST.append({"artifact": path.name, "model": model, "path": str(path),
                     "artifact_type": "csv", "synthetic_status": SYNTH_STATUS,
                     "description": description})


def save_json(payload: dict, path: Path, model: str, description: str) -> None:
    payload = {"data": "synthetic", "synthetic_status": SYNTH_STATUS, **payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    MANIFEST.append({"artifact": path.name, "model": model, "path": str(path),
                     "artifact_type": "json", "synthetic_status": SYNTH_STATUS,
                     "description": description})


FIGURES: list[tuple[Path, str]] = []


def save_fig(fig, path: Path, model: str, description: str) -> None:
    """Save at 300 dpi and register the figure so its caption goes BELOW it in FIGURE_INDEX.md."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append((path, description))
    MANIFEST.append({"artifact": path.name, "model": model, "path": str(path),
                     "artifact_type": "png", "synthetic_status": SYNTH_STATUS,
                     "description": description})


# --------------------------------------------------------------------------------------------
# placeholder record generation -- NEUTRAL, not tuned to any ordering
# --------------------------------------------------------------------------------------------
def placeholder_episode_records(model: str, rng: np.random.Generator) -> pd.DataFrame:
    """Per-episode placeholder records. All five models share ONE band; no model is favoured.

    Aggregates are computed FROM these records rather than written independently, which is the part
    of the brief worth keeping: it makes the fixture tables internally consistent, so aggregation
    bugs surface here instead of on real data.
    """
    n = N_EPISODES
    # One shared band for every configuration. The per-model offset is a fixed, tiny, arbitrary
    # jitter -- explicitly not a performance story. Keyed on a STABLE hash: Python's built-in
    # hash() is randomised per process (PYTHONHASHSEED), which silently broke reproducibility the
    # first time this was written, so crc32 is used instead.
    name_jitter = (zlib.crc32(model.encode()) % 7 - 3) * 0.0008

    del_ratio = np.clip(rng.normal(0.968, 0.010, n), 0.90, 1.0)
    fatigue_relief = 1.0 - del_ratio
    power_loss = np.clip(rng.normal(0.012, 0.005, n), 0.0, 1.0)
    duty = np.clip(rng.normal(0.13, 0.03, n), 0.0, 1.0)
    smooth = np.clip(rng.normal(0.06, 0.02, n), 0.0, 1.0)

    # placeholder objective, assembled from the placeholder components with the brief's planning
    # weights, so the table is self-consistent
    lam = dict(fat=1.0, pow=0.5, act=0.25, smooth=0.1)
    objective = (lam["fat"] * fatigue_relief * 2.0
                 - lam["pow"] * power_loss
                 - lam["act"] * duty * 0.05
                 - lam["smooth"] * smooth * 0.02) + name_jitter

    return pd.DataFrame({
        "episode_id": np.arange(n),
        "tower": rng.choice(["ref", "opt1", "opt2"], n),
        "n_steps": N_STEPS,
        "matched_sweep_objective": objective,
        "del_ratio": del_ratio,
        "fatigue_relief": fatigue_relief,
        "power_loss_fraction": power_loss,
        "actuator_duty": duty,
        "action_smoothness": smooth,
        "notice": "placeholder value, not a measurement",
    })


def placeholder_action_records(model: str, rng: np.random.Generator) -> pd.DataFrame:
    """Per-step placeholder action records, including a controllability-linked no-action decision.

    The controllable-share -> intervention link is kept because it is a STRUCTURAL relationship the
    plotting and aggregation code must handle (a no-action rate has to come from actual decisions,
    not a written-in constant). Its magnitude here is arbitrary.
    """
    n = N_EPISODES * N_STEPS
    controllable_share = np.clip(rng.beta(2.0, 3.0, n), 0.0, 1.0)
    # low controllable share -> more likely to hold off; arbitrary logistic, not a fitted model
    p_act = 1.0 / (1.0 + np.exp(-(controllable_share - 0.30) * 8.0))
    acts = rng.random(n) < p_act

    pitch = np.where(acts, np.clip(rng.gamma(1.6, 0.9, n), 0.0, 8.0), 0.0)
    ipc = np.where(acts, np.clip(rng.beta(2.0, 2.0, n), 0.0, 1.0), 0.0)
    yaw = np.where(acts, np.clip(rng.normal(0.0, 1.2, n), -30.0, 30.0), 0.0)

    return pd.DataFrame({
        "episode_id": np.repeat(np.arange(N_EPISODES), N_STEPS),
        "step": np.tile(np.arange(N_STEPS), N_EPISODES),
        "controllable_share": controllable_share,
        "action_pitch_offset_deg": pitch,
        "action_yaw_error_deg": yaw,
        "action_ipc_level": ipc,
        "intervened": acts,
        "no_action": ~acts,
        "notice": "placeholder value, not a measurement",
    })


def placeholder_robustness(model: str, clean: float, rng: np.random.Generator) -> pd.DataFrame:
    """Per-degradation-mode placeholder rows. Degradation magnitudes are arbitrary and shared."""
    rows = []
    for mode, scale in (("clean", 0.0), ("noisy", 1.0), ("dropout", 0.7),
                        ("biased", 0.8), ("stale", 0.4), ("combined", 1.6)):
        drop = abs(rng.normal(0.0006, 0.0002)) * scale
        deg = clean - drop
        rows.append({
            "mode": mode, "clean_performance": clean, "degraded_performance": deg,
            "iot_performance_gap": clean - deg,
            "robustness_drop_pct": (100.0 * (clean - deg) / abs(clean)) if abs(clean) > 1e-12 else np.nan,
            "mean_action_shift": abs(rng.normal(0.02, 0.004)) * max(scale, 0.05),
            "notice": "placeholder value, not a measurement",
        })
    return pd.DataFrame(rows)


def placeholder_history(model: str) -> pd.DataFrame:
    """Correct COLUMNS so plotting/schema code is exercised; values are a clean monotone ramp.

    Deliberately NOT a simulated training curve: no noise, no plateau, no early stopping, no LR
    schedule events. The point of a fixture is to exercise the code path, not to look like a run.
    Plot it and you see straight lines -- which is the intended, unmistakable signal.
    """
    e = np.arange(1, N_EPOCHS + 1)
    t = (e - 1) / max(N_EPOCHS - 1, 1)          # 0 -> 1 linear ramp
    ramp = lambda a, b: a + (b - a) * t          # noqa: E731
    return pd.DataFrame({
        "epoch": e,
        "train_loss": ramp(1.0, 0.2),
        "validation_loss": ramp(1.0, 0.3),
        "mean_reward": ramp(0.0, 0.03),
        "validation_reward": ramp(0.0, 0.03),
        "mean_del_ratio": ramp(1.0, 0.968),
        "median_del_ratio": ramp(1.0, 0.995),
        "fatigue_relief_percent": ramp(0.0, 3.2),
        "power_loss_percent": ramp(0.0, 1.2),
        "actuator_duty": ramp(0.0, 0.13),
        "mean_action_magnitude": ramp(0.0, 0.34),
        "mean_yaw_action": ramp(0.0, 0.09),
        "no_action_rate": ramp(1.0, 0.44),
        "clean_performance": ramp(0.0, 0.03),
        "iot_degraded_performance": ramp(0.0, 0.029),
        "iot_performance_gap": ramp(0.0, 0.001),
        "robustness_loss": ramp(0.5, 0.05),
        "learning_rate": np.full(N_EPOCHS, 3e-4),
        "notice": "LINEAR RAMP PLACEHOLDER -- not a training curve, no early stopping, no LR schedule",
    })


# --------------------------------------------------------------------------------------------
def build(out_root: Path) -> pd.DataFrame:
    random.seed(SEED)
    np.random.seed(SEED)

    per_model_metrics = {}
    for i, model in enumerate(MODELS):
        rng = np.random.default_rng(SEED + i)
        d = out_root / model
        d.mkdir(parents=True, exist_ok=True)

        ep = placeholder_episode_records(model, rng)
        ac = placeholder_action_records(model, rng)

        clean = float(ep["matched_sweep_objective"].mean())
        rb = placeholder_robustness(model, clean, rng)
        deg = float(rb.set_index("mode").loc["combined", "degraded_performance"])

        # aggregates computed FROM the records above -- never written independently
        metrics = {
            "Method": model,
            "Mean Matched Sweep Objective": clean,
            "Mean DEL Ratio": float(ep["del_ratio"].mean()),
            "Median DEL Ratio": float(ep["del_ratio"].median()),
            "Fatigue Relief %": float(100.0 * ep["fatigue_relief"].mean()),
            "Power Loss %": float(100.0 * ep["power_loss_fraction"].mean()),
            "Actuator Duty Proxy": float(ep["actuator_duty"].mean()),
            "Mean Action Magnitude": float(np.mean(
                (np.abs(ac["action_pitch_offset_deg"]) / 8.0
                 + np.abs(ac["action_ipc_level"])
                 + np.abs(ac["action_yaw_error_deg"]) / 30.0) / 3.0)),
            "Mean |Yaw Action|": float(np.abs(ac["action_yaw_error_deg"]).mean()),
            "No-Action Rate": float(ac["no_action"].mean()),
            "Action-Sweep Coverage %": 100.0,
            "Clean Performance": clean,
            "IoT-Degraded Performance": deg,
            "IoT Performance Gap": clean - deg,
            "Robustness Drop %": float(100.0 * (clean - deg) / abs(clean)) if abs(clean) > 1e-12 else np.nan,
            "Parameter Count": int(42697 if model != "ABLATION_N1" else 29289),
            "Best Validation Epoch": int(N_EPOCHS),   # placeholder: no model selection happened
        }
        per_model_metrics[model] = metrics

        save_csv(ep, d / "episode_metrics.csv", model, "per-episode placeholder records")
        save_csv(ac, d / "action_statistics.csv", model, "per-step placeholder action records")
        save_csv(rb, d / "robustness_metrics.csv", model, "per-mode placeholder robustness rows")
        save_csv(pd.DataFrame([metrics]), d / "metrics.csv", model, "aggregates, computed from records")
        save_csv(placeholder_history(model), d / "history.csv", model,
                 "column-schema fixture; LINEAR RAMP, not a training curve")
        save_csv(ep.head(500).assign(split="test"), d / "test_predictions.csv", model,
                 "placeholder prediction rows")
        save_csv(ep.head(200).assign(split="validation"), d / "validation_metrics.csv", model,
                 "placeholder validation rows")
        save_json(metrics, d / "metrics.json", model, "aggregates, computed from records")
        save_json({
            "model_name": model,
            "notice": "No model was trained. This file exists so the output-tree layout and any "
                      "code that reads a training summary can be validated. It carries no "
                      "validation metric, no best epoch and no early-stopping decision, because "
                      "none occurred.",
            "epochs_in_history_fixture": N_EPOCHS,
            "history_values": "linear ramp placeholder",
        }, d / "training_summary.json", model, "structural stub; no fabricated run statistics")

    df = pd.DataFrame([per_model_metrics[m] for m in MODELS], columns=COMPARISON_COLUMNS)

    # ---- comparison tables: placeholders and real baselines kept in SEPARATE files ----
    cmp_dir = out_root / "comparison"
    save_csv(df, cmp_dir / "ablation_comparison_SYNTHETIC.csv", "all",
             "placeholder ablation comparison -- values NOT ordered to favour any model")
    try:
        with pd.ExcelWriter(cmp_dir / "ablation_comparison_SYNTHETIC.xlsx") as w:
            df.assign(data="synthetic").to_excel(w, index=False, sheet_name="synthetic_fixture")
        MANIFEST.append({"artifact": "ablation_comparison_SYNTHETIC.xlsx", "model": "all",
                         "path": str(cmp_dir / "ablation_comparison_SYNTHETIC.xlsx"),
                         "artifact_type": "xlsx", "synthetic_status": SYNTH_STATUS,
                         "description": "placeholder ablation comparison"})
    except Exception as e:  # openpyxl absent
        print(f"  (xlsx skipped: {e})")

    ref = pd.DataFrame([{"Method": k, **v} for k, v in REAL_BASELINES.items()])
    ref["provenance"] = "REAL BASELINE REFERENCE — reproduced verbatim from the project brief"
    ref.to_csv(cmp_dir / "REAL_baseline_reference_values.csv", index=False)
    MANIFEST.append({"artifact": "REAL_baseline_reference_values.csv", "model": "baselines",
                     "path": str(cmp_dir / "REAL_baseline_reference_values.csv"),
                     "artifact_type": "csv",
                     "synthetic_status": "REAL REFERENCE — not synthetic, not modified",
                     "description": "real baseline values, kept in their own file"})

    swp = pd.DataFrame([{"Method": "Action-Sweep Reference", **ACTION_SWEEP_REFERENCE}])
    swp["provenance"] = ("REAL REFERENCE — counterfactual upper-bound-like reference, NOT a "
                         "deployable policy")
    swp.to_csv(cmp_dir / "REAL_action_sweep_reference.csv", index=False)
    MANIFEST.append({"artifact": "REAL_action_sweep_reference.csv", "model": "reference",
                     "path": str(cmp_dir / "REAL_action_sweep_reference.csv"),
                     "artifact_type": "csv",
                     "synthetic_status": "REAL REFERENCE — not synthetic, not modified",
                     "description": "action-sweep reference, kept in its own file"})
    return df


def make_figures(df: pd.DataFrame, out_root: Path) -> None:
    """One figure per chart, fontsize 20, dpi 300. Captions go BELOW, in FIGURE_INDEX.md."""
    pdir = out_root / "plots"
    colors = ["#1e8449", "#b9770e", "#2471a3", "#943126", "#6c3483"]

    bars = [
        ("Mean Matched Sweep Objective", "objective (placeholder)", "objective_comparison.png"),
        ("Mean DEL Ratio", "DEL ratio — LOWER is better", "del_ratio_comparison.png"),
        ("Fatigue Relief %", "fatigue relief % — HIGHER is better", "fatigue_relief_comparison.png"),
        ("Power Loss %", "power loss % — LOWER is better", "power_loss_comparison.png"),
        ("Actuator Duty Proxy", "duty proxy — LOWER is better", "actuator_duty_comparison.png"),
        ("Mean Action Magnitude", "mean |a| (normalised)", "action_magnitude_comparison.png"),
        ("Mean |Yaw Action|", "mean |yaw| [deg]", "yaw_action_comparison.png"),
        ("No-Action Rate", "fraction of steps with no action", "no_action_rate_comparison.png"),
        ("IoT Performance Gap", "clean − degraded — LOWER is better", "iot_gap_comparison.png"),
        ("Robustness Drop %", "robustness drop % — LOWER is better", "robustness_drop_comparison.png"),
    ]
    for col, ylabel, fname in bars:
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.bar(df["Method"], df[col], color=colors, edgecolor="black")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{col} (placeholder fixture)")
        ax.tick_params(axis="x", rotation=20)
        save_fig(fig, pdir / fname, "all", f"{col} across configurations — placeholder values")

    # clean vs degraded, grouped
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["Clean Performance"], 0.4, label="clean", color="#1e8449", edgecolor="black")
    ax.bar(x + 0.2, df["IoT-Degraded Performance"], 0.4, label="IoT-degraded", color="#b9770e",
           edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(df["Method"], rotation=20)
    ax.set_ylabel("objective (placeholder)"); ax.legend()
    ax.set_title("Clean vs IoT-degraded (placeholder fixture)")
    save_fig(fig, pdir / "clean_vs_degraded.png", "all", "clean vs degraded — placeholder values")

    # fatigue/power trade-off scatter
    fig, ax = plt.subplots(figsize=(12, 9))
    for (_, r), c in zip(df.iterrows(), colors):
        ax.scatter(r["Power Loss %"], r["Fatigue Relief %"], s=320, color=c, edgecolor="black",
                   label=r["Method"])
    ax.set_xlabel("power loss % — LOWER is better")
    ax.set_ylabel("fatigue relief % — HIGHER is better")
    ax.set_title("Fatigue / power trade-off (placeholder fixture)")
    ax.legend(fontsize=FONT_SIZE * 0.7)
    save_fig(fig, pdir / "fatigue_power_tradeoff.png", "all", "trade-off scatter — placeholder values")

    # history-schema fixtures: straight lines, by design
    h = placeholder_history("FOWT_ARISE")
    for col, ylabel, fname in (
        ("train_loss", "loss", "history_train_loss.png"),
        ("validation_loss", "loss", "history_validation_loss.png"),
        ("mean_reward", "reward", "history_train_reward.png"),
        ("validation_reward", "reward", "history_validation_reward.png"),
        ("learning_rate", "learning rate", "history_learning_rate.png"),
    ):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(h["epoch"], h[col], linewidth=3, color="#2471a3")
        ax.set_xlabel("epoch"); ax.set_ylabel(ylabel)
        ax.set_title(f"{col} — LINEAR RAMP fixture, not a training curve")
        save_fig(fig, pdir / fname, "FOWT_ARISE",
                 f"{col} column-schema fixture — deliberately a straight line, not a run")

    # action distributions
    ac = placeholder_action_records("FOWT_ARISE", np.random.default_rng(SEED))
    for col, xlabel, fname in (
        ("action_pitch_offset_deg", "pitch offset [deg]", "action_distribution_pitch.png"),
        ("action_ipc_level", "IPC level [-]", "action_distribution_ipc.png"),
        ("action_yaw_error_deg", "yaw [deg]", "action_distribution_yaw.png"),
    ):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist(ac[col], bins=40, color="#2471a3", edgecolor="black")
        ax.set_xlabel(xlabel); ax.set_ylabel("count")
        ax.set_title(f"{col} distribution (placeholder fixture)")
        save_fig(fig, pdir / fname, "FOWT_ARISE", f"{col} distribution — placeholder values")

    # no-action vs controllability -- the structural relationship the code must handle
    fig, ax = plt.subplots(figsize=(12, 8))
    bins = np.linspace(0, 1, 11)
    mid = 0.5 * (bins[1:] + bins[:-1])
    rate = [ac.loc[(ac["controllable_share"] >= a) & (ac["controllable_share"] < b),
                   "no_action"].mean() for a, b in zip(bins[:-1], bins[1:])]
    ax.plot(mid, rate, marker="o", linewidth=3, markersize=10, color="#943126")
    ax.set_xlabel("controllable-load share (placeholder)")
    ax.set_ylabel("no-action rate")
    ax.set_title("No-action vs controllability (structural fixture)")
    save_fig(fig, pdir / "no_action_vs_controllability.png", "FOWT_ARISE",
             "no-action rate against controllable-load share — structural fixture")

    # episode objective distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    ep = placeholder_episode_records("FOWT_ARISE", np.random.default_rng(SEED))
    ax.hist(ep["matched_sweep_objective"], bins=30, color="#1e8449", edgecolor="black")
    ax.set_xlabel("episode objective (placeholder)"); ax.set_ylabel("count")
    ax.set_title("Episode objective distribution (placeholder fixture)")
    save_fig(fig, pdir / "episode_objective_distribution.png", "FOWT_ARISE",
             "episode objective distribution — placeholder values")


def write_figure_index(out_root: Path) -> None:
    """Contact sheet with each caption BELOW its image, as required."""
    p = out_root / "FIGURE_INDEX.md"
    lines = [
        "# Figure index — SYNTHETIC / PLANNING ONLY", "",
        "Every figure below is a **placeholder fixture** for validating figure design and the",
        "plotting code path. None is a measurement. Values are not ordered to favour any",
        "configuration, and the history figures are deliberately straight lines.", "",
        "---", "",
    ]
    for path, desc in FIGURES:
        rel = path.relative_to(out_root)
        lines += [f"### {path.stem}", "", f"![{path.stem}]({rel})", "",
                  f"*{CAPTION}*", "", f"<sub>{desc}</sub>", "", "---", ""]
    p.write_text("\n".join(lines))
    MANIFEST.append({"artifact": "FIGURE_INDEX.md", "model": "all", "path": str(p),
                     "artifact_type": "markdown", "synthetic_status": SYNTH_STATUS,
                     "description": "figure contact sheet with captions below each image"})


def validate(out_root: Path, df: pd.DataFrame) -> None:
    checks = []

    def ck(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))

    print("\n" + "=" * 74)
    print("FIXTURE VALIDATION")
    print("=" * 74)
    for m in MODELS:
        ck(f"{m}: directory exists", (out_root / m).is_dir())
    required = ["metrics.json", "metrics.csv", "history.csv", "test_predictions.csv",
                "episode_metrics.csv", "action_statistics.csv", "robustness_metrics.csv",
                "training_summary.json", "validation_metrics.csv"]
    for m in MODELS:
        missing = [f for f in required if not (out_root / m / f).exists()]
        ck(f"{m}: required artefacts", not missing, "all present" if not missing else str(missing))

    csvs = sorted(out_root.rglob("*.csv"))
    no_marker = [p.name for p in csvs
                 if "data" not in pd.read_csv(p, nrows=0).columns and not p.name.startswith("REAL_")]
    ck("every synthetic CSV carries data=\"synthetic\"", not no_marker,
       f"{len(csvs)} CSVs checked" if not no_marker else str(no_marker))
    real_files = [p for p in csvs if p.name.startswith("REAL_")]
    ck("real reference values kept in their own files", len(real_files) == 2,
       f"{[p.name for p in real_files]}")

    pngs = sorted(out_root.rglob("*.png"))
    ck("figures written", len(pngs) > 0, f"{len(pngs)} PNGs")
    try:
        from PIL import Image
        bad = [p.name for p in pngs if Image.open(p).info.get("dpi", (0,))[0] < 299]
        ck("figures at 300 dpi", not bad, "verified via PIL" if not bad else str(bad[:5]))
    except Exception:
        ck("figures at 300 dpi", True, "PIL unavailable; saved with dpi=300")
    ck("captions rendered BELOW each image", (out_root / "FIGURE_INDEX.md").exists(),
       "FIGURE_INDEX.md")
    ck("plot fontsize 20", plt.rcParams["font.size"] == FONT_SIZE,
       f"rcParams font.size={plt.rcParams['font.size']}")

    num = df.select_dtypes(include=[np.number])
    ck("no NaN in aggregate table", not num.isna().any().any())
    ck("no Inf in aggregate table", not np.isinf(num.to_numpy()).any())
    ck("DEL ratios within (0, 1]", bool((df["Mean DEL Ratio"] > 0).all()
                                        and (df["Mean DEL Ratio"] <= 1).all()))
    ck("no-action rate within [0, 1]", bool((df["No-Action Rate"] >= 0).all()
                                            and (df["No-Action Rate"] <= 1).all()))
    ck("all five configurations present", len(df) == 5 and set(df["Method"]) == set(MODELS))

    # This is a fixture, so the ordering is reported, NOT asserted.
    order = df.sort_values("Mean Matched Sweep Objective", ascending=False)["Method"].tolist()
    print(f"\n    placeholder objective ordering (reported, not enforced): {order}")
    print("    NOTE: this fixture does not order values to favour any configuration, so whatever")
    print("    order appears above carries no meaning. A fixture that asserted 'FOWT_ARISE is best'")
    print("    would be encoding a conclusion into test data.")

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n    {n_pass}/{len(checks)} checks passed")
    pd.DataFrame([{"check": c, "status": "PASS" if ok else "FAIL", "detail": d}
                  for c, ok, d in checks]).assign(data="synthetic").to_csv(
        out_root / "validation" / "fixture_validation.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./fixtures", help="fixture output root")
    ap.add_argument("--real-output-root", default=None,
                    help="real experiment output root, to assert separation against")
    a = ap.parse_args()

    out = Path(a.out)
    real = Path(a.real_output_root) if a.real_output_root else None

    print("=" * 74)
    print("FOWT-ARISE PIPELINE FIXTURES — SYNTHETIC / PLANNING ONLY")
    print("=" * 74)
    assert_separated(out, real)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation").mkdir(parents=True, exist_ok=True)

    print(f"\nseed={SEED}  models={len(MODELS)}  episodes/model={N_EPISODES}  "
          f"steps/episode={N_STEPS}  history epochs={N_EPOCHS}")
    print(f"sample size: {N_EPISODES * N_STEPS:,} placeholder step records per model")

    df = build(out)
    make_figures(df, out)
    write_figure_index(out)

    print("\n" + "=" * 74)
    print("PLACEHOLDER AGGREGATE TABLE (fixture — no meaning)")
    print("=" * 74)
    show = ["Method", "Mean Matched Sweep Objective", "Mean DEL Ratio", "Fatigue Relief %",
            "Power Loss %", "Actuator Duty Proxy", "No-Action Rate", "IoT Performance Gap"]
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:,.5f}"))

    print("\n" + "=" * 74)
    print("REAL BASELINE REFERENCE VALUES (verbatim from the brief; kept in separate files)")
    print("=" * 74)
    for k, v in REAL_BASELINES.items():
        print(f"    {k:9s} objective={v['Mean Matched Sweep Objective']:+.4f}  "
              f"DEL={v['Mean DEL Ratio']:.4f}  fatigue={v['Fatigue Relief %']:.4f}%  "
              f"power={v['Power Loss %']:.4f}%")
    print(f"    {'Sweep ref':9s} objective="
          f"{ACTION_SWEEP_REFERENCE['Mean Matched Sweep Objective']:+.4f}  "
          f"(counterfactual reference, NOT a deployable policy)")
    print("\n    These are the only non-synthetic numbers here. They are not combined into any")
    print("    table with placeholder values, because a label column does not survive a")
    print("    copy-paste out of a shared table.")

    validate(out, df)

    man = pd.DataFrame(MANIFEST)
    man["data"] = "synthetic"
    man.to_csv(out / "synthetic_output_manifest.csv", index=False)
    print(f"\n    manifest: {len(man)} artefacts -> {out / 'synthetic_output_manifest.csv'}")

    print("\n" + "=" * 74)
    print("FIXTURE GENERATION COMPLETE")
    print("=" * 74)
    print("STATUS: SYNTHETIC / PLANNING ONLY")
    print("NOT FOR: experimental claims, research claims, publication results,")
    print("         thesis numerical results, scientific evidence")
    print("PURPOSE: output-structure validation, figure design, aggregation-code validation")
    print("\nNOT GENERATED, deliberately: simulated training curves, checkpoint validation")
    print("metrics, SHAP attributions, or values ordered so a chosen model wins. See")
    print("TARGET_PROFILE.md section 5.")
    print("=" * 74)


if __name__ == "__main__":
    main()
