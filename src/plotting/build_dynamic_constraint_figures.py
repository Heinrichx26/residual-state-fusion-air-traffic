from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "src" / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from dynamic_constraint_state_inversion import MAIN_10, MAIN_2025_PANEL, attach_dynamic, load_panel, model_rows  # noqa: E402


STRENGTH = ROOT / "results" / "experiments" / "fusion_framework_strengthening"
DCSI = STRENGTH / "dynamic_constraint_state_inversion_full_2025"
OUT_DIR = ROOT / "article" / "elsarticle_manuscript" / "figures"
RESULT_DIR = STRENGTH / "event_case_full"

RHO = 0.95
CASE = {
    "airport": "EWR",
    "tmi_type": "GDP",
    "start": pd.Timestamp("2025-02-06 12:26:00"),
    "end": pd.Timestamp("2025-02-07 02:59:00"),
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
        }
    )


def weighted_mean(df: pd.DataFrame, col: str) -> float:
    weights = df["arrivals"].astype(float)
    if weights.sum() <= 0:
        return float("nan")
    return float((df[col].astype(float) * weights).sum() / weights.sum())


def summarize_window(window: pd.DataFrame, name: str) -> dict[str, float | str]:
    return {
        "window": name,
        "hours": int(len(window)),
        "arrivals": int(window["arrivals"].sum()),
        "mild_weather_share": weighted_mean(window, "mild_weather_abs"),
        "long_delay_rate": weighted_mean(window, "arr_delay60_rate"),
        "cancel_rate": weighted_mean(window, "cancel_rate"),
        "constraint_state": weighted_mean(window, "dynamic_constraint_state"),
        "recovery_state": weighted_mean(window, "dynamic_recovery_state"),
    }


def build_event_curve() -> None:
    curve = pd.read_csv(DCSI / "dcsi_event_peak_curve.csv")
    fig, ax_state = plt.subplots(figsize=(7.2, 3.15), constrained_layout=True)
    ax_out = ax_state.twinx()

    ax_state.axvline(0, color="#555555", linestyle="--", linewidth=1.0)
    ax_state.axvspan(0, 6, color="#F3E8D8", alpha=0.42)
    ax_state.plot(curve["rel_hour"], curve["mean_state"], color="#B45309", marker="o", linewidth=1.7, label="Constraint state")
    ax_state.plot(
        curve["rel_hour"],
        curve["mean_recovery_state"],
        color="#7B5EA7",
        marker="s",
        linewidth=1.4,
        label="Recovery state",
    )
    ax_out.plot(curve["rel_hour"], curve["delay_rate"], color="#1F5C99", marker="^", linewidth=1.4, label="Long-delay rate")
    ax_out.plot(curve["rel_hour"], curve["cancel_rate"], color="#1F7A45", marker="d", linewidth=1.3, label="Cancellation rate")

    ax_state.set_xlabel("Hours from advisory start")
    ax_state.set_ylabel("DCSI state")
    ax_out.set_ylabel("Outcome rate")
    ax_state.set_xlim(-6.3, 18.3)
    ax_state.set_ylim(0, 8.2)
    ax_out.set_ylim(0, 0.42)
    ax_state.grid(True, axis="y", linewidth=0.35, alpha=0.45)
    ax_state.annotate("advisory start", xy=(0, 7.4), xytext=(0.6, 7.65), fontsize=8, color="#333333")
    ax_state.annotate("state peak region", xy=(9, 7.3), xytext=(11.0, 7.65), fontsize=8, color="#B45309")
    ax_out.annotate("delay peak", xy=(4, 0.36), xytext=(5.2, 0.385), fontsize=8, color="#1F5C99")

    lines_a, labels_a = ax_state.get_legend_handles_labels()
    lines_b, labels_b = ax_out.get_legend_handles_labels()
    ax_state.legend(lines_a + lines_b, labels_a + labels_b, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    fig.savefig(OUT_DIR / "fig4_dynamic_constraint_event_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def build_deployment_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.45), constrained_layout=True)
    ax.set_axis_off()
    steps = [
        ("Source ingestion", "four source roles", "#1F5C99"),
        ("Clock alignment", "issue, active, recovery", "#B45309"),
        ("Action impulse", "GDP/GS overlap", "#7B5EA7"),
        ("DCSI state update", "memory and recovery", "#B45309"),
        ("Closure diagnostics", "outcomes and controls", "#1F7A45"),
    ]
    xs = [0.10, 0.30, 0.50, 0.70, 0.90]
    y = 0.62
    w = 0.16
    h = 0.24
    for i, (title, subtitle, color) in enumerate(steps):
        box = FancyBboxPatch(
            (xs[i] - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.15,
            edgecolor="#111827",
            facecolor="white",
        )
        ax.add_patch(box)
        ax.plot([xs[i] - w / 2 + 0.015, xs[i] + w / 2 - 0.015], [y + h / 2 - 0.045, y + h / 2 - 0.045], color=color, linewidth=2.0)
        ax.text(xs[i], y + 0.025, title, ha="center", va="center", fontsize=9.0, fontweight="bold", color="#111827")
        ax.text(xs[i], y - 0.055, subtitle, ha="center", va="center", fontsize=6.9, color="#374151")
        if i < len(steps) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (xs[i] + w / 2 + 0.012, y),
                    (xs[i + 1] - w / 2 - 0.012, y),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=1.1,
                    color="#111827",
                )
            )
    ax.text(0.5, 0.92, "Deployment workflow", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#111827")
    ax.plot([0.11, 0.89], [0.30, 0.30], linestyle="--", color="#374151", linewidth=0.8)
    lower = [
        ("source roles", "#1F5C99"),
        ("availability", "#B45309"),
        ("constraint state", "#7B5EA7"),
        ("negative controls", "#B45309"),
        ("transfer audit", "#1F7A45"),
    ]
    for x, (label, color) in zip(xs, lower):
        ax.plot([x - 0.045, x + 0.045], [0.20, 0.20], color=color, linewidth=1.7)
        ax.text(x, 0.16, label, ha="center", va="center", fontsize=7.2, color="#111827", fontweight="bold")
    ax.text(
        0.5,
        0.055,
        "Each airport-hour retains source evidence, timestamp status, inferred constraint state, memory parameter, and closure record.",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#374151",
    )
    fig.savefig(OUT_DIR / "fig3_deployment_workflow.pdf", bbox_inches="tight")
    plt.close(fig)


def build_event_case() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel(MAIN_2025_PANEL, 2025, list(range(1, 13)), MAIN_10)
    panel = model_rows(attach_dynamic(panel, RHO))

    airport = CASE["airport"]
    start = CASE["start"]
    end = CASE["end"]
    start_hour = start.floor("h")
    end_hour = end.ceil("h")
    case = panel[
        (panel["airport"] == airport)
        & (panel["utc_hour"] >= start_hour - pd.Timedelta(hours=4))
        & (panel["utc_hour"] <= end_hour + pd.Timedelta(hours=4))
    ].copy()
    case["relative_hour"] = (case["utc_hour"] - start_hour).dt.total_seconds() / 3600

    pre = case[(case["utc_hour"] >= start_hour - pd.Timedelta(hours=3)) & (case["utc_hour"] < start_hour)]
    active = case[(case["utc_hour"] < end_hour) & (case["utc_hour"] + pd.Timedelta(hours=1) > start)]
    post = case[(case["utc_hour"] >= end_hour) & (case["utc_hour"] < end_hour + pd.Timedelta(hours=3))]
    summary = pd.DataFrame(
        [
            summarize_window(pre, "pre -3 to 0 h"),
            summarize_window(active, "active GDP"),
            summarize_window(post, "post 0 to 3 h"),
        ]
    )
    summary.to_csv(RESULT_DIR / "event_case_summary_dcsi.csv", index=False)
    case.to_csv(RESULT_DIR / "event_case_timeline_dcsi.csv", index=False)

    fig, ax_state = plt.subplots(figsize=(7.2, 3.05), constrained_layout=True)
    ax_out = ax_state.twinx()
    end_x = (end - start_hour).total_seconds() / 3600
    ax_state.axvspan(0, end_x, color="#F3E8D8", alpha=0.62, label="GDP active")
    ax_state.axvspan(end_x, end_x + 3, color="#E8F2EA", alpha=0.70, label="Post 3 h")
    ax_state.axvline(0, color="#B45309", linestyle="-", linewidth=1.2)
    ax_state.axvline(end_x, color="#1F7A45", linestyle="-", linewidth=1.2)

    ax_state.plot(
        case["relative_hour"],
        case["dynamic_constraint_state"],
        marker="s",
        linewidth=1.6,
        color="#B45309",
        label="DCSI state",
    )
    ax_out.plot(
        case["relative_hour"],
        case["arr_delay60_rate"],
        marker="o",
        linewidth=1.5,
        color="#1F5C99",
        label="Long-delay rate",
    )
    ax_state.annotate("start", xy=(0, 8.2), xytext=(4, 7), textcoords="offset points", ha="left", fontsize=8)
    ax_state.annotate("end", xy=(end_x, 8.2), xytext=(4, 7), textcoords="offset points", ha="left", fontsize=8)
    ax_state.set_title("Decision-support case trace: EWR GDP, February 6--7, 2025", loc="center", pad=8)
    ax_state.set_xlabel("Hours from GDP start")
    ax_state.set_ylabel("DCSI state")
    ax_out.set_ylabel("Long-delay rate")
    ax_state.set_xlim(case["relative_hour"].min() - 0.25, case["relative_hour"].max() + 0.25)
    ax_state.set_ylim(-0.2, 9.0)
    ax_out.set_ylim(-0.04, 0.88)
    ax_state.grid(True, axis="y", linewidth=0.3, alpha=0.45)

    lines_a, labels_a = ax_state.get_legend_handles_labels()
    lines_b, labels_b = ax_out.get_legend_handles_labels()
    ax_state.legend(lines_a + lines_b, labels_a + labels_b, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    fig.savefig(OUT_DIR / "fig5_event_case_timeline.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    build_deployment_workflow()
    build_event_curve()
    build_event_case()
    print(OUT_DIR / "fig3_deployment_workflow.pdf")
    print(OUT_DIR / "fig4_dynamic_constraint_event_curve.pdf")
    print(OUT_DIR / "fig5_event_case_timeline.pdf")


if __name__ == "__main__":
    main()
