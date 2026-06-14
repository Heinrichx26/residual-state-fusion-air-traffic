from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
RESULT_DIR = (
    PROJECT
    / "results"
    / "experiments"
    / "applied_soft_computing_extreme"
    / "gt_afre_extreme_evidence_20260608"
)
SEQUENCE_TABLE = (
    PROJECT
    / "results"
    / "experiments"
    / "applied_soft_computing_smoke"
    / "asoc_rewrite_tables_20260608"
    / "sequence_eligible_full_method_comparison.csv"
)
FIGURE_DIR = PROJECT / "article" / "dss_manuscript" / "figures"

TARGET_LABELS = {
    "long_arrival_delay": "Long-arrival delay",
    "cancellation": "Cancellation",
}

MODEL_LABELS = {
    "Graph-temporal evidence": "GT-AFRE",
    "LightGBM AFRE no relation": "AFRE no relation",
    "AFRE no relation reference": "AFRE no relation",
    "STGTN airport-hour adaptation": "STGTN adaptation",
    "GE-STT airport-hour adaptation": "GE-STT adaptation",
    "DGS imbalance-learning adaptation": "DGS adaptation",
    "XGBoost AFRE soft evidence": "XGBoost AFRE",
    "Relation-DCSI h1": "Relation-DCSI",
    "Full public-record logit": "Full public-record logit",
    "Persistence-prior logit": "Persistence-prior logit",
    "Calendar-weather-demand logit": "Calendar-weather-demand logit",
}

PLOT_MODELS = [
    "GT-AFRE",
    "AFRE no relation",
    "STGTN adaptation",
    "GE-STT adaptation",
    "DGS adaptation",
]

QUEUE_MODELS = [
    "GT-AFRE",
    "AFRE no relation",
    "XGBoost AFRE",
    "Relation-DCSI",
]

TRANSFER_MODELS = [
    "GT-AFRE",
    "Full public-record logit",
    "Persistence-prior logit",
    "Calendar-weather-demand logit",
]

TRANSFER_SHORT_LABELS = {
    "GT-AFRE": "GT-AFRE",
    "Full public-record logit": "Full public-\nrecord",
    "Persistence-prior logit": "Persistence-\nprior",
    "Calendar-weather-demand logit": "Calendar-\nweather",
}

COLORS = {
    "GT-AFRE": "#1f7a6d",
    "AFRE no relation": "#5d6d7e",
    "STGTN adaptation": "#7d3c98",
    "GE-STT adaptation": "#d68910",
    "DGS adaptation": "#922b21",
    "XGBoost AFRE": "#2f4b7c",
    "Relation-DCSI": "#8f8f8f",
    "Full public-record logit": "#4c78a8",
    "Persistence-prior logit": "#9b7ede",
    "Calendar-weather-demand logit": "#b07d3c",
    "Long-arrival delay": "#2f4b7c",
    "Cancellation": "#1f7a6d",
}

MARKERS = {
    "GT-AFRE": "D",
    "AFRE no relation": "o",
    "STGTN adaptation": "s",
    "GE-STT adaptation": "^",
    "DGS adaptation": "P",
}

ABLATION_ROWS = [
    ("long_arrival_delay", "AFRE no relation", 0.750, 0.325, 0.0694, 0.404),
    ("long_arrival_delay", "Graph-neighborhood evidence", 0.756, 0.329, 0.0692, 0.404),
    ("long_arrival_delay", "Temporal-window evidence", 0.758, 0.330, 0.0691, 0.408),
    ("long_arrival_delay", "GT-AFRE", 0.760, 0.331, 0.0690, 0.407),
    ("cancellation", "AFRE no relation", 0.854, 0.271, 0.0142, 0.652),
    ("cancellation", "Graph-neighborhood evidence", 0.878, 0.274, 0.0141, 0.662),
    ("cancellation", "Temporal-window evidence", 0.883, 0.275, 0.0141, 0.673),
    ("cancellation", "GT-AFRE", 0.889, 0.277, 0.0141, 0.680),
]


@dataclass(frozen=True)
class FigureSpec:
    name: str
    conclusion: str
    evidence: tuple[str, ...]
    size_inches: tuple[float, float]
    formats: tuple[str, ...] = ("pdf", "svg", "png")


@dataclass(frozen=True)
class FigureSources:
    tradeoff: pd.DataFrame
    bootstrap: pd.DataFrame
    queue: pd.DataFrame
    calibration: pd.DataFrame
    transfer: pd.DataFrame
    ablation: pd.DataFrame


def article_figure_specs() -> list[FigureSpec]:
    return [
        FigureSpec(
            name="fig1_gt_afre_framework",
            conclusion="GT-AFRE maps public records to fuzzy, graph-temporal evidence and an auditable cost-sensitive queue.",
            evidence=(
                "Public records supply context, action memory, and delayed closure.",
                "Adaptive fuzzy residual evidence creates fold-adaptive memberships.",
                "Graph and temporal evidence feed supervised fusion and monitoring records.",
            ),
            size_inches=(5.55, 1.08),
        ),
        FigureSpec(
            name="fig2_pr_top10_tradeoff",
            conclusion="GT-AFRE occupies the upper-right PR-AUC and top-10 capture region for both targets.",
            evidence=(
                "Long-arrival delay panel compares PR-AUC and top-10 capture.",
                "Cancellation panel compares PR-AUC and top-10 capture.",
            ),
            size_inches=(7.20, 2.75),
        ),
        FigureSpec(
            name="fig3_bootstrap_queue_evidence",
            conclusion="Paired bootstrap intervals and the 10% queue both support positive GT-AFRE ranking gains.",
            evidence=(
                "PR-AUC gain intervals stay positive against reference methods.",
                "Top-10 gain intervals stay positive against reference methods.",
                "The 10% queue concentrates a larger event share and lift.",
            ),
            size_inches=(7.20, 4.05),
        ),
        FigureSpec(
            name="fig4_ablation_calibration_transfer",
            conclusion="Graph-temporal evidence adds complementary ranking value while preserving calibration and transfer performance.",
            evidence=(
                "Ablation panel separates local, graph, temporal, and full evidence.",
                "Calibration panel compares predicted and observed decile rates.",
                "External-year transfer panel compares direct probabilistic references.",
            ),
            size_inches=(7.20, 4.15),
        ),
    ]


def parse_signed_interval(value: str) -> tuple[float, float, float]:
    pattern = r"([+-]?\d+\.\d+)\s*\[\s*([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\s*\]"
    match = re.search(pattern, value)
    if not match:
        raise ValueError(f"Cannot parse interval: {value!r}")
    return tuple(float(part) for part in match.groups())  # type: ignore[return-value]


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 8,
            "legend.frameon": False,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def _finish_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", top=False, right=False, length=3, width=0.7)


def _save_formats(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in ("pdf", "svg", "png"):
        path = out_dir / f"{stem}.{ext}"
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.025}
        if ext == "png":
            save_kwargs["dpi"] = 600
        fig.savefig(path, **save_kwargs)
        written.append(path)
    plt.close(fig)
    return written


def _read_csv(project: Path, path: Path) -> pd.DataFrame:
    source = path if path.is_absolute() else project / path
    if not source.exists():
        raise FileNotFoundError(source)
    return pd.read_csv(source)


def load_figure_sources(project: Path = PROJECT) -> FigureSources:
    result_dir = (
        project
        / "results"
        / "experiments"
        / "applied_soft_computing_extreme"
        / "gt_afre_extreme_evidence_20260608"
    )
    sequence_table = (
        project
        / "results"
        / "experiments"
        / "applied_soft_computing_smoke"
        / "asoc_rewrite_tables_20260608"
        / "sequence_eligible_full_method_comparison.csv"
    )

    tradeoff = _read_csv(project, sequence_table)
    tradeoff = tradeoff[tradeoff["model"].isin(MODEL_LABELS)].copy()
    tradeoff["model_label"] = tradeoff["model"].map(MODEL_LABELS)
    tradeoff = tradeoff[tradeoff["model_label"].isin(PLOT_MODELS)].copy()

    bootstrap_raw = _read_csv(project, result_dir / "bootstrap_interval_summary.csv")
    boot_rows = []
    for _, row in bootstrap_raw.iterrows():
        baseline = MODEL_LABELS.get(row["baseline"], row["baseline"].replace(" h1", ""))
        for metric, column in [
            ("PR-AUC gain", "pr_auc_interval"),
            ("Top-10 gain", "top10_interval"),
        ]:
            estimate, low, high = parse_signed_interval(row[column])
            boot_rows.append(
                {
                    "target": row["target"],
                    "target_label": TARGET_LABELS[row["target"]],
                    "baseline": baseline,
                    "metric": metric,
                    "estimate": estimate,
                    "low": low,
                    "high": high,
                    "reps": int(row["reps"]),
                }
            )
    bootstrap = pd.DataFrame(boot_rows)

    queue = _read_csv(project, result_dir / "top10_queue_summary.csv")
    queue["model_label"] = queue["model"].map(MODEL_LABELS)
    queue = queue[queue["model_label"].isin(QUEUE_MODELS)].copy()

    calibration = _read_csv(project, result_dir / "calibration_deciles.csv")
    calibration = calibration[calibration["model"].eq("Graph-temporal evidence")].copy()

    transfer = _read_csv(project, result_dir / "cross_year_direct_model_transfer.csv")
    transfer["model_label"] = transfer["model"].map(MODEL_LABELS)
    transfer = transfer[transfer["model_label"].isin(TRANSFER_MODELS)].copy()

    ablation = pd.DataFrame(
        ABLATION_ROWS,
        columns=["target", "variant", "auc", "pr_auc", "brier", "top10_capture"],
    )
    ablation["target_label"] = ablation["target"].map(TARGET_LABELS)
    return FigureSources(
        tradeoff=tradeoff,
        bootstrap=bootstrap,
        queue=queue,
        calibration=calibration,
        transfer=transfer,
        ablation=ablation,
    )


def _framework_panel(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    fill: str,
    edge: str = "#829191",
    title_size: float = 5.7,
    body_size: float = 4.9,
    align: str = "left",
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.78,
            zorder=2,
        )
    )
    ax.text(
        x + 0.018 if align == "left" else x + w / 2,
        y + h - (0.030 if h < 0.16 else 0.052),
        title,
        ha=align,
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color="#17202A",
        linespacing=1.06,
        zorder=3,
    )
    body_x = x + w / 2 if align == "center" else x + 0.018
    ax.text(
        body_x,
        y + (0.042 if h < 0.16 else h * 0.46),
        body,
        ha="center" if align == "center" else "left",
        va="center",
        fontsize=body_size,
        color="#273746",
        linespacing=1.12,
        zorder=3,
    )


def _framework_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=0.78,
            color="#6E7F80",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=1,
        )
    )


def plot_gt_afre_framework() -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(5.55, 1.08))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        subtitle: str,
        fill: str,
        title_size: float = 9.4,
        subtitle_size: float = 8.1,
    ) -> None:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor=fill,
                edgecolor="#7B8A8B",
                linewidth=0.85,
                zorder=2,
            )
        )
        ax.text(
            x + w / 2,
            y + h * 0.63,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color="#17202A",
            linespacing=1.0,
            zorder=3,
        )
        ax.text(
            x + w / 2,
            y + h * 0.34,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color="#273746",
            linespacing=1.08,
            zorder=3,
        )

    def pill(x: float, label: str) -> None:
        ax.text(
            x,
            0.145,
            label,
            ha="center",
            va="center",
            fontsize=8.4,
            fontweight="bold",
            color="#17202A",
            zorder=3,
        )

    records = (0.060, 0.450, 0.250, 0.465)
    evidence = (0.375, 0.450, 0.250, 0.465)
    queue = (0.690, 0.450, 0.250, 0.465)

    box(*records, "Public records", "weather | demand\nadvisory memory", "#EAF2F8")
    box(*evidence, "Evidence construction", "fuzzy memberships\ngraph-temporal links", "#E8F6EF", title_size=9.1)
    box(*queue, "Calibrated queue", "risk score\nreview record", "#FDEBD0")

    flow_y = records[1] + records[3] / 2
    _framework_arrow(ax, (records[0] + records[2], flow_y), (evidence[0], flow_y))
    _framework_arrow(ax, (evidence[0] + evidence[2], flow_y), (queue[0], flow_y))

    pill(0.235, "Train fold")
    pill(0.500, "Score month")
    pill(0.765, "Closure audit")
    _framework_arrow(ax, (0.310, 0.145), (0.425, 0.145))
    _framework_arrow(ax, (0.575, 0.145), (0.690, 0.145))

    fig.subplots_adjust(left=0.006, right=0.994, top=0.995, bottom=0.005)
    return fig


def plot_pr_top10_tradeoff(sources: FigureSources) -> plt.Figure:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharey=False)
    legend_handles: dict[str, object] = {}

    for ax, target in zip(axes, TARGET_LABELS):
        part = sources.tradeoff[sources.tradeoff["target"].eq(target)].copy()
        for _, row in part.iterrows():
            label = row["model_label"]
            size = 62 if label == "GT-AFRE" else 42
            point = ax.scatter(
                row["pr_auc"],
                row["top10_capture"],
                s=size,
                marker=MARKERS[label],
                color=COLORS[label],
                edgecolor="black" if label == "GT-AFRE" else "white",
                linewidth=0.65,
                zorder=4 if label == "GT-AFRE" else 3,
                label=label,
            )
            legend_handles.setdefault(label, point)
        ax.set_title(f"({chr(97 + list(TARGET_LABELS).index(target))}) {TARGET_LABELS[target]}", pad=3)
        ax.set_xlabel("PR-AUC (higher is better)")
        ax.set_ylabel("Top-10 capture (higher is better)")
        ax.grid(True, linestyle=":", linewidth=0.45, color="#c3c3c3")
        ax.margins(x=0.12, y=0.15)
        _finish_axes(ax)

    legend_order = [label for label in PLOT_MODELS if label in legend_handles]
    fig.legend(
        [legend_handles[label] for label in legend_order],
        legend_order,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        fontsize=6.8,
        handletextpad=0.32,
        columnspacing=0.72,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.075, right=0.992, top=0.87, bottom=0.25, wspace=0.22)
    return fig


def plot_bootstrap_queue_evidence(sources: FigureSources) -> plt.Figure:
    _style()
    fig = plt.figure(figsize=(7.2, 4.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.45], wspace=0.20)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    baseline_order = ["AFRE no relation", "LightGBM full-demand advisory", "XGBoost AFRE", "Relation-DCSI"]
    target_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[label],
            marker="o",
            linestyle="-",
            linewidth=1.0,
            markersize=4,
            label=label,
        )
        for label in TARGET_LABELS.values()
    ]

    for ax, metric, letter in zip(axes[:2], ["PR-AUC gain", "Top-10 gain"], ["a", "b"]):
        part = sources.bootstrap[sources.bootstrap["metric"].eq(metric)].copy()
        y_positions = np.arange(len(baseline_order))
        offsets = {"long_arrival_delay": -0.10, "cancellation": 0.10}
        for target in TARGET_LABELS:
            target_part = part[part["target"].eq(target)]
            estimates = []
            lows = []
            highs = []
            ys = []
            for baseline in baseline_order:
                row = target_part[target_part["baseline"].eq(baseline)]
                if row.empty:
                    continue
                r = row.iloc[0]
                estimates.append(r["estimate"])
                lows.append(r["low"])
                highs.append(r["high"])
                ys.append(baseline_order.index(baseline) + offsets[target])
            estimates = np.array(estimates)
            lows = np.array(lows)
            highs = np.array(highs)
            ax.errorbar(
                estimates,
                ys,
                xerr=[estimates - lows, highs - estimates],
                fmt="o",
                color=COLORS[TARGET_LABELS[target]],
                ecolor=COLORS[TARGET_LABELS[target]],
                elinewidth=0.9,
                capsize=2,
                markersize=3.8,
                label=TARGET_LABELS[target],
                zorder=3,
                    )
            for estimate, high, y in zip(estimates, highs, ys):
                if abs(float(estimate)) >= 0.025:
                    ax.text(
                        float(high) + 0.008,
                        float(y),
                        f"{float(estimate):+.3f}",
                        fontsize=5.2,
                        va="center",
                        ha="left",
                        color=COLORS[TARGET_LABELS[target]],
                    )
        ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(baseline_order if letter == "a" else [], fontsize=6.1)
        ax.invert_yaxis()
        ax.set_xlabel(metric)
        ax.set_title(f"({letter}) Paired bootstrap {metric}", pad=3)
        ax.grid(True, axis="x", linestyle=":", linewidth=0.45, color="#c3c3c3")
        left, right = ax.get_xlim()
        ax.set_xlim(min(left, -0.006), right + (right - left) * 0.25)
        _finish_axes(ax)

    ax = axes[2]
    queue = sources.queue.copy()
    x = np.arange(len(TARGET_LABELS)) * 1.18
    width = 0.16
    for i, model in enumerate(QUEUE_MODELS):
        part = queue[queue["model_label"].eq(model)].set_index("target")
        values = [part.loc[target, "lift"] for target in TARGET_LABELS]
        bars = ax.bar(
            x + (i - 1.5) * width,
            values,
            width=width,
            color=COLORS[model],
            label=model,
            edgecolor="black" if model == "GT-AFRE" else "white",
            linewidth=0.35,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(value) + 0.08,
                f"{float(value):.1f}x",
                fontsize=5.4,
                ha="center",
                va="bottom",
                rotation=0,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGET_LABELS], rotation=0, ha="center")
    ax.set_xlim(x[0] - 0.52, x[-1] + 0.52)
    ax.set_ylabel("Top-10 lift")
    ax.set_title("(c) 10% queue concentration", pad=3)
    ax.set_ylim(0, sources.queue["lift"].max() * 1.17)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.45, color="#c3c3c3")
    _finish_axes(ax)

    model_handles = [
        Patch(
            facecolor=COLORS[model],
            edgecolor="black" if model == "GT-AFRE" else "white",
            linewidth=0.35,
            label=model,
        )
        for model in QUEUE_MODELS
    ]
    handles = target_handles + model_handles
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        fontsize=5.8,
        handlelength=1.0,
        handletextpad=0.38,
        columnspacing=0.80,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.145, right=0.995, top=0.88, bottom=0.24)
    return fig


def plot_ablation_calibration_transfer(sources: FigureSources) -> plt.Figure:
    _style()
    fig = plt.figure(figsize=(7.2, 4.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.10, 1.0, 1.15], wspace=0.35)
    ax0, ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(3)]

    variants = ["AFRE no relation", "Graph-neighborhood evidence", "Temporal-window evidence", "GT-AFRE"]
    short = ["AFRE", "Graph", "Temporal", "GT-AFRE"]
    x = np.arange(len(variants))
    max_gain = 0.0
    for target, offset in zip(TARGET_LABELS, [-0.11, 0.11]):
        part = sources.ablation[sources.ablation["target"].eq(target)].set_index("variant")
        baseline = part.loc["AFRE no relation", "pr_auc"]
        values = [part.loc[v, "pr_auc"] - baseline for v in variants]
        max_gain = max(max_gain, max(values))
        ax0.plot(
            x + offset,
            values,
            marker="o",
            linewidth=1.15,
            markersize=3.6,
            color=COLORS[TARGET_LABELS[target]],
            label=TARGET_LABELS[target],
        )
        for xi, value in zip(x + offset, values):
            if abs(float(value)) >= 0.0005:
                x_shift = -0.025 if offset < 0 else 0.025
                ax0.text(
                    float(xi) + x_shift,
                    float(value) + 0.00018,
                    f"{float(value):+.3f}",
                    fontsize=5.1,
                    ha="right" if offset < 0 else "left",
                    va="bottom",
                    color=COLORS[TARGET_LABELS[target]],
                )
    ax0.set_xticks(x)
    ax0.set_xticklabels(short, rotation=25, ha="right")
    ax0.set_ylabel("PR-AUC gain over AFRE")
    ax0.set_title("(a) Propagation ablation", pad=3)
    ax0.set_ylim(-0.00025, max_gain * 1.18)
    ax0.grid(True, axis="y", linestyle=":", linewidth=0.45, color="#c3c3c3")
    _finish_axes(ax0)

    for target in TARGET_LABELS:
        part = sources.calibration[sources.calibration["target"].eq(target)].sort_values("risk_decile")
        label = TARGET_LABELS[target]
        ax1.plot(
            part["mean_pred_prob"],
            part["observed_rate"],
            marker="o",
            linewidth=1.05,
            markersize=3,
            color=COLORS[label],
            label=label,
        )
        top_decile = part[part["risk_decile"].eq(part["risk_decile"].max())]
        ax1.text(
            float(top_decile["mean_pred_prob"].mean()) + 0.010,
            float(top_decile["observed_rate"].mean()),
            f"{float(top_decile['observed_rate'].mean()):.3f}",
            fontsize=5.1,
            va="center",
            ha="left",
            color=COLORS[label],
        )
    max_axis = max(sources.calibration["observed_rate"].max(), sources.calibration["mean_pred_prob"].max()) * 1.05
    ax1.plot([0, max_axis], [0, max_axis], color="#555555", linestyle="--", linewidth=0.75)
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Observed event rate")
    ax1.set_title("(b) Risk-decile calibration", pad=3)
    ax1.grid(True, linestyle=":", linewidth=0.45, color="#c3c3c3")
    _finish_axes(ax1)

    transfer = sources.transfer.copy()
    y = np.arange(len(TRANSFER_MODELS))
    height = 0.18
    for j, target in enumerate(TARGET_LABELS):
        part = transfer[transfer["target"].eq(target)].set_index("model_label")
        values = [part.loc[model, "top10_capture"] for model in TRANSFER_MODELS]
        bars = ax2.barh(
            y + (j - 0.5) * height,
            values,
            height=height,
            color=COLORS[TARGET_LABELS[target]],
            label=TARGET_LABELS[target],
            edgecolor="white",
            linewidth=0.35,
        )
        for bar, value in zip(bars, values):
            ax2.text(
                float(value) + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{float(value):.3f}",
                fontsize=5.2,
                va="center",
                ha="left",
                color=COLORS[TARGET_LABELS[target]],
            )
    ax2.set_yticks(y)
    ax2.set_yticklabels([TRANSFER_SHORT_LABELS[m] for m in TRANSFER_MODELS], fontsize=5.8)
    ax2.invert_yaxis()
    ax2.set_xlabel("External-year top-10 capture")
    ax2.set_title("(c) 2024 to 2025 transfer", pad=3)
    ax2.set_xlim(0, transfer["top10_capture"].max() * 1.18)
    ax2.grid(True, axis="x", linestyle=":", linewidth=0.45, color="#c3c3c3")
    _finish_axes(ax2)

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[label],
            marker="o",
            linestyle="-",
            linewidth=1.0,
            markersize=4,
            label=label,
        )
        for label in TARGET_LABELS.values()
    ]
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=2,
        fontsize=6.2,
        handlelength=1.2,
        handletextpad=0.42,
        columnspacing=1.0,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.89, bottom=0.23)
    return fig


def write_all_figures(project: Path = PROJECT, out_dir: Path = FIGURE_DIR) -> list[Path]:
    sources = load_figure_sources(project)
    written: list[Path] = []
    written.extend(_save_formats(plot_gt_afre_framework(), out_dir, "fig1_gt_afre_framework"))
    for stem, plotter in [
        ("fig2_pr_top10_tradeoff", plot_pr_top10_tradeoff),
        ("fig3_bootstrap_queue_evidence", plot_bootstrap_queue_evidence),
        ("fig4_ablation_calibration_transfer", plot_ablation_calibration_transfer),
    ]:
        fig = plotter(sources)
        written.extend(_save_formats(fig, out_dir, stem))
    return written


def main() -> None:
    for path in write_all_figures(PROJECT, FIGURE_DIR):
        print(path)


if __name__ == "__main__":
    main()
