from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_gt_afre_article_figures import (
    TARGET_LABELS,
    article_figure_specs,
    load_figure_sources,
    parse_signed_interval,
    plot_ablation_calibration_transfer,
    plot_bootstrap_queue_evidence,
    plot_gt_afre_framework,
    write_all_figures,
)


def test_parse_signed_interval_reads_estimate_and_bounds() -> None:
    assert parse_signed_interval("+0.007 [+0.003, +0.010]") == (0.007, 0.003, 0.010)
    assert parse_signed_interval("+0.000 [-0.001, +0.002]") == (0.0, -0.001, 0.002)


def test_article_figure_specs_define_four_main_figures() -> None:
    specs = article_figure_specs()
    assert [spec.name for spec in specs] == [
        "fig1_gt_afre_framework",
        "fig2_pr_top10_tradeoff",
        "fig3_bootstrap_queue_evidence",
        "fig4_ablation_calibration_transfer",
    ]
    assert all(spec.formats == ("pdf", "svg", "png") for spec in specs)
    assert all(spec.conclusion for spec in specs)


def test_load_figure_sources_filters_to_manuscript_evidence() -> None:
    sources = load_figure_sources(Path(__file__).resolve().parents[2])

    assert set(sources.tradeoff["target"]) == set(TARGET_LABELS)
    assert sources.tradeoff.groupby("target").size().to_dict() == {
        "cancellation": 5,
        "long_arrival_delay": 5,
    }
    assert len(sources.bootstrap) == 16
    assert set(sources.bootstrap["metric"]) == {"PR-AUC gain", "Top-10 gain"}
    assert set(sources.queue["model_label"]) == {"GT-AFRE", "AFRE no relation", "XGBoost AFRE", "Relation-DCSI"}
    assert set(sources.calibration["risk_decile"]) == set(range(1, 11))
    assert set(sources.transfer["model_label"]) == {
        "GT-AFRE",
        "Full public-record logit",
        "Persistence-prior logit",
        "Calendar-weather-demand logit",
    }


def test_write_all_figures_exports_pdf_svg_and_png() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        written = write_all_figures(Path(__file__).resolve().parents[2], out_dir)

        expected = {
            out_dir / f"{stem}.{ext}"
            for stem in [
                "fig1_gt_afre_framework",
                "fig2_pr_top10_tradeoff",
                "fig3_bootstrap_queue_evidence",
                "fig4_ablation_calibration_transfer",
            ]
            for ext in ["pdf", "svg", "png"]
        }
        assert expected.issubset(set(written))
        for path in expected:
            assert path.exists()
            assert path.stat().st_size > 1000


def test_fig1_is_python_generated_framework_with_core_evidence_labels() -> None:
    fig = plot_gt_afre_framework()
    text_objects = [text for ax in fig.axes for text in ax.texts]
    labels = [" ".join(text.get_text().split()) for text in text_objects]
    text_blob = " | ".join(labels)

    expected = [
        "Public records",
        "Evidence construction",
        "fuzzy memberships",
        "graph-temporal links",
        "Calibrated queue",
        "Train fold",
        "Score month",
        "Closure audit",
    ]
    for label in expected:
        assert label in text_blob
    assert max(len(label) for label in labels) <= 58
    for retired_label in [
        "Public operational records",
        "Supervised fusion and calibration",
        "Cost-sensitive queue and monitoring record",
        "GT-AFRE evidence construction",
        "training-period airport correlations",
        "delayed outcome closure",
        "Fuzzy memberships",
        "Graph-temporal evidence",
    ]:
        assert retired_label not in text_blob
    width, height = fig.get_size_inches()
    assert width <= 5.7
    assert height <= 1.10
    assert min(text.get_fontsize() for text in text_objects) >= 8.0
    assert len(fig.axes) == 1
    assert not fig.axes[0].axison


def _annotation_texts(fig) -> list[str]:
    return [text.get_text() for ax in fig.axes for text in ax.texts]


def _legend_texts(fig) -> list[str]:
    return [text.get_text() for legend in fig.legends for text in legend.get_texts()]


def test_fig3_uses_bottom_figure_legend_and_value_labels() -> None:
    sources = load_figure_sources(Path(__file__).resolve().parents[2])
    fig = plot_bootstrap_queue_evidence(sources)

    assert all(ax.get_legend() is None for ax in fig.axes)
    assert set(_legend_texts(fig)) == {
        "Long-arrival delay",
        "Cancellation",
        "GT-AFRE",
        "AFRE no relation",
        "XGBoost AFRE",
        "Relation-DCSI",
    }
    annotations = _annotation_texts(fig)
    assert "+0.087" in annotations
    assert "+0.159" in annotations
    assert "7.1x" in annotations


def test_fig3_uses_compact_spacing_and_horizontal_queue_labels() -> None:
    sources = load_figure_sources(Path(__file__).resolve().parents[2])
    fig = plot_bootstrap_queue_evidence(sources)
    positions = [ax.get_position() for ax in fig.axes]

    assert positions[2].width > positions[0].width * 1.15
    assert positions[1].x0 - positions[0].x1 < 0.075
    assert positions[2].x0 - positions[1].x1 < 0.075
    assert all(label.get_rotation() == 0 for label in fig.axes[2].get_xticklabels())
    queue_value_labels = [text for text in fig.axes[2].texts if text.get_text().endswith("x")]
    assert queue_value_labels
    assert all(text.get_rotation() == 0 for text in queue_value_labels)


def test_fig4_uses_bottom_figure_legend_and_value_labels() -> None:
    sources = load_figure_sources(Path(__file__).resolve().parents[2])
    fig = plot_ablation_calibration_transfer(sources)

    assert all(ax.get_legend() is None for ax in fig.axes)
    assert set(_legend_texts(fig)) == {"Long-arrival delay", "Cancellation"}
    annotations = _annotation_texts(fig)
    assert "+0.006" in annotations
    assert "0.673" in annotations
    assert "0.389" in annotations


if __name__ == "__main__":
    test_parse_signed_interval_reads_estimate_and_bounds()
    test_article_figure_specs_define_four_main_figures()
    test_load_figure_sources_filters_to_manuscript_evidence()
    test_write_all_figures_exports_pdf_svg_and_png()
    test_fig1_is_python_generated_framework_with_core_evidence_labels()
    test_fig3_uses_bottom_figure_legend_and_value_labels()
    test_fig3_uses_compact_spacing_and_horizontal_queue_labels()
    test_fig4_uses_bottom_figure_legend_and_value_labels()
    print("GT-AFRE article figure tests passed")
