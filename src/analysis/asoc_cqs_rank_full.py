from __future__ import annotations

from asoc_cqs_rank import MAIN_AIRPORTS, build_arg_parser, run_from_args


if __name__ == "__main__":
    default_panel = (
        "results/experiments/applied_soft_computing_smoke/"
        "raeg_scope_panel_2025_base10_allmonths_cf_severity/raeg_scope_panel.csv"
    )
    default_ref = (
        "results/experiments/applied_soft_computing_smoke/"
        "raeg_rank_base10_2025_cf_severity_residual01_component_ablation_qrolling_e3/raeg_rank_predictions.csv"
    )
    parser = build_arg_parser(
        "cqs_rank_base10_2025_rolling_quarter",
        "1-12",
        MAIN_AIRPORTS,
        default_panel,
        default_ref,
    )
    parser.set_defaults(validation="rolling_quarter", first_test_month=4, min_train_months=3, trees=260)
    out = run_from_args(parser.parse_args())
    print(f"wrote {out}")
