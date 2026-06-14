from __future__ import annotations

from asoc_cqs_rank import SMOKE_AIRPORTS, build_arg_parser, run_from_args


if __name__ == "__main__":
    default_panel = (
        "results/experiments/applied_soft_computing_smoke/"
        "raeg_scope_panel_smoke_2025_4airports_2months_cf_severity/raeg_scope_panel.csv"
    )
    default_ref = (
        "results/experiments/applied_soft_computing_smoke/"
        "raeg_rank_cf_severity_smoke_2025_4airports_2months_residualforced01_e3/raeg_rank_predictions.csv"
    )
    parser = build_arg_parser(
        "cqs_rank_smoke_4airports_2months",
        "1,2",
        SMOKE_AIRPORTS,
        default_panel,
        default_ref,
    )
    parser.set_defaults(targets="long_arrival_delay,cancellation", trees=160)
    out = run_from_args(parser.parse_args())
    print(f"wrote {out}")
