from __future__ import annotations

from asoc_raeg_rank import MAIN_AIRPORTS, build_arg_parser, run_from_args


def main() -> None:
    parser = build_arg_parser(
        default_output="raeg_rank_full_10airports_2025",
        default_months="1-12",
        default_airports=MAIN_AIRPORTS,
    )
    args = parser.parse_args()
    out = run_from_args(args)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
