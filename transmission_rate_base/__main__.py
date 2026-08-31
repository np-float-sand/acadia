from __future__ import annotations

import argparse

from transmission_rate_base import config, report


def main() -> None:
    ap = argparse.ArgumentParser(prog="transmission_rate_base")
    ap.add_argument("--start", default=config.PRIMARY_START)
    ap.add_argument("--end", default=config.PRIMARY_END)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--refresh-ferc", action="store_true")
    ap.add_argument("--pre-thesis", action="store_true")
    ap.add_argument("--sensitivity", action="store_true",
                    help="(writeup only) also print gross-vs-net / 5yr-window variants")
    args = ap.parse_args()

    res = report.run_pipeline(start=args.start, end=args.end, offline=args.offline,
                              refresh_ferc=args.refresh_ferc, pre_thesis=args.pre_thesis)
    print("\n".join(res["verdict"]["reasons"]))


if __name__ == "__main__":
    main()
