"""Purchase recommendations from your own demand history.

    python -m csc demand.csv [--stock stock.csv] [--lead 3] [--target 0.95] [--out recommendations.csv]

demand.csv: one row per part and month, columns part, month (YYYY-MM), qty. Months missing for a part count as
zero demand. stock.csv (optional): part, on_hand, on_order.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .data import from_long
from .recommend import FLAG_TEXT, recommend


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m csc", description=__doc__.splitlines()[0])
    ap.add_argument("demand")
    ap.add_argument("--stock")
    ap.add_argument("--lead", type=int, default=3, help="lead time in months")
    ap.add_argument("--target", type=float, default=0.95, help="service target used for the safety stock")
    ap.add_argument("--out", default="recommendations.csv")
    a = ap.parse_args(argv)
    d = pd.read_csv(a.demand, dtype={"part": str, "month": str})
    missing = {"part", "month", "qty"} - set(d.columns)
    if missing:
        print(f"demand file is missing columns: {sorted(missing)}", file=sys.stderr)
        return 2
    if d["qty"].lt(0).any():
        print(f"{int(d['qty'].lt(0).sum())} negative quantities (returns?) set to zero", file=sys.stderr)
        d["qty"] = d["qty"].clip(lower=0)
    stock = pd.read_csv(a.stock, dtype={"part": str}) if a.stock else None
    try:
        rec = recommend(from_long(d), a.lead, a.target, stock)
    except ValueError as err:
        print(err, file=sys.stderr)
        return 2
    rec.to_csv(a.out, index=False)
    print(rec["demand_class"].value_counts().to_string())
    for key, text in FLAG_TEXT.items():
        n = int(rec["flags"].str.contains(key).sum())
        if n:
            print(f"{n} parts: {text}")
    if stock is not None:
        print(f"{int((rec['proposed_order'] > 0).sum())} parts to order, {int(rec['proposed_order'].sum())} units.")
    print(f"Written to {a.out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
