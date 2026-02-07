from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from utils.helpers import coerce_datetime, format_compact_number, pick_best_datetime_column


@dataclass(frozen=True)
class KPI:
    name: str
    value: str
    detail: Optional[str] = None


@dataclass(frozen=True)
class KPIResult:
    kpis: List[KPI]
    top_tables: Dict[str, pd.DataFrame]  # title -> df


def compute_kpis(df: pd.DataFrame, roles: Dict[str, str]) -> KPIResult:
    kpis: List[KPI] = []
    top_tables: Dict[str, pd.DataFrame] = {}

    numeric_cols = [c for c, r in roles.items() if r == "numeric"]
    cat_cols = [c for c, r in roles.items() if r == "categorical"]

    kpis.append(KPI("Rows", str(len(df))))
    kpis.append(KPI("Columns", str(len(df.columns))))

    if numeric_cols:
        for c in numeric_cols[:6]:
            s = pd.to_numeric(df[c], errors="coerce")
            kpis.append(KPI(f"Avg {c}", format_compact_number(s.mean(skipna=True))))
            kpis.append(KPI(f"Sum {c}", format_compact_number(s.sum(skipna=True))))

    dt_col = pick_best_datetime_column(df, roles)
    if dt_col and numeric_cols:
        dt, ratio = coerce_datetime(df[dt_col])
        if ratio >= 0.6:
            tmp = df.copy()
            tmp[dt_col] = dt
            tmp = tmp.dropna(subset=[dt_col])
            if not tmp.empty:
                tmp["_period"] = tmp[dt_col].dt.to_period("M").dt.to_timestamp()
                target = numeric_cols[0]
                tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
                agg = tmp.dropna(subset=[target]).groupby("_period", as_index=False)[target].sum()
                if len(agg) >= 2:
                    last = float(agg[target].iloc[-1])
                    prev = float(agg[target].iloc[-2])
                    delta = last - prev
                    pct = (delta / prev) if prev != 0 else None
                    detail = None if pct is None else f"{pct*100:.1f}% vs prior month"
                    kpis.append(KPI(f"MoM change ({target})", format_compact_number(delta), detail=detail))

    if numeric_cols and cat_cols:
        num = numeric_cols[0]
        for cat in cat_cols[:2]:
            g = (
                df[[cat, num]]
                .copy()
                .assign(**{num: pd.to_numeric(df[num], errors="coerce")})
                .dropna(subset=[cat, num])
            )
            if g.empty:
                continue
            top = (
                g.groupby(cat, as_index=False)[num]
                .sum()
                .sort_values(num, ascending=False)
                .head(10)
                .reset_index(drop=True)
            )
            top_tables[f"Top {cat} by {num}"] = top

    return KPIResult(kpis=kpis, top_tables=top_tables)

