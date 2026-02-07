from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import plotly.express as px

from utils.helpers import coerce_datetime, pick_best_datetime_column


@dataclass(frozen=True)
class Visual:
    title: str
    fig: object  # plotly Figure


def build_visuals(df: pd.DataFrame, roles: dict) -> List[Visual]:
    visuals: List[Visual] = []

    numeric_cols = [c for c, r in roles.items() if r == "numeric"]
    cat_cols = [c for c, r in roles.items() if r == "categorical"]
    dt_col = pick_best_datetime_column(df, roles)

    if dt_col and numeric_cols:
        dt, ratio = coerce_datetime(df[dt_col])
        if ratio >= 0.6:
            tmp = df.copy()
            tmp[dt_col] = dt
            tmp = tmp.dropna(subset=[dt_col])
            tmp["_period"] = tmp[dt_col].dt.to_period("D").dt.to_timestamp()
            y = numeric_cols[0]
            tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
            ts = tmp.dropna(subset=[y]).groupby("_period", as_index=False)[y].sum()
            if len(ts) >= 2:
                visuals.append(Visual(f"Time Series: {y} over time", px.line(ts, x="_period", y=y)))

    if numeric_cols:
        y = numeric_cols[0]
        s = pd.to_numeric(df[y], errors="coerce")
        hist_df = pd.DataFrame({y: s})
        visuals.append(Visual(f"Distribution: {y}", px.histogram(hist_df, x=y, nbins=30)))

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True)
            visuals.append(Visual("Correlation Heatmap", px.imshow(corr, text_auto=".2f", aspect="auto")))

        box_df = df[numeric_cols[: min(5, len(numeric_cols))]].melt(var_name="metric", value_name="value")
        visuals.append(Visual("Box Plot (numeric columns)", px.box(box_df, x="metric", y="value")))

    if numeric_cols and cat_cols:
        num = numeric_cols[0]
        cat = cat_cols[0]
        tmp = df[[cat, num]].copy()
        tmp[num] = pd.to_numeric(tmp[num], errors="coerce")
        tmp = tmp.dropna(subset=[cat, num])
        if not tmp.empty:
            top = tmp.groupby(cat, as_index=False)[num].sum().sort_values(num, ascending=False).head(12)
            visuals.append(Visual(f"Top {cat} by {num}", px.bar(top, x=cat, y=num)))

            if top[cat].nunique() >= 3:
                visuals.append(Visual(f"Share of {num} (top {cat})", px.pie(top, names=cat, values=num, hole=0.45)))

    return visuals

