from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

from utils.helpers import coerce_datetime


@dataclass(frozen=True)
class ForecastResult:
    history: pd.DataFrame  # columns: ds, y
    forecast: pd.DataFrame  # columns: ds, yhat
    model_name: str


def _infer_freq(ds: pd.Series) -> str:
    try:
        f = pd.infer_freq(ds.sort_values())
        if f:
            return f
    except Exception:
        pass
    return "D"


def forecast_time_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int = 30,
) -> Optional[ForecastResult]:
    dt, ratio = coerce_datetime(df[date_col])
    if ratio < 0.6:
        return None

    tmp = df[[date_col, target_col]].copy()
    tmp[date_col] = dt
    tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, target_col])
    if tmp.empty:
        return None

    tmp = tmp.sort_values(date_col)
    tmp = tmp.rename(columns={date_col: "ds", target_col: "y"})

    freq = _infer_freq(tmp["ds"])
    tmp = tmp.set_index("ds").groupby(pd.Grouper(freq=freq))["y"].sum().reset_index()
    tmp = tmp.dropna()
    if len(tmp) < 10:
        return None

    X = tmp["ds"].map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
    y = tmp["y"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    last = tmp["ds"].max()
    future_ds = pd.date_range(start=last, periods=horizon + 1, freq=freq, inclusive="right")
    Xf = future_ds.map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
    yhat = model.predict(Xf)
    yhat = np.maximum(yhat, 0)

    hist = tmp.copy()
    fc = pd.DataFrame({"ds": future_ds, "yhat": yhat})
    return ForecastResult(history=hist, forecast=fc, model_name="LinearRegression (date ordinal)")


def forecast_figure(res: ForecastResult, title: str = "Forecast") -> object:
    h = res.history.copy()
    h["type"] = "history"
    h = h.rename(columns={"y": "value"})
    f = res.forecast.copy()
    f["type"] = "forecast"
    f = f.rename(columns={"yhat": "value"})
    both = pd.concat([h[["ds", "value", "type"]], f[["ds", "value", "type"]]], ignore_index=True)
    fig = px.line(both, x="ds", y="value", color="type", title=title)
    return fig

