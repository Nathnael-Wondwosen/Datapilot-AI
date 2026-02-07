from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class OutlierFinding:
    column: str
    method: str  # "zscore"
    count: int
    pct: float


@dataclass(frozen=True)
class QualityReport:
    missing_by_column: pd.DataFrame
    outliers: List[OutlierFinding]


def compute_quality(df: pd.DataFrame, numeric_columns: List[str]) -> QualityReport:
    miss = (
        pd.DataFrame(
            {
                "column": list(df.columns),
                "missing": [int(df[c].isna().sum()) for c in df.columns],
                "missing_pct": [float(df[c].isna().mean()) for c in df.columns],
            }
        )
        .sort_values(["missing_pct", "missing"], ascending=False)
        .reset_index(drop=True)
    )

    outliers: List[OutlierFinding] = []
    for c in numeric_columns:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 10:
            continue
        mu = float(s.mean())
        sd = float(s.std(ddof=0))
        if sd <= 0:
            continue
        z = (s - mu) / sd
        cnt = int((z.abs() > 3.0).sum())
        if cnt <= 0:
            continue
        outliers.append(OutlierFinding(column=c, method="zscore", count=cnt, pct=float(cnt / max(1, len(s)))))

    outliers.sort(key=lambda o: o.pct, reverse=True)
    return QualityReport(missing_by_column=miss, outliers=outliers)

