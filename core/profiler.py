from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from utils.helpers import coerce_datetime, infer_column_roles, safe_jsonable


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    role: str
    dtype: str
    non_nulls: int
    nulls: int
    null_pct: float
    uniques: int
    unique_pct: float
    examples: List[str]


@dataclass(frozen=True)
class DatasetProfile:
    rows: int
    cols: int
    roles: Dict[str, str]
    columns: List[ColumnProfile]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    identifier_columns: List[str]
    numeric_summary: Optional[pd.DataFrame]
    correlation: Optional[pd.DataFrame]


def _col_examples(series: pd.Series, n: int = 3) -> List[str]:
    vals = series.dropna().head(50)
    if vals.empty:
        return []
    uniq: List[str] = []
    for v in vals.tolist():
        s = str(v)
        if s not in uniq:
            uniq.append(s)
        if len(uniq) >= n:
            break
    return uniq


def profile_df(df: pd.DataFrame) -> DatasetProfile:
    df2 = df.copy()
    df2.columns = [str(c) for c in df2.columns]

    roles = infer_column_roles(df2)
    rows, cols = int(df2.shape[0]), int(df2.shape[1])

    cols_prof: List[ColumnProfile] = []
    for col in df2.columns:
        s = df2[col]
        non_nulls = int(s.notna().sum())
        nulls = int(s.isna().sum())
        uniq = int(s.nunique(dropna=True))
        cols_prof.append(
            ColumnProfile(
                name=col,
                role=roles.get(col, "categorical"),
                dtype=str(s.dtype),
                non_nulls=non_nulls,
                nulls=nulls,
                null_pct=float(nulls / max(1, len(s))),
                uniques=uniq,
                unique_pct=float(uniq / max(1, len(s))),
                examples=_col_examples(s),
            )
        )

    numeric_columns = [c for c, r in roles.items() if r == "numeric"]
    categorical_columns = [c for c, r in roles.items() if r == "categorical"]
    datetime_columns = [c for c, r in roles.items() if r == "datetime"]
    identifier_columns = [c for c, r in roles.items() if r == "identifier"]

    num_summary = None
    corr = None
    if numeric_columns:
        num_summary = df2[numeric_columns].describe().T
        if len(numeric_columns) >= 2:
            corr = df2[numeric_columns].corr(numeric_only=True)

    # Ensure datetime-ish columns are actually converted for downstream features
    for c in datetime_columns:
        dt, ratio = coerce_datetime(df2[c])
        if ratio >= 0.5:
            df2[c] = dt

    return DatasetProfile(
        rows=rows,
        cols=cols,
        roles=roles,
        columns=cols_prof,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        identifier_columns=identifier_columns,
        numeric_summary=num_summary,
        correlation=corr,
    )


def profile_as_dict(profile: DatasetProfile) -> Dict:
    return safe_jsonable(
        {
            "rows": profile.rows,
            "cols": profile.cols,
            "roles": profile.roles,
            "columns": [
                {
                    "name": c.name,
                    "role": c.role,
                    "dtype": c.dtype,
                    "non_nulls": c.non_nulls,
                    "nulls": c.nulls,
                    "null_pct": c.null_pct,
                    "uniques": c.uniques,
                    "unique_pct": c.unique_pct,
                    "examples": c.examples,
                }
                for c in profile.columns
            ],
            "numeric_columns": profile.numeric_columns,
            "categorical_columns": profile.categorical_columns,
            "datetime_columns": profile.datetime_columns,
            "identifier_columns": profile.identifier_columns,
        }
    )

