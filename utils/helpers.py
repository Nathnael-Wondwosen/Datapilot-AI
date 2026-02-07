from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def load_dotenv_if_present(path: str = ".env") -> None:
    """
    Minimal .env loader (KEY=VALUE).
    - Ignores comments and blank lines
    - Does not override existing environment variables
    """
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f.readlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k and k not in os.environ:
                    os.environ[k] = v
    except OSError:
        return


def detect_text_encoding(data: bytes) -> str:
    """
    Best-effort encoding detection for CSV bytes using charset-normalizer.
    Defaults to utf-8 if uncertain.
    """
    try:
        from charset_normalizer import from_bytes  # type: ignore

        matches = from_bytes(data)
        best = matches.best()
        if best and best.encoding:
            return best.encoding
    except Exception:
        pass
    return "utf-8"


def coerce_datetime(series: pd.Series) -> Tuple[pd.Series, float]:
    """
    Try to convert a series to datetime. Returns (converted_series, success_ratio).
    """
    s = series.copy()
    if pd.api.types.is_datetime64_any_dtype(s):
        return s, 1.0
    if pd.api.types.is_numeric_dtype(s):
        # Avoid turning numeric IDs into dates
        return s, 0.0
    # Try ISO first to avoid slow per-row parsing and noisy warnings.
    converted = pd.to_datetime(s, errors="coerce", utc=False, format="ISO8601")
    if len(converted) and float(converted.notna().mean()) >= 0.8:
        ratio = float(converted.notna().mean())
        return converted, ratio

    converted = pd.to_datetime(s, errors="coerce", utc=False)
    ratio = float(converted.notna().mean()) if len(converted) else 0.0
    return converted, ratio


def is_identifier_like(series: pd.Series, col_name: Optional[str] = None) -> bool:
    if len(series) == 0:
        return False

    name = (col_name or "").strip().lower()
    name_suggests_id = bool(re.search(r"(^|[_\\-\\s])(id|uuid|guid|key)([_\\-\\s]|$)", name))

    if pd.api.types.is_numeric_dtype(series):
        # Only treat numeric columns as identifiers when they look explicitly like IDs.
        if not name_suggests_id:
            return False
        uniq_ratio = series.nunique(dropna=True) / max(1, len(series))
        # Guard against small datasets where uniqueness is common.
        return len(series) >= 50 and uniq_ratio > 0.98

    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        s = series.dropna().astype(str)
        if s.empty:
            return False
        uniq_ratio = s.nunique() / max(1, len(s))
        avg_len = float(s.str.len().mean())
        return len(s) >= 30 and (name_suggests_id or (uniq_ratio > 0.98 and avg_len >= 8))

    return False


def infer_column_roles(df: pd.DataFrame) -> Dict[str, str]:
    """
    Returns role per column: numeric | datetime | categorical | identifier
    """
    roles: Dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            # Numeric IDs are hard to distinguish; only classify as identifier if the column name suggests it.
            roles[col] = "identifier" if is_identifier_like(s, col_name=str(col)) else "numeric"
            continue

        # Only attempt datetime parsing when it is plausible (avoid slow parsing and noisy warnings).
        col_name = str(col)
        name = col_name.strip().lower()
        name_hint = bool(re.search(r"(^|[_\\-\\s])(date|time|datetime|timestamp|created|updated)([_\\-\\s]|$)", name))
        sample = s.dropna().astype(str).head(20)
        value_hint = False
        if not sample.empty:
            # ISO-like dates/times (very common in CSV/Excel exports)
            value_hint = bool(sample.str.match(r"^\\d{4}[-/]\\d{1,2}[-/]\\d{1,2}").mean() >= 0.6) or bool(
                sample.str.match(r"^\\d{1,2}[-/]\\d{1,2}[-/]\\d{2,4}").mean() >= 0.6
            )

        if name_hint or value_hint:
            dt, ratio = coerce_datetime(s)
            if ratio >= 0.8:
                roles[col] = "datetime"
                continue

        if is_identifier_like(s, col_name=str(col)):
            roles[col] = "identifier"
            continue

        roles[col] = "categorical"
    return roles


def pick_best_datetime_column(df: pd.DataFrame, roles: Dict[str, str]) -> Optional[str]:
    dts = [c for c, r in roles.items() if r == "datetime"]
    if not dts:
        return None
    dts.sort(key=lambda c: float(pd.to_datetime(df[c], errors="coerce").notna().mean()), reverse=True)
    return dts[0]


def format_compact_number(x: float) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if ax >= 1_000:
        return f"{x/1_000:.2f}K"
    if ax >= 1:
        return f"{x:.2f}"
    return f"{x:.4f}"


def sanitize_question(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\\s+", " ", q)
    return q


@dataclass(frozen=True)
class ApiConfig:
    provider: str  # "openai_compatible"
    api_key: Optional[str]
    base_url: Optional[str]
    model: str


def load_api_config() -> ApiConfig:
    load_dotenv_if_present()
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("DATAPILOT_MODEL") or os.environ.get("OPENAI_MODEL")

    # Convenience: if the user provided a Groq key but forgot the base URL, default it.
    if api_key and api_key.startswith("gsk_") and not base_url:
        base_url = "https://api.groq.com/openai/v1"

    if not model:
        # If pointing at Groq, default to a strong free-tier model; otherwise keep a sane OpenAI default.
        if base_url and "groq.com" in base_url.lower():
            model = "llama-3.3-70b-versatile"
        else:
            model = "gpt-4o-mini"

    return ApiConfig(
        provider="openai_compatible",
        api_key=api_key,
        base_url=base_url,
        model=model,
    )


def safe_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_jsonable(x) for x in obj]
    return str(obj)
