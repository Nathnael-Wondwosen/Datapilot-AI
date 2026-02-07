from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import io
import pandas as pd

from utils.helpers import detect_text_encoding


@dataclass(frozen=True)
class LoadResult:
    df: pd.DataFrame
    source_name: str
    file_type: str  # "csv" | "xlsx"
    sheet_name: Optional[str] = None
    available_sheets: Optional[List[str]] = None
    encoding: Optional[str] = None


def list_excel_sheets(file_bytes: bytes) -> List[str]:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    return list(xls.sheet_names)


def load_csv(file_bytes: bytes, source_name: str) -> LoadResult:
    enc = detect_text_encoding(file_bytes)
    df = pd.read_csv(
        io.BytesIO(file_bytes),
        encoding=enc,
        engine="python",
        on_bad_lines="skip",
    )
    return LoadResult(df=df, source_name=source_name, file_type="csv", encoding=enc)


def load_excel(file_bytes: bytes, source_name: str, sheet_name: Optional[str]) -> LoadResult:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = list(xls.sheet_names)
    chosen = sheet_name or (sheets[0] if sheets else None)
    df = pd.read_excel(xls, sheet_name=chosen)
    return LoadResult(
        df=df,
        source_name=source_name,
        file_type="xlsx",
        sheet_name=chosen,
        available_sheets=sheets,
    )


def load_tabular(file_bytes: bytes, source_name: str, file_type: str, sheet_name: Optional[str] = None) -> LoadResult:
    ft = (file_type or "").lower().strip(".")
    if ft == "csv":
        return load_csv(file_bytes, source_name)
    if ft in ("xlsx", "xlsm", "xls"):
        return load_excel(file_bytes, source_name, sheet_name=sheet_name)
    raise ValueError(f"Unsupported file type: {file_type}")
