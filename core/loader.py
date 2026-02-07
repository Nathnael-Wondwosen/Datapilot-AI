from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    notes: Optional[List[str]] = None
    available_tables: Optional[List[str]] = None
    table_index: Optional[int] = None


def _normalize_col_name(x: object, idx: int) -> str:
    s = "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
    s = " ".join(s.replace("\n", " ").replace("\r", " ").split()).strip()
    s = s.rstrip(":").strip()
    if not s or s.lower().startswith("unnamed"):
        return f"col_{idx+1}"
    return s


def _dedupe_cols(cols: List[str]) -> List[str]:
    seen: dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        base = c
        n = seen.get(base, 0) + 1
        seen[base] = n
        out.append(base if n == 1 else f"{base}_{n}")
    return out


def _infer_header_from_excel_table(
    df_raw: pd.DataFrame,
    max_scan_rows: int = 30,
    keep_named_empty_cols: bool = True,
) -> tuple[pd.DataFrame, List[str]]:
    """
    Excel templates often have title rows and merged cells, so the true header isn't row 0.
    This tries to find a header row and returns a cleaned dataframe.
    """
    notes: List[str] = []
    df0 = df_raw.copy()

    # Drop fully-empty rows/cols first to reduce noise.
    df0 = df0.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df0.empty:
        return df0, ["Excel sheet is empty after removing blank rows/columns."]

    scan = min(max_scan_rows, len(df0))
    best_i = None
    best_score = float("-inf")
    keywords = {"no", "item", "items", "qty", "quantity", "unit", "price", "cost", "value", "date", "id", "total"}

    for i in range(scan):
        row = df0.iloc[i]
        non_null = int(row.notna().sum())
        if non_null < 2:
            continue

        texts = []
        kw_hits = 0
        for v in row.tolist():
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            if isinstance(v, str):
                t = " ".join(v.split()).strip().lower()
                if not t:
                    continue
                texts.append(t)
                if t in keywords:
                    kw_hits += 1
            else:
                # numeric-only rows are rarely headers
                pass

        text_count = len(texts)
        uniq_text = len(set(texts))
        # Heuristic score: prefer many non-null text cells, uniqueness, and keyword presence.
        score = (2.0 * non_null) + (3.0 * text_count) + (2.0 * uniq_text) + (6.0 * kw_hits)
        # Penalize if most cells are long free-text (often titles/notes).
        long_texts = sum(1 for t in texts if len(t) >= 30)
        score -= 2.0 * long_texts

        if score > best_score:
            best_score = score
            best_i = i

    # If we can't find a better header, fall back to first non-empty row as header.
    if best_i is None:
        best_i = 0
        notes.append("Could not confidently infer header row; using first non-empty row.")
    else:
        notes.append(f"Inferred header row at Excel row {best_i + 1}.")

    header_row = df0.iloc[best_i].tolist()
    cols = [_normalize_col_name(v, idx=j) for j, v in enumerate(header_row)]
    cols = _dedupe_cols(cols)

    df = df0.iloc[best_i + 1 :].copy()
    df.columns = cols[: len(df.columns)]
    df = df.dropna(axis=0, how="all")

    # Drop completely empty columns.
    # In Excel templates, some named columns may exist but be entirely empty (still useful to keep for editing).
    if keep_named_empty_cols:
        auto_cols = [c for c in df.columns if str(c).startswith("col_")]
        drop_cols = [c for c in auto_cols if df[c].isna().all()]
        if drop_cols:
            df = df.drop(columns=drop_cols)
    else:
        df = df.dropna(axis=1, how="all")

    # If there's a clear row-number column (e.g., "no"), keep only data rows that have it populated.
    lower_cols = {str(c).strip().lower(): str(c) for c in df.columns}
    no_col = None
    for candidate in ("no", "no.", "number", "row", "s/n", "sn"):
        if candidate in lower_cols:
            no_col = lower_cols[candidate]
            break
    if no_col:
        s = pd.to_numeric(df[no_col], errors="coerce")
        mask = s.notna()
        if int(mask.sum()) >= 1:
            df = df.loc[mask].copy()
            df[no_col] = s.loc[mask].astype("Int64")
            notes.append(f"Kept {int(mask.sum())} data rows where '{no_col}' is numeric.")

    return df.reset_index(drop=True), notes


def _header_score(row: pd.Series) -> float:
    non_null = int(row.notna().sum())
    if non_null < 2:
        return float("-inf")
    keywords = {"no", "item", "items", "qty", "quantity", "unit", "price", "cost", "value", "date", "id", "total"}
    texts: List[str] = []
    kw_hits = 0
    for v in row.tolist():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if isinstance(v, str):
            t = " ".join(v.split()).strip().lower()
            if not t:
                continue
            texts.append(t)
            if t in keywords:
                kw_hits += 1
    text_count = len(texts)
    uniq_text = len(set(texts))
    long_texts = sum(1 for t in texts if len(t) >= 30)
    return (2.0 * non_null) + (3.0 * text_count) + (2.0 * uniq_text) + (6.0 * kw_hits) - (2.0 * long_texts)


def _extract_tables_from_excel_grid(df_raw: pd.DataFrame, max_scan_rows: int = 80) -> Tuple[List[pd.DataFrame], List[str], List[str]]:
    """
    Attempt to extract multiple tables from a single Excel sheet grid.
    Returns (tables, table_labels, global_notes).
    """
    global_notes: List[str] = []

    grid0 = df_raw.copy()

    # Treat empty strings as missing.
    try:
        grid0 = grid0.replace(r"^\s*$", pd.NA, regex=True)
    except Exception:
        pass

    if grid0.dropna(axis=0, how="all").empty:
        return [], [], ["Excel sheet is empty after removing blank rows."]

    # 1) Prefer splitting by blank-row separators: this handles multiple distinct tables in one sheet.
    blank_row = grid0.isna().all(axis=1).to_numpy()
    separators: List[int] = []
    run = 0
    for i in range(len(blank_row)):
        run = run + 1 if blank_row[i] else 0
        if run == 2:
            separators.append(i)  # end of a section (exclusive)

    row_blocks: List[Tuple[int, int]] = []
    start = 0
    for sep in separators:
        end = sep - 1  # include the first blank row, exclude the second
        if end > start:
            row_blocks.append((start, end))
        start = sep + 1
    if start < len(grid0):
        row_blocks.append((start, len(grid0)))

    # Keep only blocks with enough non-empty rows.
    cleaned_blocks: List[Tuple[int, int]] = []
    for a, b in row_blocks:
        seg = grid0.iloc[a:b].copy()
        seg = seg.dropna(axis=0, how="all")
        if len(seg) >= 2:
            cleaned_blocks.append((a, b))

    # If we don't have multiple blocks, fall back to single-table inference.
    if len(cleaned_blocks) <= 1:
        df1, n1 = _infer_header_from_excel_table(df_raw, max_scan_rows=max_scan_rows, keep_named_empty_cols=True)
        return ([df1] if not df1.empty else []), (["Table 1"] if not df1.empty else []), n1

    tables: List[pd.DataFrame] = []
    labels: List[str] = []

    for (a, b) in cleaned_blocks:
        seg = grid0.iloc[a:b].copy()
        df1, notes = _infer_header_from_excel_table(seg, max_scan_rows=min(max_scan_rows, len(seg)), keep_named_empty_cols=True)
        if df1.empty or df1.shape[1] < 2:
            continue
        non_null_cells = int(df1.notna().sum().sum())
        # Keep sparse templates too (e.g., rows numbered but cells not filled yet).
        if non_null_cells < 2:
            continue
        labels.append(f"Table {len(labels)+1} (rows {a+1}-{b}, cells {non_null_cells})")
        tables.append(df1.reset_index(drop=True))
        if notes:
            global_notes.append(f"{labels[-1]}: " + "; ".join(notes))

    if not tables:
        df1, n1 = _infer_header_from_excel_table(df_raw, max_scan_rows=max_scan_rows, keep_named_empty_cols=True)
        return ([df1] if not df1.empty else []), (["Table 1"] if not df1.empty else []), n1

    global_notes.insert(0, f"Detected {len(tables)} table(s) in the sheet.")
    return tables, labels, global_notes


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


def load_excel(file_bytes: bytes, source_name: str, sheet_name: Optional[str], table_index: Optional[int] = None) -> LoadResult:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = list(xls.sheet_names)
    chosen = sheet_name or (sheets[0] if sheets else None)
    notes: List[str] = []

    # First attempt: standard header parsing.
    df = pd.read_excel(xls, sheet_name=chosen)

    # If this looks like a template (many Unnamed columns), try to infer headers.
    cols = [str(c) for c in df.columns]
    unnamed_ratio = sum(1 for c in cols if c.lower().startswith("unnamed")) / max(1, len(cols))
    if unnamed_ratio >= 0.4:
        df_raw = pd.read_excel(xls, sheet_name=chosen, header=None)
        tables, labels, gnotes = _extract_tables_from_excel_grid(df_raw)
        if tables:
            idx = int(table_index or 0)
            if idx < 0 or idx >= len(tables):
                idx = 0
            df = tables[idx]
            notes.extend(gnotes)
            if len(tables) > 1:
                notes.append(f"Selected {labels[idx]}.")
        else:
            df2, n2 = _infer_header_from_excel_table(df_raw, keep_named_empty_cols=True)
            if not df2.empty and len(df2.columns) >= 2:
                df = df2
                notes.extend(n2)
            else:
                notes.append("Detected template-like sheet, but header inference did not improve parsing; using default read.")

    return LoadResult(
        df=df,
        source_name=source_name,
        file_type="xlsx",
        sheet_name=chosen,
        available_sheets=sheets,
        notes=notes or None,
        available_tables=labels if (unnamed_ratio >= 0.4 and len(labels) > 1) else None,
        table_index=int(table_index or 0) if (unnamed_ratio >= 0.4 and len(labels) >= 1) else None,
    )


def load_tabular(
    file_bytes: bytes,
    source_name: str,
    file_type: str,
    sheet_name: Optional[str] = None,
    table_index: Optional[int] = None,
) -> LoadResult:
    ft = (file_type or "").lower().strip(".")
    if ft == "csv":
        return load_csv(file_bytes, source_name)
    if ft in ("xlsx", "xlsm", "xls"):
        return load_excel(file_bytes, source_name, sheet_name=sheet_name, table_index=table_index)
    raise ValueError(f"Unsupported file type: {file_type}")
