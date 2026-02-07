from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.helpers import ApiConfig


ALLOWED_OPS = {
    "drop_columns",
    "rename_columns",
    "add_column_expr",
    "add_column_constant",
    "fill_missing",
    "cast_type",
    "deduplicate",
}


@dataclass(frozen=True)
class EditPlan:
    ops: List[Dict[str, Any]]
    notes: str = ""


def _json_loads_maybe(text: str) -> Any:
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty response")

    # Try direct JSON.
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to extract JSON object from a fenced block or surrounding text.
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        return json.loads(s[start : end + 1])

    raise ValueError("Response was not valid JSON")


def _validate_plan(obj: Any) -> EditPlan:
    if not isinstance(obj, dict):
        raise ValueError("Plan must be a JSON object")
    ops = obj.get("ops")
    if not isinstance(ops, list) or not ops:
        raise ValueError("Plan must include a non-empty 'ops' list")

    out_ops: List[Dict[str, Any]] = []
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            raise ValueError(f"Op {i} must be an object")
        name = op.get("op")
        if name not in ALLOWED_OPS:
            raise ValueError(f"Op {i} has unsupported op '{name}'")
        out_ops.append(op)

    notes = obj.get("notes") or ""
    if not isinstance(notes, str):
        notes = str(notes)
    return EditPlan(ops=out_ops, notes=notes)


def plan_edits(api: ApiConfig, instruction: str, df: pd.DataFrame) -> EditPlan:
    """
    Ask the LLM to produce a strict JSON edit plan limited to ALLOWED_OPS.
    """
    if not api.api_key:
        raise ValueError("Missing API key")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise ValueError("openai package is not available") from e

    client = OpenAI(api_key=api.api_key, base_url=api.base_url) if api.base_url else OpenAI(api_key=api.api_key)

    cols = [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns]
    sample = df.head(5).to_dict(orient="records")

    system = (
        "You are DataPilot Edit Copilot. Convert user edit instructions into a STRICT JSON plan.\n"
        "Rules:\n"
        "- Output JSON only (no markdown, no explanations outside JSON).\n"
        "- Use only allowed ops.\n"
        "- Never reference columns that do not exist.\n"
        "- Prefer minimal changes.\n"
        "- If the request is ambiguous, include a best-effort plan and write the question in notes.\n\n"
        "Allowed ops schema:\n"
        "1) drop_columns: {op:'drop_columns', columns:[...]}.\n"
        "2) rename_columns: {op:'rename_columns', mapping:{old:new,...}}.\n"
        "3) add_column_expr: {op:'add_column_expr', name:'NewCol', expr:'<DataFrame.eval expression>'}.\n"
        "   - expr must be compatible with pandas.DataFrame.eval using engine='numexpr'.\n"
        "   - Use backticks for columns with spaces, e.g. `Order Amount`.\n"
        "4) add_column_constant: {op:'add_column_constant', name:'NewCol', value:<string|number|bool|null>}.\n"
        "5) fill_missing: {op:'fill_missing', column:'Col', method:'value|mean|median|mode', value?:<...>}.\n"
        "6) cast_type: {op:'cast_type', column:'Col', to:'numeric|text|datetime'}.\n"
        "7) deduplicate: {op:'deduplicate', subset?:[...], keep:'first|last'}.\n\n"
        "Response format:\n"
        "{'ops':[...], 'notes':'...'}"
    )

    user = (
        f"Instruction: {instruction}\n\n"
        f"Columns: {cols}\n\n"
        f"Sample rows (first 5): {sample}\n"
    )

    resp = client.chat.completions.create(
        model=api.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=500,
    )
    content = (resp.choices[0].message.content or "").strip()
    obj = _json_loads_maybe(content)
    return _validate_plan(obj)


def apply_edit_plan(df: pd.DataFrame, plan: EditPlan) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply a validated plan deterministically.
    Returns (new_df, log_lines).
    """
    out = df.copy()
    log: List[str] = []

    for op in plan.ops:
        name = op["op"]

        if name == "drop_columns":
            cols = [str(c) for c in (op.get("columns") or [])]
            keep = [c for c in out.columns if c not in cols]
            out = out[keep]
            log.append(f"Dropped columns: {cols}")

        elif name == "rename_columns":
            mapping = op.get("mapping") or {}
            if not isinstance(mapping, dict):
                raise ValueError("rename_columns.mapping must be an object")
            mapping = {str(k): str(v) for k, v in mapping.items()}
            out = out.rename(columns=mapping)
            log.append(f"Renamed columns: {mapping}")

        elif name == "add_column_expr":
            col = str(op.get("name") or "").strip()
            expr = str(op.get("expr") or "").strip()
            if not col or not expr:
                raise ValueError("add_column_expr requires name and expr")
            # Use numexpr for safety; it supports arithmetic and boolean expressions.
            # If it's missing, require installation rather than falling back to python eval.
            try:
                out[col] = out.eval(expr, engine="numexpr")
            except ImportError as e:
                raise ValueError("numexpr is required for add_column_expr. Install it with: pip install -r requirements.txt") from e
            log.append(f"Added column '{col}' from expr: {expr}")

        elif name == "add_column_constant":
            col = str(op.get("name") or "").strip()
            if not col:
                raise ValueError("add_column_constant requires name")
            out[col] = op.get("value")
            log.append(f"Added constant column '{col}'")

        elif name == "fill_missing":
            col = str(op.get("column") or "").strip()
            method = str(op.get("method") or "").strip().lower()
            if col not in out.columns:
                raise ValueError(f"fill_missing column not found: {col}")
            if method == "value":
                out[col] = out[col].fillna(op.get("value"))
                log.append(f"Filled missing in '{col}' with value")
            elif method == "mean":
                s = pd.to_numeric(out[col], errors="coerce")
                out[col] = s.fillna(s.mean())
                log.append(f"Filled missing in '{col}' with mean")
            elif method == "median":
                s = pd.to_numeric(out[col], errors="coerce")
                out[col] = s.fillna(s.median())
                log.append(f"Filled missing in '{col}' with median")
            elif method == "mode":
                mode = out[col].mode(dropna=True)
                fill = mode.iloc[0] if not mode.empty else None
                out[col] = out[col].fillna(fill)
                log.append(f"Filled missing in '{col}' with mode")
            else:
                raise ValueError(f"Unsupported fill_missing method: {method}")

        elif name == "cast_type":
            col = str(op.get("column") or "").strip()
            to = str(op.get("to") or "").strip().lower()
            if col not in out.columns:
                raise ValueError(f"cast_type column not found: {col}")
            if to == "numeric":
                out[col] = pd.to_numeric(out[col], errors="coerce")
            elif to == "text":
                out[col] = out[col].astype("string")
            elif to == "datetime":
                out[col] = pd.to_datetime(out[col], errors="coerce")
            else:
                raise ValueError(f"Unsupported cast_type.to: {to}")
            log.append(f"Casted '{col}' to {to}")

        elif name == "deduplicate":
            subset = op.get("subset")
            keep = str(op.get("keep") or "first").strip().lower()
            if keep not in ("first", "last"):
                keep = "first"
            if subset is None:
                out = out.drop_duplicates(keep=keep)
                log.append(f"Deduplicated rows (keep={keep})")
            else:
                cols = [str(c) for c in subset]
                out = out.drop_duplicates(subset=cols, keep=keep)
                log.append(f"Deduplicated rows on {cols} (keep={keep})")

        else:
            raise ValueError(f"Unsupported op: {name}")

    return out, log
