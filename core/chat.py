from __future__ import annotations

from typing import List, Optional

import pandas as pd

from utils.helpers import ApiConfig, coerce_datetime, pick_best_datetime_column, sanitize_question


def _keyword_match_summary(df: pd.DataFrame, keyword: str, max_cols: int = 8) -> str:
    """
    Offline fallback: count rows related to a keyword by searching text-like columns.
    """
    kw = (keyword or "").strip()
    if not kw:
        return "Provide a keyword to search for."

    text_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
    if not text_cols:
        return "No text-like columns found to search."

    per_col = []
    for c in text_cols:
        s = df[c].astype("string")
        m = s.str.contains(kw, case=False, na=False, regex=False)
        cnt = int(m.sum())
        if cnt > 0:
            per_col.append((c, cnt))

    if not per_col:
        return f"No rows matched '{kw}' in text-like columns."

    per_col.sort(key=lambda x: x[1], reverse=True)
    cols_shown = per_col[:max_cols]

    # Union across the shown columns (a practical approximation for "related").
    union_mask = None
    for c, _ in cols_shown:
        m = df[c].astype("string").str.contains(kw, case=False, na=False, regex=False)
        union_mask = m if union_mask is None else (union_mask | m)
    union_cnt = int(union_mask.sum()) if union_mask is not None else 0

    lines = [f"- Rows with '{kw}' in any of the top {len(cols_shown)} matching columns: {union_cnt:,}"]
    lines.append("- Breakdown (matches by column):")
    for c, cnt in cols_shown:
        lines.append(f"  - {c}: {cnt:,}")
    return "\n".join(lines)


def _basic_answer(question: str, df: pd.DataFrame, profile_dict: dict) -> str:
    q = question.lower()
    if "how many rows" in q or "row count" in q:
        return f"Rows: {len(df):,}."
    if "how many columns" in q or "column count" in q:
        return f"Columns: {len(df.columns):,}."
    if "columns" in q:
        cols = ", ".join([str(c) for c in df.columns[:50]])
        suffix = "" if len(df.columns) <= 50 else " (showing first 50)"
        return f"Columns: {cols}{suffix}."
    if "missing" in q or "null" in q:
        miss = df.isna().mean().sort_values(ascending=False).head(10)
        parts = [f"{idx}: {val*100:.1f}%" for idx, val in miss.items()]
        return "Missingness (top 10): " + "; ".join(parts)
    if "correlation" in q or "correlate" in q:
        roles = profile_dict.get("roles", {}) or {}
        nums = [c for c, r in roles.items() if r == "numeric"]
        if len(nums) < 2:
            return "Not enough numeric columns to compute correlations."
        corr = df[nums].corr(numeric_only=True)
        best = None
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                v = float(corr.iloc[i, j])
                if pd.isna(v):
                    continue
                if best is None or abs(v) > abs(best[2]):
                    best = (nums[i], nums[j], v)
        if not best:
            return "Could not compute a reliable correlation from the available numeric data."
        a, b, v = best
        return f"Strongest correlation (by absolute value) is between '{a}' and '{b}': {v:.2f}."
    if "top" in q or "best" in q:
        roles = profile_dict.get("roles", {}) or {}
        cats = [c for c, r in roles.items() if r == "categorical"]
        nums = [c for c, r in roles.items() if r == "numeric"]
        if cats and nums:
            cat = cats[0]
            num = nums[0]
            tmp = df[[cat, num]].copy()
            tmp[num] = pd.to_numeric(tmp[num], errors="coerce")
            tmp = tmp.dropna(subset=[cat, num])
            if not tmp.empty:
                top = tmp.groupby(cat, as_index=False)[num].sum().sort_values(num, ascending=False).head(5)
                lines = [f"{row[cat]}: {row[num]:,.2f}" for _, row in top.iterrows()]
                return f"Top {cat} by total {num}:\n" + "\n".join([f"- {x}" for x in lines])
    if "summarize" in q or "summary" in q:
        return (
            f"Dataset has {profile_dict.get('rows'):,} rows and {profile_dict.get('cols'):,} columns. "
            f"Detected roles: {profile_dict.get('roles', {})}."
        )

    # Keyword-style questions (offline): "related with X", "about X", "contains X"
    for phrase in ["related with", "related to", "about", "contains", "contain"]:
        if phrase in q:
            kw = question.split(phrase, 1)[-1].strip().strip("?").strip('"').strip("'")
            if kw:
                return _keyword_match_summary(df, kw)

    return "I can answer questions about row/column counts, columns, missingness, and basic summaries without an API key. If you set OPENAI_API_KEY, I can answer broader questions."


def answer_question(
    api: ApiConfig,
    question: str,
    df: pd.DataFrame,
    profile_dict: dict,
    kpis_text: str,
    history: Optional[List[dict]] = None,
) -> str:
    q = sanitize_question(question)
    if not q:
        return "Ask a question about this dataset."

    if not api.api_key:
        return _basic_answer(q, df, profile_dict)

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return _basic_answer(q, df, profile_dict)

    client = OpenAI(api_key=api.api_key, base_url=api.base_url) if api.base_url else OpenAI(api_key=api.api_key)

    roles = profile_dict.get("roles", {}) or {}
    nums = [c for c, r in roles.items() if r == "numeric"]
    cats = [c for c, r in roles.items() if r == "categorical"]
    dt_col = pick_best_datetime_column(df, roles)

    # Tiny, concrete aggregates improve answer quality without leaking the full dataset.
    extra = {}
    if cats and nums:
        cat = cats[0]
        num = nums[0]
        tmp = df[[cat, num]].copy()
        tmp[num] = pd.to_numeric(tmp[num], errors="coerce")
        tmp = tmp.dropna(subset=[cat, num])
        if not tmp.empty:
            extra["top_categories"] = (
                tmp.groupby(cat, as_index=False)[num].sum().sort_values(num, ascending=False).head(10).to_dict(orient="records")
            )

    if dt_col and nums:
        dt, ratio = coerce_datetime(df[dt_col])
        if ratio >= 0.6:
            target = nums[0]
            tmp = df[[dt_col, target]].copy()
            tmp[dt_col] = dt
            tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
            tmp = tmp.dropna(subset=[dt_col, target]).sort_values(dt_col)
            if not tmp.empty:
                tmp["_m"] = tmp[dt_col].dt.to_period("M").dt.to_timestamp()
                monthly = tmp.groupby("_m", as_index=False)[target].sum().tail(12)
                extra["recent_monthly"] = monthly.to_dict(orient="records")

    sample = df.head(8).to_dict(orient="records")
    context = (
        "Use ONLY the provided dataset context. Do not invent values. "
        "If you cannot answer from this context, say exactly what is missing and propose a specific next step.\n"
        "Prefer short, direct answers. Use bullets when helpful. If asked for a number, lead with the number.\n\n"
        f"Schema (col -> role): {profile_dict.get('roles', {})}\n\n"
        f"KPIs:\n{kpis_text}\n\n"
        f"Extra aggregates (if available): {extra}\n\n"
        f"Sample rows (first 8): {sample}\n"
    )

    # Conversation memory: include a small window of prior turns.
    convo: List[dict] = []
    if history:
        for m in history[-8:]:
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                convo.append({"role": role, "content": content.strip()})

    try:
        resp = client.chat.completions.create(
            model=api.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are DataPilot, a careful data analyst. Keep answers concise, professional, and fast. Do not hallucinate.",
                },
                {"role": "system", "content": context},
                *convo,
                {"role": "user", "content": q},
            ],
            temperature=0.1,
            max_tokens=350,
        )
        return (resp.choices[0].message.content or "").strip() or "No response."
    except Exception as e:
        # Don't silently downgrade if the user configured an API key; return a helpful config error.
        offline = _basic_answer(q, df, profile_dict)
        return (
            "AI request failed.\n"
            f"- Base URL: {api.base_url or '(not set)'}\n"
            f"- Model: {api.model}\n"
            f"- Error: {type(e).__name__}: {e}\n"
            "Fix: for Groq set OPENAI_BASE_URL=https://api.groq.com/openai/v1 and DATAPILOT_MODEL=llama-3.3-70b-versatile.\n\n"
            "Offline answer (limited):\n"
            f"{offline}"
        )
