from __future__ import annotations

from typing import List, Optional

from core.quality import QualityReport
from utils.helpers import ApiConfig


def rule_based_insights(profile_dict: dict, kpis: list, quality: QualityReport) -> List[str]:
    rows = int(profile_dict.get("rows", 0))
    cols = int(profile_dict.get("cols", 0))
    roles = profile_dict.get("roles", {}) or {}
    numeric_cols = [c for c, r in roles.items() if r == "numeric"]
    dt_cols = [c for c, r in roles.items() if r == "datetime"]
    cat_cols = [c for c, r in roles.items() if r == "categorical"]

    out: List[str] = []
    out.append(f"Dataset has {rows:,} rows and {cols:,} columns.")
    out.append(f"Detected {len(numeric_cols)} numeric, {len(dt_cols)} date/time, and {len(cat_cols)} categorical columns.")

    if not quality.missing_by_column.empty:
        worst = quality.missing_by_column.iloc[0]
        if float(worst["missing_pct"]) >= 0.2:
            out.append(
                f"Data quality warning: column '{worst['column']}' is missing {int(worst['missing']):,} values ({float(worst['missing_pct'])*100:.1f}%)."
            )

    if quality.outliers:
        o = quality.outliers[0]
        out.append(f"Outliers detected in '{o.column}' ({o.count:,} rows, {o.pct*100:.1f}% by z-score).")

    if kpis:
        out.append(f"Computed {len(kpis)} quick KPIs (totals/averages and basic changes where possible).")

    return out


def llm_insights(
    api: ApiConfig,
    profile_dict: dict,
    kpis_text: str,
    max_points: int = 8,
) -> Optional[str]:
    if not api.api_key:
        return None

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    client = OpenAI(api_key=api.api_key, base_url=api.base_url) if api.base_url else OpenAI(api_key=api.api_key)

    roles = profile_dict.get("roles", {}) or {}
    numeric_cols = [c for c, r in roles.items() if r == "numeric"][:10]
    dt_cols = [c for c, r in roles.items() if r == "datetime"][:5]
    cat_cols = [c for c, r in roles.items() if r == "categorical"][:10]

    prompt = (
        "You are an analytics assistant. Generate a short, concrete executive summary strictly from the provided dataset profile and KPIs. "
        "Do not invent data. If something cannot be determined, say so.\n\n"
        f"Rows: {profile_dict.get('rows')}\n"
        f"Cols: {profile_dict.get('cols')}\n"
        f"Numeric columns (sample): {numeric_cols}\n"
        f"Datetime columns: {dt_cols}\n"
        f"Categorical columns (sample): {cat_cols}\n\n"
        f"KPIs:\n{kpis_text}\n\n"
        f"Output: {max_points} bullet points max."
    )

    try:
        resp = client.chat.completions.create(
            model=api.model,
            messages=[
                {"role": "system", "content": "You produce concise, defensible analytics narratives."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        return None
