from __future__ import annotations

from typing import List


def build_markdown_report(
    title: str,
    dataset_name: str,
    profile_dict: dict,
    kpis_text: str,
    insights: List[str],
    quality_notes: List[str],
) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Dataset: `{dataset_name}`")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    for s in insights:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## Dataset Profile")
    lines.append("")
    lines.append(f"- Rows: {profile_dict.get('rows')}")
    lines.append(f"- Columns: {profile_dict.get('cols')}")
    lines.append("")
    lines.append("## KPIs")
    lines.append("")
    lines.append("```")
    lines.append(kpis_text.strip())
    lines.append("```")
    if quality_notes:
        lines.append("")
        lines.append("## Data Quality Notes")
        lines.append("")
        for n in quality_notes:
            lines.append(f"- {n}")
    lines.append("")
    return "\n".join(lines)

