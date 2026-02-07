from __future__ import annotations

from typing import List


def _html_escape(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


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


def build_html_report(
    title: str,
    dataset_name: str,
    profile_dict: dict,
    kpis_text: str,
    insights: List[str],
    quality_notes: List[str],
) -> str:
    title_e = _html_escape(title)
    dataset_e = _html_escape(dataset_name)
    rows = _html_escape(str(profile_dict.get("rows")))
    cols = _html_escape(str(profile_dict.get("cols")))

    ins_li = "\n".join([f"<li>{_html_escape(x)}</li>" for x in insights])
    q_li = "\n".join([f"<li>{_html_escape(x)}</li>" for x in quality_notes])
    kpis_pre = _html_escape(kpis_text.strip())

    quality_section = ""
    if quality_notes:
        quality_section = f"""
        <section class="card">
          <h2>Data Quality Notes</h2>
          <ul>
            {q_li}
          </ul>
        </section>
        """

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title_e}</title>
    <style>
      :root {{
        --bg: #0b1220;
        --card: rgba(255,255,255,0.06);
        --text: #e8eefc;
        --muted: rgba(232,238,252,0.75);
        --line: rgba(232,238,252,0.18);
        --accent: #62d6ff;
        --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      }}
      body {{
        margin: 0;
        font-family: var(--sans);
        background: radial-gradient(1200px 700px at 10% 0%, rgba(98,214,255,0.20), transparent 55%),
                    radial-gradient(900px 600px at 90% 10%, rgba(98,214,255,0.12), transparent 60%),
                    var(--bg);
        color: var(--text);
      }}
      .wrap {{
        max-width: 980px;
        margin: 0 auto;
        padding: 28px 16px 40px;
      }}
      header {{
        padding: 18px 18px 0;
      }}
      h1 {{
        margin: 0 0 6px;
        letter-spacing: -0.02em;
        font-size: 28px;
      }}
      .meta {{
        margin: 0;
        color: var(--muted);
        font-size: 14px;
      }}
      .grid {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 12px;
        margin-top: 14px;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px 16px;
        backdrop-filter: blur(8px);
      }}
      h2 {{
        margin: 0 0 10px;
        font-size: 16px;
        color: var(--text);
      }}
      ul {{
        margin: 8px 0 0 18px;
      }}
      code, pre {{
        font-family: var(--mono);
      }}
      pre {{
        background: rgba(0,0,0,0.35);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 12px;
        overflow: auto;
        margin: 10px 0 0;
      }}
      .kpi {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }}
      .pill {{
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 10px 12px;
        color: var(--muted);
        display: flex;
        justify-content: space-between;
        gap: 10px;
      }}
      .pill b {{
        color: var(--text);
        font-weight: 600;
      }}
      .footer {{
        color: var(--muted);
        font-size: 12px;
        margin-top: 12px;
      }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <header class="card">
        <h1>{title_e}</h1>
        <p class="meta">Dataset: <code>{dataset_e}</code></p>
      </header>

      <div class="grid">
        <section class="card">
          <h2>Executive Summary</h2>
          <ul>
            {ins_li}
          </ul>
        </section>

        <section class="card">
          <h2>Dataset Profile</h2>
          <div class="kpi">
            <div class="pill"><span>Rows</span><b>{rows}</b></div>
            <div class="pill"><span>Columns</span><b>{cols}</b></div>
          </div>
        </section>

        <section class="card">
          <h2>KPIs</h2>
          <pre>{kpis_pre}</pre>
        </section>

        {quality_section}
      </div>

      <p class="footer">Generated by DataPilot AI.</p>
    </div>
  </body>
</html>"""
