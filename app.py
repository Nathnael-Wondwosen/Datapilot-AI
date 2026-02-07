from __future__ import annotations

import os
import html as _html
from typing import Optional

import pandas as pd
import streamlit as st

from core.chat import answer_question
from core.forecasting import forecast_figure, forecast_time_series
from core.insights import llm_insights, rule_based_insights
from core.kpis import compute_kpis
from core.loader import list_excel_sheets, load_tabular
from core.profiler import profile_as_dict, profile_df
from core.quality import compute_quality
from core.report import build_html_report, build_markdown_report
from core.visualizer import build_visuals
from utils.helpers import ApiConfig, load_api_config


st.set_page_config(page_title="DataPilot AI", layout="wide")

st.title("DataPilot AI")
st.caption("Upload any CSV or Excel file. Get instant insights, visual analytics, and AI-powered explanations.")

api = load_api_config()

with st.sidebar:
    st.header("Upload")
    up = st.file_uploader("CSV or Excel (.csv, .xlsx)", type=["csv", "xlsx", "xls", "xlsm"])

    # AI config UI is intentionally hidden; no sidebar status text.


def _file_type(name: str) -> str:
    parts = (name or "").lower().rsplit(".", 1)
    return parts[-1] if len(parts) == 2 else ""


def _effective_api() -> ApiConfig:
    # Always use env-based config (no runtime overrides) as requested.
    return api


use_sample = False
if not up:
    st.markdown("### Quick start")
    st.markdown("Upload a file, or load the included sample dataset.")
    if st.button("Use sample dataset", type="primary"):
        use_sample = True
    else:
        st.stop()

if use_sample:
    df = pd.read_csv("data/samples/sample_sales.csv")
    load = type("Load", (), {"source_name": "data/samples/sample_sales.csv"})()
    ft = "csv"
    file_bytes = b""
else:
    file_bytes = up.getvalue()
    ft = _file_type(up.name)

if not use_sample:
    sheet: Optional[str] = None
    if ft in ("xlsx", "xls", "xlsm"):
        try:
            sheets = list_excel_sheets(file_bytes)
            if sheets:
                sheet = st.sidebar.selectbox("Sheet", options=sheets, index=0)
        except Exception:
            st.sidebar.warning("Could not read sheet list; attempting default sheet.")

    load = load_tabular(file_bytes=file_bytes, source_name=up.name, file_type=ft, sheet_name=sheet)
    df = load.df

if df is None or df.empty:
    st.error("No data found in the uploaded file.")
    st.stop()

df.columns = [str(c) for c in df.columns]

st.subheader("Preview")
st.dataframe(df.head(200), use_container_width=True)

profile = profile_df(df)
profile_dict = profile_as_dict(profile)
quality = compute_quality(df, profile.numeric_columns)
kpis = compute_kpis(df, profile.roles)
visuals = build_visuals(df, profile.roles)

kpis_text = "\n".join([f"{k.name}: {k.value}" + (f" ({k.detail})" if k.detail else "") for k in kpis.kpis])

tab_overview, tab_visuals, tab_quality, tab_forecast, tab_chat, tab_report = st.tabs(
    ["Overview", "Visuals", "Quality", "Forecast", "Chat", "Report"]
)

with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{profile.rows:,}")
    c2.metric("Columns", f"{profile.cols:,}")
    c3.metric("Numeric columns", f"{len(profile.numeric_columns):,}")

    st.markdown("#### Column profiling")
    prof_df = pd.DataFrame(
        [
            {
                "column": c.name,
                "role": c.role,
                "dtype": c.dtype,
                "null_pct": c.null_pct,
                "uniques": c.uniques,
                "examples": ", ".join(c.examples),
            }
            for c in profile.columns
        ]
    ).sort_values(["role", "null_pct"], ascending=[True, False])
    st.dataframe(prof_df, use_container_width=True)

    if profile.numeric_summary is not None:
        st.markdown("#### Numeric summary")
        st.dataframe(profile.numeric_summary, use_container_width=True)

    st.markdown("#### KPIs")
    kpi_cols = st.columns(3)
    for i, k in enumerate(kpis.kpis[:12]):
        kpi_cols[i % 3].metric(k.name, k.value, help=k.detail)

    for title, tdf in kpis.top_tables.items():
        st.markdown(f"#### {title}")
        st.dataframe(tdf, use_container_width=True)

    st.markdown("#### Insights")
    base_ins = rule_based_insights(profile_dict, kpis.kpis, quality)
    for s in base_ins:
        st.write(f"- {s}")

    ai_text = llm_insights(api=_effective_api(), profile_dict=profile_dict, kpis_text=kpis_text)
    if ai_text:
        st.markdown("**AI insights**")
        st.write(ai_text)

with tab_visuals:
    if not visuals:
        st.info("No visuals could be generated for this dataset.")
    for v in visuals:
        st.markdown(f"#### {v.title}")
        st.plotly_chart(v.fig, use_container_width=True)

with tab_quality:
    st.markdown("#### Missing values")
    st.dataframe(quality.missing_by_column, use_container_width=True)

    st.markdown("#### Outliers (z-score)")
    if not quality.outliers:
        st.write("No strong outliers detected (or not enough numeric data).")
    else:
        st.dataframe(
            pd.DataFrame([{"column": o.column, "method": o.method, "count": o.count, "pct": o.pct} for o in quality.outliers]),
            use_container_width=True,
        )

with tab_forecast:
    st.markdown("Forecasting works when you have a date/time column and a numeric metric.")
    dt_cols = profile.datetime_columns
    num_cols = profile.numeric_columns
    if not dt_cols or not num_cols:
        st.info("No datetime and/or numeric columns detected.")
    else:
        date_col = st.selectbox("Date column", options=dt_cols, index=0)
        target_col = st.selectbox("Target metric", options=num_cols, index=0)
        horizon = st.slider("Horizon", min_value=7, max_value=180, value=30, step=1)
        res = forecast_time_series(df, date_col=date_col, target_col=target_col, horizon=horizon)
        if not res:
            st.warning("Not enough clean time series data to forecast (need at least ~10 points).")
        else:
            st.caption(f"Model: {res.model_name}")
            st.plotly_chart(forecast_figure(res, title=f"Forecast: {target_col}"), use_container_width=True)
            st.dataframe(res.forecast.head(50), use_container_width=True)

with tab_chat:
    left, right = st.columns([3, 1])
    with left:
        st.markdown("Ask questions about your dataset.")
    with right:
        if st.button("Reset chat", use_container_width=True):
            st.session_state.pop("chat_messages", None)
            st.session_state.pop("dp_stage", None)
            st.session_state.pop("dp_prompt", None)

    st.markdown(
        """
        <style>
          /* Chat bubbles (WhatsApp/Telegram-ish) */
          .dp-chat {
            display: flex;
            flex-direction: column;
            gap: 10px;
            padding: 6px 4px;
          }
          .dp-bubble {
            max-width: 78%;
            padding: 10px 12px;
            border-radius: 16px;
            line-height: 1.35;
            border: 1px solid rgba(49, 51, 63, 0.12);
            box-shadow: 0 1px 0 rgba(0,0,0,0.03);
            word-wrap: break-word;
            overflow-wrap: anywhere;
          }
          .dp-bubble p {
            margin: 0;
          }
          .dp-bubble p + p {
            margin-top: 8px;
          }
          .dp-bubble ul {
            margin: 8px 0 0 18px;
          }
          .dp-bubble pre {
            margin: 8px 0 0;
            padding: 10px;
            border-radius: 12px;
            background: rgba(0,0,0,0.06);
            border: 1px solid rgba(49, 51, 63, 0.12);
            overflow: auto;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 0.9em;
          }
          .dp-assistant {
            align-self: flex-start;
            background: rgba(49, 51, 63, 0.04);
          }
          .dp-user {
            align-self: flex-end;
            margin-left: auto;
            background: rgba(46, 204, 113, 0.18);
            border-color: rgba(46, 204, 113, 0.35);
          }

          /* Status banner (above input) */
          .dp-status {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border-radius: 12px;
            border: 1px solid rgba(49, 51, 63, 0.12);
            background: linear-gradient(90deg, rgba(98, 214, 255, 0.18), rgba(46, 204, 113, 0.12));
            color: rgba(49, 51, 63, 0.85);
          }
          .dp-dot {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: rgba(49, 51, 63, 0.55);
            animation: dp-bounce 1.1s infinite ease-in-out;
          }
          .dp-dot:nth-child(2) { animation-delay: 0.15s; opacity: 0.85; }
          .dp-dot:nth-child(3) { animation-delay: 0.30s; opacity: 0.70; }
          @keyframes dp-bounce {
            0%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-5px); }
          }

          /* Make chat input look more like a messenger bar */
          div[data-testid="stChatInput"] {
            border-top: 1px solid rgba(49, 51, 63, 0.12);
            padding-top: 10px;
          }
          div[data-testid="stChatInput"] textarea {
            border-radius: 14px !important;
            border: 1px solid rgba(49, 51, 63, 0.18) !important;
            box-shadow: 0 6px 24px rgba(0,0,0,0.06);
            padding: 12px 14px !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Ask me about trends, outliers, missing values, top performers, or a plain-English summary. "
                "I will answer strictly from your uploaded data.",
            }
        ]

    if "dp_stage" not in st.session_state:
        st.session_state.dp_stage = None  # None | "render" | "compute"
    if "dp_prompt" not in st.session_state:
        st.session_state.dp_prompt = None

    def _format_message_html(text: str) -> str:
        """
        Minimal markdown-to-HTML for chat bubbles (safe).
        - escapes HTML
        - supports bullet lists and fenced code blocks
        """
        s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        out: list[str] = []

        in_code = False
        code_lines: list[str] = []
        buf: list[str] = []
        ul_open = False

        def flush_paragraph() -> None:
            nonlocal buf
            if not buf:
                return
            p = _html.escape("\n".join(buf))
            p = p.replace("\n", "<br/>")
            out.append(f"<p>{p}</p>")
            buf = []

        def open_ul() -> None:
            nonlocal ul_open
            if not ul_open:
                flush_paragraph()
                out.append("<ul>")
                ul_open = True

        def close_ul() -> None:
            nonlocal ul_open
            if ul_open:
                out.append("</ul>")
                ul_open = False

        for line in s.split("\n"):
            if line.strip().startswith("```"):
                if not in_code:
                    close_ul()
                    flush_paragraph()
                    in_code = True
                    code_lines = []
                else:
                    code = _html.escape("\n".join(code_lines))
                    out.append(f"<pre><code>{code}</code></pre>")
                    in_code = False
                continue

            if in_code:
                code_lines.append(line)
                continue

            bullet = line.lstrip().startswith(("- ", "* "))
            if bullet:
                open_ul()
                item = line.lstrip()[2:]
                out.append(f"<li>{_html.escape(item)}</li>")
                continue

            if line.strip() == "":
                close_ul()
                flush_paragraph()
                continue

            close_ul()
            buf.append(line)

        close_ul()
        flush_paragraph()
        return "".join(out) if out else "<p></p>"

    def _render_chat(messages: list[dict]) -> None:
        bubbles: list[str] = []
        for m in messages[-30:]:
            role = m.get("role")
            content = m.get("content", "")
            cls = "dp-user" if role == "user" else "dp-assistant"
            body = _format_message_html(str(content))
            bubbles.append(f'<div class="dp-bubble {cls}">{body}</div>')
        st.markdown('<div class="dp-chat">' + "".join(bubbles) + "</div>", unsafe_allow_html=True)

    def _set_pending_prompt(p: str) -> None:
        p = (p or "").strip()
        if not p:
            return
        st.session_state.dp_prompt = p
        st.session_state.dp_stage = "render"
        st.rerun()

    # Suggested prompts keep the UX professional and reduce user friction.
    sug_cols = st.columns(3)
    suggestions = [
        "Summarize this dataset in 5 bullet points.",
        "Which columns have the most missing values?",
        "What are the top performers and why?",
    ]
    for i, s in enumerate(suggestions):
        if sug_cols[i].button(s, use_container_width=True):
            _set_pending_prompt(s)

    st.divider()

    _render_chat(st.session_state.chat_messages)

    # Two-phase send to keep the input pinned and show a nicer status above it.
    # Phase 1 (render): append user message + typing bubble, then rerun so UI updates before compute.
    if st.session_state.dp_stage == "render" and st.session_state.dp_prompt:
        p = st.session_state.dp_prompt
        st.session_state.chat_messages.append({"role": "user", "content": p})
        st.session_state.chat_messages.append({"role": "assistant", "content": "Typing…"})
        st.session_state.dp_stage = "compute"
        st.rerun()

    status_slot = st.empty()
    if st.session_state.dp_stage == "compute" and st.session_state.dp_prompt:
        status_slot.markdown(
            """
            <div class="dp-status">
              <span>Analyzing</span>
              <span class="dp-dot"></span><span class="dp-dot"></span><span class="dp-dot"></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        p = st.session_state.dp_prompt
        ans = answer_question(
            api=_effective_api(),
            question=p,
            df=df,
            profile_dict=profile_dict,
            kpis_text=kpis_text,
            history=st.session_state.chat_messages,
        )
        st.session_state.chat_messages[-1]["content"] = ans
        st.session_state.dp_prompt = None
        st.session_state.dp_stage = None
        st.rerun()

    # Input at the bottom
    prompt = st.chat_input("Message DataPilot…")
    if prompt:
        _set_pending_prompt(prompt)

with tab_report:
    st.markdown("Generate a simple Markdown report you can download and share.")
    report_title = st.text_input("Report title", value="DataPilot AI Report")

    base_ins = rule_based_insights(profile_dict, kpis.kpis, quality)
    quality_notes = []
    if not quality.missing_by_column.empty:
        worst = quality.missing_by_column.iloc[0]
        if float(worst["missing_pct"]) > 0:
            quality_notes.append(f"Highest missingness: {worst['column']} ({float(worst['missing_pct'])*100:.1f}%).")
    if quality.outliers:
        o = quality.outliers[0]
        quality_notes.append(f"Outliers: {o.column} ({o.count} rows, {o.pct*100:.1f}% by z-score).")

    md = build_markdown_report(
        title=report_title,
        dataset_name=load.source_name,
        profile_dict=profile_dict,
        kpis_text=kpis_text,
        insights=base_ins,
        quality_notes=quality_notes,
    )

    st.download_button("Download report.md", data=md.encode("utf-8"), file_name="report.md", mime="text/markdown")
    html = build_html_report(
        title=report_title,
        dataset_name=load.source_name,
        profile_dict=profile_dict,
        kpis_text=kpis_text,
        insights=base_ins,
        quality_notes=quality_notes,
    )
    st.download_button("Download report.html", data=html.encode("utf-8"), file_name="report.html", mime="text/html")
    with st.expander("Preview"):
        st.markdown(md)
