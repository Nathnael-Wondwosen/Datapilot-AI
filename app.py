from __future__ import annotations

import os
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
from core.report import build_markdown_report
from core.visualizer import build_visuals
from utils.helpers import ApiConfig, load_api_config


st.set_page_config(page_title="DataPilot AI", layout="wide")

st.title("DataPilot AI")
st.caption("Upload any CSV or Excel file. Get instant insights, visual analytics, and AI-powered explanations.")

api = load_api_config()

with st.sidebar:
    st.header("Upload")
    up = st.file_uploader("CSV or Excel (.csv, .xlsx)", type=["csv", "xlsx", "xls", "xlsm"])

    # Minimal status (AI config UI is intentionally hidden).
    dotenv_path = os.environ.get("DATAPILOT_DOTENV_PATH")
    if dotenv_path:
        st.caption(f"Env: loaded ({dotenv_path})")
    else:
        st.caption("Env: not loaded (.env not found)")

    if api.api_key:
        if api.base_url and "groq.com" in api.base_url.lower():
            st.caption("AI: enabled (Groq)")
        elif api.base_url:
            st.caption("AI: enabled (OpenAI-compatible)")
        else:
            st.caption("AI: enabled (base URL not set)")
    else:
        st.caption("AI: disabled (set .env to enable)")


def _file_type(name: str) -> str:
    parts = (name or "").lower().rsplit(".", 1)
    return parts[-1] if len(parts) == 2 else ""


def _effective_api() -> ApiConfig:
    # Always use env-based config (no runtime overrides) as requested.
    return api


if not up:
    st.markdown("### Quick start")
    st.markdown("Upload a file, or use the included sample at `data/samples/sample_sales.csv`.")
    st.stop()

file_bytes = up.getvalue()
ft = _file_type(up.name)

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

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Ask me about trends, outliers, missing values, top performers, or a plain-English summary. "
                "I will answer strictly from your uploaded data.",
            }
        ]

    # Suggested prompts keep the UX professional and reduce user friction.
    sug_cols = st.columns(3)
    suggestions = [
        "Summarize this dataset in 5 bullet points.",
        "Which columns have the most missing values?",
        "What are the top performers and why?",
    ]
    for i, s in enumerate(suggestions):
        if sug_cols[i].button(s, use_container_width=True):
            st.session_state.chat_messages.append({"role": "user", "content": s})

    st.divider()

    # Render messages
    for m in st.session_state.chat_messages[-20:]:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # Input at the bottom
    prompt = st.chat_input("Message DataPilot…")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                ans = answer_question(
                    api=_effective_api(),
                    question=prompt,
                    df=df,
                    profile_dict=profile_dict,
                    kpis_text=kpis_text,
                    history=st.session_state.chat_messages,
                )
            st.write(ans)
        st.session_state.chat_messages.append({"role": "assistant", "content": ans})

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
    with st.expander("Preview"):
        st.markdown(md)
