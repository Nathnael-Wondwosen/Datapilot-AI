from __future__ import annotations

import hashlib
import html as _html
import io
import json
import os
from datetime import datetime, timezone
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from core.chat import answer_question
from core.edit_copilot import EditPlan, apply_edit_plan, plan_edits
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

st.markdown(
    """
    <style>
      /* Sidebar look */
      section[data-testid="stSidebar"] > div {
        /* Professional light slate gradient */
        background: linear-gradient(180deg, rgba(250, 252, 255, 1.0), rgba(241, 245, 249, 1.0));
        border-right: 1px solid rgba(15, 23, 42, 0.08);
      }
      section[data-testid="stSidebar"] .dp-sidehead {
        margin: 6px 2px 10px;
        padding: 12px 12px;
        border-radius: 16px;
        border: 1px solid rgba(15, 23, 42, 0.10);
        background: linear-gradient(135deg, rgba(226, 232, 240, 0.65), rgba(224, 242, 254, 0.55));
      }
      section[data-testid="stSidebar"] .dp-side-title {
        font-size: 18px;
        margin: 0;
        letter-spacing: -0.02em;
      }
      section[data-testid="stSidebar"] .dp-side-sub {
        margin: 6px 0 0;
        opacity: 0.8;
        font-size: 12.5px;
        line-height: 1.4;
      }
      section[data-testid="stSidebar"] hr {
        border: none;
        border-top: 1px solid rgba(15, 23, 42, 0.10);
        margin: 10px 0;
      }
      /* Make widgets feel more "carded" */
      section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
        padding: 10px 10px;
        border-radius: 16px;
        border: 1px solid rgba(15, 23, 42, 0.10);
        background: rgba(255, 255, 255, 0.75);
      }
      section[data-testid="stSidebar"] div[data-testid="stSelectbox"] {
        padding: 10px 10px;
        border-radius: 16px;
        border: 1px solid rgba(15, 23, 42, 0.10);
        background: rgba(255, 255, 255, 0.75);
      }
      section[data-testid="stSidebar"] button {
        border-radius: 14px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _effective_api() -> ApiConfig:
    # Always use env-based config (no runtime overrides).
    return api


def _file_type(name: str) -> str:
    parts = (name or "").lower().rsplit(".", 1)
    return parts[-1] if len(parts) == 2 else ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dataset_id_from_upload(file_name: str, file_bytes: bytes, sheet_name: Optional[str], table_index: Optional[int]) -> str:
    h = hashlib.sha256()
    h.update(file_bytes)
    if sheet_name:
        h.update(str(sheet_name).encode("utf-8", errors="ignore"))
    if table_index is not None:
        h.update(f"table:{int(table_index)}".encode("utf-8"))
    return f"{file_name}:{h.hexdigest()[:16]}"


def _make_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "data")
    return buf.getvalue()


def _clean_df_for_export(df_in: pd.DataFrame, profile) -> pd.DataFrame:
    out = df_in.copy()
    out = out.dropna(axis=1, how="all")

    for c in profile.datetime_columns:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    for c in profile.numeric_columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in profile.categorical_columns:
        if c in out.columns:
            out[c] = out[c].astype("string").str.strip()

    return out


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
    for m in messages[-40:]:
        role = m.get("role")
        content = m.get("content", "")
        cls = "dp-user" if role == "user" else "dp-assistant"
        body = _format_message_html(str(content))
        bubbles.append(f'<div class="dp-bubble {cls}">{body}</div>')
    st.markdown('<div class="dp-chat">' + "".join(bubbles) + "</div>", unsafe_allow_html=True)


def _export_chat_markdown(dataset_name: str, messages: list[dict]) -> str:
    title = f"# DataPilot Chat Export\n\nDataset: `{dataset_name}`\nExported: `{_now_iso()}`\n\n---\n"
    body_lines: list[str] = []
    for m in messages:
        role = m.get("role", "assistant")
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        ts = m.get("ts")
        ts_s = f" ({ts})" if isinstance(ts, str) and ts else ""
        who = "User" if role == "user" else "DataPilot"
        body_lines.append(f"## {who}{ts_s}\n\n{content}\n")
    return title + "\n".join(body_lines)


with st.sidebar:
    st.markdown(
        """
        <div class="dp-sidehead">
          <p class="dp-side-title">DataPilot Control</p>
          <p class="dp-side-sub">Upload a dataset, pick a sheet, then explore insights, chat, and edits.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls", "xlsm"])

use_sample = False
if not up:
    st.markdown(
        """
        <style>
          .dp-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 12px;
            margin-top: 14px;
          }
          @media (min-width: 900px) {
            .dp-grid { grid-template-columns: 1.2fr 0.8fr; }
          }
          .dp-card {
            border-radius: 16px;
            border: 1px solid rgba(49, 51, 63, 0.12);
            background: rgba(255,255,255,0.04);
            padding: 14px 14px;
          }
          .dp-card h3 {
            margin: 0 0 8px;
            font-size: 14px;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            opacity: 0.85;
          }
          .dp-steps {
            margin: 0;
            padding-left: 18px;
            line-height: 1.7;
            opacity: 0.92;
          }
          /* Buttons: crisp CTA */
          div.stButton > button[kind="primary"] {
            border-radius: 14px;
            border: 1px solid rgba(98, 214, 255, 0.35) !important;
            background: linear-gradient(135deg, rgba(98, 214, 255, 0.95), rgba(46, 204, 113, 0.90)) !important;
            color: rgba(12, 18, 32, 0.95) !important;
            font-weight: 650 !important;
            padding: 0.8rem 1rem !important;
          }
          div.stButton > button { border-radius: 14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cta1, cta2, _cta3 = st.columns([1, 1, 2])
    with cta1:
        if st.button("Use sample dataset", type="primary", use_container_width=True):
            use_sample = True
    with cta2:
        st.button("Upload from sidebar", use_container_width=True, disabled=True)

    st.markdown(
        """
        <div class="dp-grid">
          <div class="dp-card">
            <h3>Quick Start</h3>
            <ol class="dp-steps">
              <li>Upload a CSV/XLSX from the left sidebar (or use the sample).</li>
              <li>Explore Overview + Visuals for instant profiling and charts.</li>
              <li>Use Chat to ask questions, then export Report or Chat.</li>
            </ol>
          </div>
          <div class="dp-card">
            <h3>What You Get</h3>
            <ol class="dp-steps">
              <li>Type detection + missing value analysis</li>
              <li>KPIs + top performers</li>
              <li>Forecasting (when time series exists)</li>
              <li>Edit + export cleaned/edited files</li>
            </ol>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not use_sample:
        st.stop()

dataset_name = ""
file_bytes = b""
ft = ""
sheet: Optional[str] = None
table_index: Optional[int] = None

if use_sample:
    dataset_id = "sample_sales"
    dataset_name = "data/samples/sample_sales.csv"
    df_original = pd.read_csv(dataset_name)
else:
    file_bytes = up.getvalue()
    ft = _file_type(up.name)
    if ft in ("xlsx", "xls", "xlsm"):
        try:
            sheets = list_excel_sheets(file_bytes)
            if sheets:
                sheet = st.sidebar.selectbox("Sheet", options=sheets, index=0)
        except Exception:
            st.sidebar.warning("Could not read sheet list; attempting default sheet.")

    # First load to detect multiple tables in a template-like sheet.
    load0 = load_tabular(file_bytes=file_bytes, source_name=up.name, file_type=ft, sheet_name=sheet, table_index=0)
    if getattr(load0, "available_tables", None):
        table_index = st.sidebar.selectbox(
            "Table",
            options=list(range(len(load0.available_tables or []))),
            index=int(load0.table_index or 0),
            format_func=lambda i: (load0.available_tables or [])[i],
        )
        load = load_tabular(file_bytes=file_bytes, source_name=up.name, file_type=ft, sheet_name=sheet, table_index=int(table_index))
    else:
        load = load0

    dataset_name = load.source_name
    df_original = load.df
    table_index = getattr(load, "table_index", None)
    dataset_id = _dataset_id_from_upload(up.name, file_bytes, sheet, table_index)
    if getattr(load, "notes", None):
        with st.expander("Import notes"):
            for n in load.notes or []:
                st.write(f"- {n}")

if df_original is None or df_original.empty:
    st.error("No data found in the uploaded file.")
    st.stop()

df_original = df_original.copy()
df_original.columns = [str(c) for c in df_original.columns]

# Per-dataset edited data store (session only)
store_key = f"dp_df_store::{dataset_id}"
editor_key = f"dp_editor::{dataset_id}"
undo_key = f"dp_undo::{dataset_id}"
plan_key = f"dp_edit_plan::{dataset_id}"
df_saved = st.session_state.get(store_key)
df = df_saved if isinstance(df_saved, pd.DataFrame) else df_original

# Recompute analysis on the effective df (original or edited)
profile = profile_df(df)
profile_dict = profile_as_dict(profile)
quality = compute_quality(df, profile.numeric_columns)
kpis = compute_kpis(df, profile.roles)
visuals = build_visuals(df, profile.roles)
kpis_text = "\n".join([f"{k.name}: {k.value}" + (f" ({k.detail})" if k.detail else "") for k in kpis.kpis])

tab_overview, tab_visuals, tab_quality, tab_forecast, tab_chat, tab_edit, tab_report = st.tabs(
    ["Overview", "Visuals", "Quality", "Forecast", "Chat", "Edit", "Report"]
)

with tab_edit:
    st.subheader("Edit Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("State", "Edited" if isinstance(df_saved, pd.DataFrame) else "Original")

    st.caption(f"Source: `{dataset_name}`")
    if not use_sample and sheet:
        st.caption(f"Sheet: `{sheet}`")

    st.markdown("#### Preview")
    st.dataframe(df.head(200), use_container_width=True)

    st.markdown("#### Edit (session)")
    st.caption("Edits are applied to the in-app dataset and exports. The original uploaded file is not modified on disk.")
    cell_count = int(df.shape[0] * df.shape[1])
    if cell_count > 200_000:
        st.warning("Dataset is large; interactive editing is disabled to keep the app responsive.")
        st.caption("You can still export the current dataset from below.")
        edited_view = df
    else:
        edited_view = st.data_editor(
            df,
            key=editor_key,
            use_container_width=True,
            num_rows="dynamic",
        )

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        if st.button("Save edits", type="primary", use_container_width=True):
            if isinstance(edited_view, pd.DataFrame):
                # Push undo snapshot
                undo = st.session_state.get(undo_key) or []
                if isinstance(undo, list):
                    undo.append(df.copy())
                    st.session_state[undo_key] = undo[-5:]
                st.session_state[store_key] = edited_view.copy()
            st.rerun()
    with b2:
        if st.button("Reset to original", use_container_width=True):
            st.session_state[store_key] = None
            # Also clear the editor widget state for this dataset.
            st.session_state.pop(editor_key, None)
            st.session_state.pop(undo_key, None)
            st.session_state.pop(plan_key, None)
            st.rerun()

    export_df = df if isinstance(df, pd.DataFrame) else df_original
    st.markdown("#### Export")
    exp1, exp2, exp3 = st.columns([1, 1, 2])
    with exp1:
        st.download_button(
            "Download CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="edited.csv" if isinstance(df_saved, pd.DataFrame) else "data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp2:
        st.download_button(
            "Download XLSX",
            data=_make_xlsx_bytes(export_df),
            file_name="edited.xlsx" if isinstance(df_saved, pd.DataFrame) else "data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.divider()
    st.markdown("#### Edit Copilot (AI)")
    st.caption("Describe changes in plain English. DataPilot will propose a safe edit plan you can review before applying.")

    examples = [
        "Add a column Profit = revenue - cost",
        "Remove the columns Notes and InternalID",
        "Rename column 'rev' to 'revenue'",
        "Fill missing values in revenue with the median",
        "Convert the date column to datetime",
        "Deduplicate rows based on OrderID keep last",
    ]
    ex_cols = st.columns(3)
    for i, ex in enumerate(examples[:3]):
        if ex_cols[i].button(ex, use_container_width=True):
            st.session_state["dp_edit_instruction"] = ex
    ex_cols2 = st.columns(3)
    for i, ex in enumerate(examples[3:6]):
        if ex_cols2[i].button(ex, use_container_width=True):
            st.session_state["dp_edit_instruction"] = ex

    instruction = st.text_area(
        "What do you want to change?",
        key="dp_edit_instruction",
        height=80,
        placeholder="e.g. Add a column Profit = revenue - cost",
    )

    p1, p2, p3 = st.columns([1, 1, 2])
    with p1:
        plan_btn = st.button("Plan changes", type="primary", use_container_width=True)
    with p2:
        undo_btn = st.button("Undo", use_container_width=True, disabled=not bool(st.session_state.get(undo_key)))

    if undo_btn:
        undo = st.session_state.get(undo_key) or []
        if isinstance(undo, list) and undo:
            prev = undo.pop()
            st.session_state[undo_key] = undo
            st.session_state[store_key] = prev
            st.session_state.pop(editor_key, None)
            st.session_state.pop(plan_key, None)
            st.rerun()

    if plan_btn:
        if not instruction.strip():
            st.warning("Write an edit instruction first.")
        else:
            try:
                with st.spinner("Planning edits…"):
                    plan = plan_edits(api=_effective_api(), instruction=instruction, df=df)
                st.session_state[plan_key] = plan
            except Exception as e:
                st.error(f"Could not plan edits: {type(e).__name__}: {e}")

    plan_obj = st.session_state.get(plan_key)
    if isinstance(plan_obj, EditPlan):
        st.markdown("**Proposed plan**")
        st.json({"ops": plan_obj.ops, "notes": plan_obj.notes})

        apply_btn = st.button("Apply plan", use_container_width=True)
        if apply_btn:
            try:
                with st.spinner("Applying edits…"):
                    new_df, log = apply_edit_plan(df, plan_obj)
                # Push undo snapshot
                undo = st.session_state.get(undo_key) or []
                if isinstance(undo, list):
                    undo.append(df.copy())
                    st.session_state[undo_key] = undo[-5:]
                st.session_state[store_key] = new_df
                st.session_state.pop(editor_key, None)
                st.session_state.pop(plan_key, None)
                st.success("Edits applied.")
                if log:
                    st.caption("Applied:")
                    st.write("\n".join([f"- {x}" for x in log]))
                st.rerun()
            except Exception as e:
                st.error(f"Could not apply plan: {type(e).__name__}: {e}")

with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{profile.rows:,}")
    c2.metric("Columns", f"{profile.cols:,}")
    c3.metric("Numeric columns", f"{len(profile.numeric_columns):,}")

    clean_df = _clean_df_for_export(df, profile)
    st.download_button(
        "Download cleaned CSV",
        data=clean_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned.csv",
        mime="text/csv",
        help="Coerces numeric/date columns, trims text whitespace, and drops fully-empty columns.",
    )

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
    st.markdown(
        """
        <style>
          .dp-chat { display: flex; flex-direction: column; gap: 10px; padding: 6px 4px; }
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
          .dp-bubble p { margin: 0; }
          .dp-bubble p + p { margin-top: 8px; }
          .dp-bubble ul { margin: 8px 0 0 18px; }
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
          .dp-assistant { align-self: flex-start; background: rgba(49, 51, 63, 0.04); }
          .dp-user {
            align-self: flex-end;
            margin-left: auto;
            background: rgba(46, 204, 113, 0.18);
            border-color: rgba(46, 204, 113, 0.35);
          }
          div[data-testid="stChatInput"] {
            border-top: 1px solid rgba(49, 51, 63, 0.12);
            padding-top: 10px;
            padding-bottom: 0px !important;
            margin-bottom: 0px !important;
          }
          div[data-testid="stChatInput"] textarea {
            border-radius: 14px !important;
            border: 1px solid rgba(49, 51, 63, 0.18) !important;
            box-shadow: 0 6px 24px rgba(0,0,0,0.06);
            padding: 12px 14px !important;
          }
          div[data-testid="stChatInput"] + div { margin-top: 0px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Reset chat when dataset changes (per-file memory).
    if st.session_state.get("dp_dataset_id") != dataset_id:
        st.session_state.dp_dataset_id = dataset_id
        st.session_state.pop("chat_messages", None)
        st.session_state.pop("dp_stage", None)
        st.session_state.pop("dp_prompt", None)

    left, right = st.columns([3, 1])
    with left:
        st.markdown("Chat")
        st.caption("Answers are generated strictly from your uploaded dataset.")
    with right:
        if st.button("Reset chat", use_container_width=True):
            st.session_state.pop("chat_messages", None)
            st.session_state.pop("dp_stage", None)
            st.session_state.pop("dp_prompt", None)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "ts": _now_iso(),
                "content": "Ask me about trends, outliers, missing values, top performers, or a plain-English summary.",
            }
        ]

    if "dp_stage" not in st.session_state:
        st.session_state.dp_stage = None  # None | "render" | "compute"
    if "dp_prompt" not in st.session_state:
        st.session_state.dp_prompt = None

    exp1, exp2, _sp = st.columns([1, 1, 2])
    with exp1:
        st.download_button(
            "Export chat.md",
            data=_export_chat_markdown(dataset_name, st.session_state.chat_messages).encode("utf-8"),
            file_name="datapilot_chat.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with exp2:
        st.download_button(
            "Export chat.json",
            data=json.dumps(st.session_state.chat_messages, ensure_ascii=True, indent=2).encode("utf-8"),
            file_name="datapilot_chat.json",
            mime="application/json",
            use_container_width=True,
        )

    sug_cols = st.columns(3)
    suggestions = [
        "Summarize this dataset in 5 bullet points.",
        "Which columns have the most missing values?",
        "What are the top performers and why?",
    ]
    for i, s in enumerate(suggestions):
        if sug_cols[i].button(s, use_container_width=True):
            st.session_state.dp_prompt = s
            st.session_state.dp_stage = "render"
            st.rerun()

    st.divider()
    _render_chat(st.session_state.chat_messages)

    if st.session_state.dp_stage == "render" and st.session_state.dp_prompt:
        p = st.session_state.dp_prompt
        st.session_state.chat_messages.append({"role": "user", "ts": _now_iso(), "content": p})
        st.session_state.chat_messages.append({"role": "assistant", "ts": _now_iso(), "content": "Typing..."})
        st.session_state.dp_stage = "compute"
        st.rerun()

    if st.session_state.dp_stage == "compute" and st.session_state.dp_prompt:
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
        st.session_state.chat_messages[-1]["ts"] = _now_iso()
        st.session_state.dp_prompt = None
        st.session_state.dp_stage = None
        st.rerun()

    prompt = st.chat_input("Message DataPilot...")
    if prompt:
        st.session_state.dp_prompt = prompt
        st.session_state.dp_stage = "render"
        st.rerun()

with tab_report:
    st.markdown("Generate a report you can download and share.")
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
        dataset_name=dataset_name,
        profile_dict=profile_dict,
        kpis_text=kpis_text,
        insights=base_ins,
        quality_notes=quality_notes,
    )
    html = build_html_report(
        title=report_title,
        dataset_name=dataset_name,
        profile_dict=profile_dict,
        kpis_text=kpis_text,
        insights=base_ins,
        quality_notes=quality_notes,
    )

    b1, b2 = st.columns(2)
    with b1:
        st.download_button("Download report.md", data=md.encode("utf-8"), file_name="report.md", mime="text/markdown", use_container_width=True)
    with b2:
        st.download_button("Download report.html", data=html.encode("utf-8"), file_name="report.html", mime="text/html", use_container_width=True)

    with st.expander("Preview (Markdown)"):
        st.markdown(md)
