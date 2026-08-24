from pathlib import Path
import io
import os
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from src.rag_engine import build_document_store, run_classification

load_dotenv()
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "raw_data.xlsx"

st.set_page_config(page_title="DAQO Congressional RAG", page_icon="🏛️", layout="wide")
st.title("🏛️ DAQO Congressional Risk & Opportunity RAG")
st.caption("Qualitative Congressional content analysis for DAQO's China-export, Mexico-manufacture and U.S.-manufacture strategies.")

@st.cache_data
def load_excel():
    if not DATA_PATH.exists():
        return None
    d = pd.read_excel(DATA_PATH)
    d.columns = [str(c).strip() for c in d.columns]
    return d

df = load_excel()
if df is None:
    st.error("Missing data/raw_data.xlsx")
    st.stop()

def find_col(*names):
    norm = {c.lower().replace(" ","").replace("-","").replace("_",""): c for c in df.columns}
    for n in names:
        k = n.lower().replace(" ","").replace("-","").replace("_","")
        if k in norm:
            return norm[k]
    for c in df.columns:
        if any(n.lower() in c.lower() for n in names):
            return c
    return None

cols = {
    "bill_col": find_col("Bill Number", "Bill"),
    "title_col": find_col("Title"),
    "status_col": find_col("Status"),
    "country_col": find_col("Country"),
    "type_col": find_col("Bill Type", "Type"),
    "govtrack_col": find_col("GOVTRACK URL", "GovTrack"),
    "congress_session_col": find_col("Congress"),
}

if not cols["govtrack_col"]:
    st.error('The workbook must contain the "GOVTRACK URL" column.')
    st.stop()

st.info('📌 Evidence source: the app uses only the "GOVTRACK URL" column. The Secondary Source / Congress.gov column is ignored.')
st.subheader("📊 Dataset overview")
c1, c2 = st.columns(2)
c1.metric("Congressional actions", len(df))
c2.metric("Countries", df[cols["country_col"]].nunique() if cols["country_col"] else "—")

with st.expander("View source data"):
    st.dataframe(df, use_container_width=True)

for k, v in {"docs": [], "summary": None, "log": [], "results": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.divider()
st.subheader("📚 Step 1 — Retrieve Congressional evidence")
n = st.number_input("Actions to scan", 1, max(1, len(df)), min(10, len(df)), 1)

if st.button("🌐 Build Evidence Base", type="primary"):
    p = st.progress(0)
    msg = st.empty()
    def cb(done, total, text):
        p.progress(min(done / max(total, 1), 1.0))
        msg.write(text)

    docs, s, l = build_document_store(df.head(int(n)), cols, cb)
    st.session_state.docs = docs
    st.session_state.summary = s
    st.session_state.log = l
    st.session_state.results = []
    p.progress(1.0)
    msg.empty()
    st.success(f"{len(docs)} actions loaded from GovTrack.")

if st.session_state.summary:
    s = st.session_state.summary
    a,b,c,d = st.columns(4)
    a.metric("Bills scanned", s["bills_scanned"])
    b.metric("GovTrack links checked", s["links_checked"])
    c.metric("Sources loaded", s["sources_loaded"])
    d.metric("Errors", s["source_errors"])
    with st.expander("🔎 Source diagnostics"):
        st.dataframe(pd.DataFrame(st.session_state.log), use_container_width=True, hide_index=True)

st.divider()
st.subheader("🎯 Step 2 — Qualitative relevance & classification")
st.markdown("**First:** RELEVANT = YES/NO. **Then, if relevant:** exactly one of Country Risk/Opportunity, Industry Risk/Opportunity, or Investment Risk/Opportunity.")
st.markdown("Scenarios: 🇨🇳 **CHINA-EXPORT** · 🇲🇽 **MEXICO-MANUFACTURE** · 🇺🇸 **US-MANUFACTURE**")

key = os.getenv("OPENAI_API_KEY", "").strip()
if not key:
    try:
        key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        key = ""

if st.button("🧠 Run Qualitative Classification"):
    if not st.session_state.docs:
        st.error("Run Step 1 first.")
    elif not key:
        st.error("The AI analysis service is not configured. Please contact the app administrator.")
    else:
        p = st.progress(0)
        msg = st.empty()
        def cb2(done, total, text):
            p.progress(min(done / max(total, 1), 1.0))
            msg.write(text)
        st.session_state.results = run_classification(st.session_state.docs, key, cb2)
        p.progress(1.0)
        msg.empty()
        st.success("Classification complete.")


def build_excel_export(results):
    all_rows, evidence_rows, failed_rows = [], [], []

    for r in results:
        status = r.get("analysis_status", "SUCCESS")
        scenarios = "; ".join(r.get("affected_scenarios", []))
        source_urls = "; ".join(r.get("source_urls", []))

        all_rows.append({
            "Analysis Status": status,
            "Bill": r.get("bill_number", ""),
            "Title": r.get("title", ""),
            "Country": r.get("country", ""),
            "Relevant": r.get("relevant", ""),
            "Primary Classification": r.get("primary_classification", ""),
            "Policy Stage": r.get("policy_stage", ""),
            "Directness": r.get("directness", ""),
            "Affected Scenarios": scenarios,
            "Mechanism": r.get("mechanism", ""),
            "Analytical Summary": r.get("analytical_summary", ""),
            "Confidence": r.get("confidence", ""),
            "GovTrack URL": source_urls,
            "Error Type": r.get("error_type", ""),
            "Error Message": r.get("error_message", ""),
        })

        for i, e in enumerate(r.get("evidence", []), 1):
            evidence_rows.append({
                "Bill": r.get("bill_number", ""),
                "Title": r.get("title", ""),
                "Evidence #": i,
                "Quote": e.get("quote", ""),
                "Source": e.get("source", "GovTrack"),
                "Why It Matters": e.get("why_it_matters", ""),
                "GovTrack URL": source_urls,
            })

        if status == "FAILED":
            failed_rows.append({
                "Bill": r.get("bill_number", ""),
                "Title": r.get("title", ""),
                "Country": r.get("country", ""),
                "Error Type": r.get("error_type", ""),
                "Error Message": r.get("error_message", ""),
                "GovTrack URL": source_urls,
            })

    all_df = pd.DataFrame(all_rows)
    relevant_df = all_df[(all_df["Analysis Status"] == "SUCCESS") & (all_df["Relevant"] == "YES")]
    not_relevant_df = all_df[(all_df["Analysis Status"] == "SUCCESS") & (all_df["Relevant"] == "NO")]
    evidence_df = pd.DataFrame(evidence_rows)
    failed_df = pd.DataFrame(failed_rows)
    methodology_df = pd.DataFrame({
        "Methodology": ["Evidence source", "Error rule"],
        "Description": [
            "GovTrack URL only; Congress.gov secondary source is ignored.",
            "API/model failures are ANALYSIS FAILED and are never counted as Not Relevant.",
        ],
    })

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        all_df.to_excel(writer, index=False, sheet_name="All Results")
        relevant_df.to_excel(writer, index=False, sheet_name="Relevant")
        not_relevant_df.to_excel(writer, index=False, sheet_name="Not Relevant")
        evidence_df.to_excel(writer, index=False, sheet_name="Evidence")
        failed_df.to_excel(writer, index=False, sheet_name="Failed Analysis")
        methodology_df.to_excel(writer, index=False, sheet_name="Methodology")

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                letter = col[0].column_letter
                width = 12
                for cell in list(col)[:100]:
                    width = max(width, min(len(str(cell.value or "")) + 2, 60))
                ws.column_dimensions[letter].width = width

    return buffer.getvalue()


if st.session_state.results:
    res = st.session_state.results
    success = [r for r in res if r.get("analysis_status", "SUCCESS") == "SUCCESS"]
    failed = [r for r in res if r.get("analysis_status", "SUCCESS") == "FAILED"]
    rel = [r for r in success if r.get("relevant") == "YES"]
    non = [r for r in success if r.get("relevant") == "NO"]

    st.subheader("📌 Analysis status")
    a, b, c, d = st.columns(4)
    a.metric("Reviewed", len(res))
    b.metric("Relevant", len(rel))
    c.metric("Not relevant", len(non))
    d.metric("Analysis failed", len(failed))

    if failed:
        st.warning(f"⚠️ {len(failed)} action(s) failed analysis and are NOT counted as Not Relevant.")

    if rel:
        x = pd.DataFrame(rel)["primary_classification"].value_counts().reset_index()
        x.columns = ["Classification", "Actions"]
        st.plotly_chart(
            px.bar(x, x="Classification", y="Actions", color="Classification",
                   color_discrete_sequence=px.colors.qualitative.Bold),
            use_container_width=True
        )

    st.subheader("✅ Relevant Congressional actions")
    if not rel:
        st.caption("No successfully analyzed action was classified as relevant in this run.")

    for r in rel:
        with st.expander(f"{r['bill_number']} — {r['title']} | {r.get('primary_classification','')}"):
            st.write("**Country:**", r.get("country", ""))
            st.write("**Policy stage:**", r.get("policy_stage", ""))
            st.write("**Directness:**", r.get("directness", ""))
            st.write("**Affected scenarios:**", ", ".join(r.get("affected_scenarios", [])))
            st.write("**Mechanism:**", r.get("mechanism", ""))
            st.write("**Analysis:**", r.get("analytical_summary", ""))
            for e in r.get("evidence", []):
                st.info(f"“{e.get('quote','')}”\n\nSource: {e.get('source','GovTrack')}\n\nWhy it matters: {e.get('why_it_matters','')}")

    with st.expander(f"⚪ Not relevant ({len(non)})"):
        for r in non:
            st.markdown(f"**{r['bill_number']} — {r['title']}**")
            st.write(r.get("analytical_summary", ""))
            st.divider()

    with st.expander(f"⚠️ Analysis failed ({len(failed)})"):
        for r in failed:
            st.markdown(f"**{r['bill_number']} — {r['title']}**")
            st.error(r.get("error_message", "Unknown analysis error"))
            st.divider()

    st.subheader("📥 Export research results")
    st.download_button(
        "⬇️ Download DAQO analysis + evidence (Excel)",
        data=build_excel_export(res),
        file_name="DAQO_RAG_Analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
