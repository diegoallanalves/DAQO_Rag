
from pathlib import Path
import io
import json
import os
import random

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.prescreen import (
    TOPIC_GROUPS,
    load_checkpoint,
    make_checkpoint_key,
    normalize_text,
    prescreen_dataframe,
    save_checkpoint,
)
from src.rag_engine import build_document_store, run_classification


load_dotenv()
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "raw_data.xlsx"
CHECKPOINT_PATH = BASE / "outputs" / "analysis_checkpoint.json"

st.set_page_config(
    page_title="DAQO Congressional RAG",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ DAQO Congressional Risk & Opportunity RAG")
st.caption(
    "Smart pre-screening reduces the dataset before web retrieval and OpenAI analysis."
)


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
    norm = {
        c.lower().replace(" ", "").replace("-", "").replace("_", ""): c
        for c in df.columns
    }
    for n in names:
        key = n.lower().replace(" ", "").replace("-", "").replace("_", "")
        if key in norm:
            return norm[key]
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
    "congress_col": find_col(
        "Secondary Source - full text (congress.gov)",
        "Secondary Source",
        "congress.gov",
    ),
    "congress_session_col": find_col("Congress"),
}

if not cols["govtrack_col"] and not cols["congress_col"]:
    st.error(
        'The workbook needs a "GOVTRACK URL" or '
        '"Secondary Source - full text (congress.gov)" column.'
    )
    st.stop()


# -------------------------------------------------------------------
# STEP 0 — FILTERS
# -------------------------------------------------------------------
st.subheader("🎛️ Step 0 — Reduce the search space")

f1, f2 = st.columns(2)

with f1:
    st.markdown("#### 🏛️ Congress")
    if cols["congress_session_col"]:
        congress_values = sorted(
            [
                str(v).strip()
                for v in df[cols["congress_session_col"]].dropna().unique()
                if str(v).strip()
            ]
        )
        selected_congress = st.multiselect(
            "Congress sessions",
            congress_values,
            default=congress_values,
        )
    else:
        selected_congress = []

with f2:
    st.markdown("#### 🌎 Country")
    if cols["country_col"]:
        country_values = sorted(
            [
                str(v).strip()
                for v in df[cols["country_col"]].dropna().unique()
                if str(v).strip()
            ]
        )
        selected_countries = st.multiselect(
            "Countries",
            country_values,
            default=country_values,
            help="Select China, Mexico, or multiple countries.",
        )
    else:
        selected_countries = []


mask = pd.Series(True, index=df.index)
if cols["congress_session_col"]:
    mask &= (
        df[cols["congress_session_col"]]
        .astype(str)
        .str.strip()
        .isin(selected_congress)
    )
if cols["country_col"]:
    mask &= (
        df[cols["country_col"]]
        .astype(str)
        .str.strip()
        .isin(selected_countries)
    )

filtered_df = df[mask].copy()

m1, m2, m3 = st.columns(3)
m1.metric("Original dataset", f"{len(df):,}")
m2.metric("After Congress + Country", f"{len(filtered_df):,}")
m3.metric("Reduction", f"{(1-len(filtered_df)/max(len(df),1))*100:.1f}%")


# -------------------------------------------------------------------
# SOURCE
# -------------------------------------------------------------------
st.subheader("🌐 Step 1 — Choose evidence source")

source_options = {}
if cols["govtrack_col"]:
    source_options["GovTrack Bill Page"] = "govtrack"
if cols["congress_col"]:
    source_options["Congress.gov Full Bill Text"] = "congress"

source_label = st.radio(
    "Website data source",
    list(source_options.keys()),
    horizontal=True,
)
selected_source = source_options[source_label]
source_col = (
    cols["govtrack_col"]
    if selected_source == "govtrack"
    else cols["congress_col"]
)


# -------------------------------------------------------------------
# LOCAL NLP PRE-SCREEN
# -------------------------------------------------------------------
st.subheader("🔎 Step 2 — Local relevance pre-screen")
st.caption(
    "This step runs locally and does not use OpenAI credits. "
    "It ranks records before expensive retrieval and AI analysis."
)

default_topics = [
    "China / PRC",
    "Transformers / Grid / Electrical",
    "Tariffs / Trade / Imports",
    "Sanctions / Export Controls",
    "Supply Chain / Reshoring",
    "Investment / CFIUS / Ownership",
    "Mexico / USMCA",
]

selected_topics = st.multiselect(
    "Topics to screen for",
    options=list(TOPIC_GROUPS.keys()),
    default=[t for t in default_topics if t in TOPIC_GROUPS],
)

text_columns = [
    c
    for c in [
        cols["title_col"],
        cols["status_col"],
        cols["type_col"],
        cols["country_col"],
    ]
    if c
]

screened = prescreen_dataframe(
    filtered_df,
    text_columns=text_columns,
    selected_topics=selected_topics,
    source_col=source_col,
    bill_col=cols["bill_col"],
)

q1, q2, q3, q4 = st.columns(4)
q1.metric("Valid source URLs", int((screened["Source Quality"] == "Valid URL").sum()))
q2.metric("Missing / invalid URLs", int((screened["Source Quality"] != "Valid URL").sum()))
q3.metric("Duplicate bills", int(screened["Duplicate Bill"].sum()))
q4.metric("High + Very High", int(screened["PreScreen Band"].isin(["High", "Very High"]).sum()))

bands = st.multiselect(
    "Pre-screen bands to send forward",
    ["Very High", "High", "Medium", "Low"],
    default=["Very High", "High", "Medium"],
)

candidate_df = screened[
    screened["PreScreen Band"].isin(bands)
    & (screened["Source Quality"] == "Valid URL")
    & (~screened["Duplicate Bill"])
].copy()

st.caption(
    f"**{len(candidate_df):,}** actions remain after the local relevance and data-quality screen."
)

if not candidate_df.empty:
    chart = (
        candidate_df["PreScreen Band"]
        .value_counts()
        .reindex(["Very High", "High", "Medium", "Low"], fill_value=0)
        .reset_index()
    )
    chart.columns = ["Band", "Actions"]
    st.plotly_chart(
        px.bar(
            chart,
            x="Band",
            y="Actions",
            color="Band",
            title="Local relevance screening",
            color_discrete_sequence=px.colors.qualitative.Bold,
        ),
        use_container_width=True,
    )

with st.expander("Preview ranked candidates"):
    preview_cols = [
        c
        for c in [
            cols["bill_col"],
            cols["title_col"],
            cols["congress_session_col"],
            cols["country_col"],
            "PreScreen Score",
            "PreScreen Band",
            "PreScreen Topics",
            "PreScreen Terms",
            "Source Quality",
        ]
        if c
    ]
    st.dataframe(
        candidate_df[preview_cols]
        .sort_values("PreScreen Score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


# -------------------------------------------------------------------
# PROCESSING MODE
# -------------------------------------------------------------------
st.subheader("⚙️ Step 3 — Choose processing strategy")

processing_mode = st.radio(
    "Processing mode",
    [
        "Highest relevance first",
        "Random sample",
        "All filtered candidates",
        "Resume unfinished / failed",
    ],
    horizontal=True,
)

max_available = max(1, len(candidate_df))
batch_size = st.number_input(
    "Maximum actions for this run",
    min_value=1,
    max_value=max_available,
    value=min(25, max_available),
    step=1,
    disabled=candidate_df.empty,
)

checkpoint = load_checkpoint(CHECKPOINT_PATH)

work_df = candidate_df.copy()

if processing_mode == "Highest relevance first":
    work_df = work_df.sort_values("PreScreen Score", ascending=False)
elif processing_mode == "Random sample":
    work_df = work_df.sample(frac=1, random_state=random.randint(1, 10_000))
elif processing_mode == "Resume unfinished / failed":
    source_name = source_label
    keep_rows = []
    for idx, row in work_df.iterrows():
        bill = normalize_text(row.get(cols["bill_col"])) if cols["bill_col"] else ""
        url = normalize_text(row.get(source_col))
        key = make_checkpoint_key(bill, url, source_name)
        saved = checkpoint.get(key)
        # Keep anything never completed successfully.
        if not saved or saved.get("analysis_status") != "SUCCESS":
            keep_rows.append(idx)
    work_df = work_df.loc[keep_rows]

work_df = work_df.head(int(batch_size))

c1, c2 = st.columns(2)
c1.metric("Candidates available", f"{len(candidate_df):,}")
c2.metric("Actions in this run", f"{len(work_df):,}")

if work_df.empty:
    st.info("Nothing remains to process under the current settings.")


# -------------------------------------------------------------------
# RETRIEVAL + AI
# -------------------------------------------------------------------
st.subheader("📚 Step 4 — Retrieve evidence")

for key_name, default in {
    "docs": [],
    "summary": None,
    "log": [],
    "results": [],
}.items():
    if key_name not in st.session_state:
        st.session_state[key_name] = default


if st.button(
    "🌐 Build Evidence Base",
    type="primary",
    disabled=work_df.empty,
):
    p = st.progress(0)
    msg = st.empty()

    def cb(done, total, text):
        p.progress(min(done / max(total, 1), 1.0))
        msg.write(text)

    docs, summary, log = build_document_store(
        work_df,
        cols,
        selected_source,
        cb,
    )
    st.session_state.docs = docs
    st.session_state.summary = summary
    st.session_state.log = log
    st.session_state.results = []

    p.progress(1.0)
    msg.empty()
    st.success(f"{len(docs)} evidence documents successfully loaded.")

if st.session_state.summary:
    s = st.session_state.summary
    a, b, c, d = st.columns(4)
    a.metric("Bills scanned", s["bills_scanned"])
    b.metric("Source links checked", s["links_checked"])
    c.metric("Sources loaded", s["sources_loaded"])
    d.metric("Retrieval errors", s["source_errors"])

    with st.expander("🔎 Retrieval diagnostics"):
        st.dataframe(
            pd.DataFrame(st.session_state.log),
            use_container_width=True,
            hide_index=True,
        )


st.subheader("🧠 Step 5 — Qualitative AI classification")

api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key:
    try:
        api_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        api_key = ""

if not api_key:
    st.caption(
        "Add OPENAI_API_KEY to .env locally or Streamlit Secrets online."
    )

if st.button(
    "🧠 Run Qualitative Classification",
    disabled=(not st.session_state.docs or not api_key),
):
    p = st.progress(0)
    msg = st.empty()

    def cb2(done, total, text):
        p.progress(min(done / max(total, 1), 1.0))
        msg.write(text)

    results = run_classification(
        st.session_state.docs,
        api_key,
        cb2,
    )
    st.session_state.results = results

    # Save every returned result into a persistent local checkpoint.
    cp = load_checkpoint(CHECKPOINT_PATH)
    for result in results:
        bill = normalize_text(result.get("bill_number"))
        urls = result.get("source_urls", [])
        url = normalize_text(urls[0] if urls else "")
        key = make_checkpoint_key(bill, url, source_label)
        cp[key] = result
    save_checkpoint(CHECKPOINT_PATH, cp)

    p.progress(1.0)
    msg.empty()
    st.success(
        "Classification complete. Results were checkpointed so a later run can resume."
    )


# -------------------------------------------------------------------
# RESULTS
# -------------------------------------------------------------------
def build_excel_export(results):
    all_rows = []
    evidence_rows = []
    failed_rows = []

    for r in results:
        status = r.get("analysis_status", "SUCCESS")
        scenarios = "; ".join(r.get("affected_scenarios", []))
        source_urls = "; ".join(r.get("source_urls", []))

        all_rows.append(
            {
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
                "Evidence Source": r.get("evidence_source", ""),
                "Source URL": source_urls,
                "Error Type": r.get("error_type", ""),
                "Error Message": r.get("error_message", ""),
            }
        )

        for i, e in enumerate(r.get("evidence", []), 1):
            evidence_rows.append(
                {
                    "Bill": r.get("bill_number", ""),
                    "Title": r.get("title", ""),
                    "Evidence #": i,
                    "Quote": e.get("quote", ""),
                    "Source": e.get("source", r.get("evidence_source", "")),
                    "Why It Matters": e.get("why_it_matters", ""),
                    "Source URL": source_urls,
                }
            )

        if status == "FAILED":
            failed_rows.append(
                {
                    "Bill": r.get("bill_number", ""),
                    "Title": r.get("title", ""),
                    "Country": r.get("country", ""),
                    "Error Type": r.get("error_type", ""),
                    "Error Message": r.get("error_message", ""),
                    "Source URL": source_urls,
                }
            )

    all_df = pd.DataFrame(all_rows)
    evidence_df = pd.DataFrame(evidence_rows)
    failed_df = pd.DataFrame(failed_rows)

    if all_df.empty:
        relevant_df = pd.DataFrame()
        not_relevant_df = pd.DataFrame()
    else:
        relevant_df = all_df[
            (all_df["Analysis Status"] == "SUCCESS")
            & (all_df["Relevant"] == "YES")
        ]
        not_relevant_df = all_df[
            (all_df["Analysis Status"] == "SUCCESS")
            & (all_df["Relevant"] == "NO")
        ]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        all_df.to_excel(writer, index=False, sheet_name="All Results")
        relevant_df.to_excel(writer, index=False, sheet_name="Relevant")
        not_relevant_df.to_excel(writer, index=False, sheet_name="Not Relevant")
        evidence_df.to_excel(writer, index=False, sheet_name="Evidence")
        failed_df.to_excel(writer, index=False, sheet_name="Failed Analysis")

    return buffer.getvalue()


if st.session_state.results:
    results = st.session_state.results
    success = [
        r for r in results if r.get("analysis_status", "SUCCESS") == "SUCCESS"
    ]
    failed = [
        r for r in results if r.get("analysis_status", "SUCCESS") == "FAILED"
    ]
    relevant = [r for r in success if r.get("relevant") == "YES"]
    not_relevant = [r for r in success if r.get("relevant") == "NO"]

    st.subheader("📌 Run results")
    a, b, c, d = st.columns(4)
    a.metric("Reviewed", len(results))
    b.metric("Relevant", len(relevant))
    c.metric("Not relevant", len(not_relevant))
    d.metric("Failed", len(failed))

    if relevant:
        classifications = (
            pd.DataFrame(relevant)["primary_classification"]
            .value_counts()
            .reset_index()
        )
        classifications.columns = ["Classification", "Actions"]
        st.plotly_chart(
            px.pie(
                classifications,
                names="Classification",
                values="Actions",
                hole=0.5,
                title="Relevant DAQO classifications",
            ),
            use_container_width=True,
        )

    with st.expander(f"✅ Relevant ({len(relevant)})", expanded=True):
        for r in relevant:
            st.markdown(
                f"**{r.get('bill_number','')} — {r.get('title','')}**"
            )
            st.write(r.get("analytical_summary", ""))
            for e in r.get("evidence", [])[:5]:
                st.info(
                    f"{e.get('quote','')}\n\n"
                    f"Why it matters: {e.get('why_it_matters','')}"
                )
            st.divider()

    with st.expander(f"⚪ Not relevant ({len(not_relevant)})"):
        for r in not_relevant:
            st.markdown(
                f"**{r.get('bill_number','')} — {r.get('title','')}**"
            )
            st.write(r.get("analytical_summary", ""))
            st.divider()

    with st.expander(f"⚠️ Failed ({len(failed)})"):
        for r in failed:
            st.markdown(
                f"**{r.get('bill_number','')} — {r.get('title','')}**"
            )
            st.error(r.get("error_message", "Unknown analysis error"))

    st.download_button(
        "⬇️ Download this run to Excel",
        data=build_excel_export(results),
        file_name="DAQO_RAG_Analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# -------------------------------------------------------------------
# CHECKPOINT STATUS
# -------------------------------------------------------------------
st.divider()
st.subheader("💾 Checkpoint status")

cp = load_checkpoint(CHECKPOINT_PATH)
success_count = sum(
    1 for v in cp.values() if v.get("analysis_status") == "SUCCESS"
)
failed_count = sum(
    1 for v in cp.values() if v.get("analysis_status") == "FAILED"
)

x1, x2, x3 = st.columns(3)
x1.metric("Saved results", len(cp))
x2.metric("Successful", success_count)
x3.metric("Failed / retryable", failed_count)

if st.button("🗑️ Clear checkpoint"):
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    st.success("Checkpoint cleared.")
    st.rerun()
