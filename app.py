from pathlib import Path
import os
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from src.rag_engine import (
    build_document_store,
    run_gate_analysis,
    run_custom_filter,
    make_final_summary,
    save_results_csv,
)

load_dotenv()

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "raw_data.xlsx"
OUTPUT_DIR = BASE / "outputs"

st.set_page_config(
    page_title="DAQO Congressional Risk RAG",
    page_icon="🏛️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1500px; padding-top: 1.15rem; padding-bottom: 2.5rem;}
    .hero {
        border-radius: 22px;
        padding: 24px 28px;
        background: linear-gradient(135deg, #12263a 0%, #244f73 45%, #3d7ea6 100%);
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,.14);
    }
    .card {
        border: 1px solid rgba(130,140,160,.22);
        border-radius: 18px;
        padding: 18px 20px;
        margin: 10px 0 16px 0;
        background: rgba(255,255,255,.035);
    }
    .keep {border-left: 6px solid #2ecc71;}
    .possible {border-left: 6px solid #f5b041;}
    .drop {border-left: 6px solid #95a5a6;}
    .small {color: #8c98a8; font-size: .92rem;}
    .pill {
        display:inline-block;
        padding:4px 10px;
        border-radius:999px;
        margin-right:6px;
        background:rgba(77,130,220,.15);
        font-weight:700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🏛️ DAQO Congressional Risk & Opportunity RAG</h1>
        <div>
            A qualitative policy-analysis pipeline for a transformer manufacturer.
            The app reads Congressional documents, applies pre-defined legal/business gates,
            saves only the bills that can actually reach DAQO, and then lets you run additional
            open-ended filters on the surviving document set.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_excel():
    if not DATA_PATH.exists():
        return None
    df = pd.read_excel(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    return df

df = load_excel()

if df is None:
    st.error("The Excel file was not found.")
    st.code(r"data\raw_data.xlsx")
    st.info(
        "Keep your workbook at:\n\n"
        r"C:\Users\diego\OneDrive\Área de Trabalho\python\Superprof_Students\Daina\rag_congress_bills_app\data\raw_data.xlsx"
    )
    st.stop()


def find_col(*names):
    normalized = {
        c.lower().replace(" ", "").replace("-", "").replace("_", ""): c
        for c in df.columns
    }
    for n in names:
        key = n.lower().replace(" ", "").replace("-", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    for c in df.columns:
        lc = c.lower()
        for n in names:
            if n.lower() in lc:
                return c
    return None


columns = {
    "bill_col": find_col("Bill Number", "Bill"),
    "title_col": find_col("Title"),
    "party_col": find_col("Party"),
    "status_col": find_col("Status"),
    "country_col": find_col("Country"),
    "sponsor_col": find_col("Sponsor"),
    "cosponsor_col": find_col("Cosponsors"),
    "type_col": find_col("Bill Type", "Type"),
    "govtrack_col": find_col("GOVTRACK URL", "GovTrack"),
    "congress_col": find_col("Secondary Source", "congress.gov"),
    "congress_session_col": find_col("Congress"),
}

st.subheader("🧭 Research framework")

with st.expander("What counts as relevant to DAQO?", expanded=True):
    st.markdown(
        """
        **Company context used by the app**

        DAQO is treated as a **manufacturer of transformers / electrical power equipment**.
        The analysis therefore checks Congressional actions against three core dimensions:

        - **Product:** transformers, HTS 8504 power-electrical goods, electrical equipment and relevant components.
        - **Industry:** energy, grid equipment, transformers, inverters, power electronics, semiconductors and related supply chains.
        - **Country / ownership:** China / PRC origin, Chinese ownership, Mexico manufacturing and U.S. market access.

        **The app does not keep a bill simply because it mentions China.**
        It must have an operative mechanism or a meaningful non-binding policy signal that can plausibly reach DAQO.
        """
    )

with st.expander("Bill / resolution meaning"):
    framework_df = pd.DataFrame(
        [
            ["Bill", "Proposed law", "Binding if enacted", "Policy action"],
            ["Joint Resolution", "Proposal used to make law", "Can be binding", "Congressional pressure / action"],
            ["Concurrent Resolution", "Congress expresses a position", "Not binding", "Political sentiment"],
            ["Simple Resolution", "One chamber expresses a position", "Not binding", "Political sentiment / leading indicator"],
        ],
        columns=["Type", "Simple meaning", "If passed", "Demonstrates"],
    )
    st.dataframe(framework_df, use_container_width=True, hide_index=True)

with st.expander("The 5 qualitative gates"):
    st.markdown(
        """
        **Gate 0 — Evidence quality**  
        Use **operative sections**. Do not rely on titles, findings or “whereas” clauses alone.

        **Gate 1 — Is there a mechanism?**  
        Look for something that **obliges, prohibits, taxes, funds, restricts or conditions**.  
        Studies, reports, strategies and commissions normally fail this gate.  
        **Exception:** a non-binding resolution may still be retained when it is a credible leading indicator on one of the DAQO channels.

        **Gate 2 — Does the mechanism reach DAQO?**  
        It must reach DAQO by **product** or by **entity / ownership**.  
        Product examples: transformers, HTS 8504 power equipment, energy/grid electrical equipment and relevant components.  
        Entity example: restrictions applying to PRC-organised or Chinese-owned firms regardless of product.

        **Gate 3 — Can DAQO actually trigger the provision?**  
        The factual predicate must be satisfiable by DAQO.  
        Example: “any entity organised under PRC law” may reach DAQO; “persons militarising South China Sea features” would not.  
        Carve-outs matter: if imports are expressly excluded from sanctions, shipment risk may fail unless another section applies.

        **Gate 4 — Which business channel is affected?**  
        **Country** — Chinese origin or ownership.  
        **Industry** — energy, grid, transformers, inverters, solar, semiconductors; industry beats country when a named electrical product is directly targeted.  
        **Investment** — affects ability/cost to establish U.S. or Mexico manufacturing.  
        **Opportunity** — DAQO receives a benefit and is actually eligible. If PRC-owned firms are excluded from the benefit, treat it as an **investment risk**, not an opportunity.
        """
    )

st.divider()
st.subheader("📊 Dataset overview")

m1, m2, m3, m4 = st.columns(4)
m1.metric("📄 Bills", len(df))

if columns["country_col"]:
    m2.metric("🌍 Countries", df[columns["country_col"]].nunique())
else:
    m2.metric("🌍 Countries", "—")

if columns["party_col"]:
    m3.metric("🏷️ Parties", df[columns["party_col"]].nunique())
else:
    m3.metric("🏷️ Parties", "—")

if columns["status_col"]:
    enacted = (
        df[columns["status_col"]]
        .fillna("")
        .astype(str)
        .str.contains("Enacted|Signed", case=False, regex=True)
        .sum()
    )
    m4.metric("✅ Enacted / Signed", int(enacted))
else:
    m4.metric("✅ Enacted / Signed", "—")

c1, c2 = st.columns(2)

with c1:
    if columns["party_col"]:
        x = df[columns["party_col"]].fillna("Unknown").astype(str).value_counts().reset_index()
        x.columns = ["Party", "Bills"]
        fig = px.bar(x, x="Party", y="Bills", color="Party", title="Bills by Party", color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with c2:
    if columns["status_col"]:
        tmp = df.copy()
        tmp["Outcome"] = (
            tmp[columns["status_col"]]
            .fillna("Unknown")
            .astype(str)
            .apply(lambda x: "Became law" if ("enacted" in x.lower() or "signed" in x.lower()) else ("Moved / active" if any(k in x.lower() for k in ["passed", "committee", "reported"]) else "Died / other"))
        )
        x = tmp["Outcome"].value_counts().reset_index()
        x.columns = ["Outcome", "Bills"]
        fig = px.pie(x, names="Outcome", values="Bills", title="Did the bills go anywhere?", hole=.45, color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

with st.expander("👀 View raw Excel data"):
    st.dataframe(df, use_container_width=True)

st.divider()

defaults = {
    "doc_store": [],
    "build_summary": None,
    "link_log": [],
    "gate_results": [],
    "stage1_docs": [],
    "custom_results": {},
    "stage_docs": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.subheader("📚 Step 1 — Build the Congressional document base")
st.write("The app checks every selected bill and tries both GovTrack and Congress.gov. If one source fails, the other can still support the analysis.")

max_bills = st.number_input("How many bills should we scan?", min_value=1, max_value=max(1, len(df)), value=len(df), step=1)

if st.button("🌐 Build / Refresh Document Base", type="primary"):
    progress = st.progress(0)
    status = st.empty()

    def cb(done, total, message):
        progress.progress(min(done / max(total, 1), 1.0))
        status.write(message)

    docs, summary, log = build_document_store(df=df.head(int(max_bills)), column_map=columns, progress_callback=cb)
    st.session_state.doc_store = docs
    st.session_state.build_summary = summary
    st.session_state.link_log = log
    st.session_state.gate_results = []
    st.session_state.stage1_docs = []
    st.session_state.custom_results = {}
    st.session_state.stage_docs = {}

    progress.progress(1.0)
    status.empty()
    if docs:
        st.success("✅ Document base ready.")
    else:
        st.error("No readable documents were loaded.")

if st.session_state.build_summary:
    s = st.session_state.build_summary
    st.markdown(f"""
        <div class="card">
            <b>🌐 {s['links_checked']} source links checked</b><br>
            ✅ {s['sources_loaded']} source pages successfully read.<br>
            ⚠️ {s['source_errors']} source links could not be read.<br>
            📚 {s['documents_created']} bill-level documents are ready for qualitative review.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔧 Source diagnostics"):
        st.dataframe(pd.DataFrame(st.session_state.link_log), use_container_width=True, hide_index=True)

st.divider()
st.subheader("🎯 Step 2 — Run the DAQO gate analysis")

mandatory_questions = [
    "Could this Congressional action affect DAQO's ability to manufacture in China and serve the U.S. market?",
    "Could this Congressional action affect DAQO's ability to manufacture in Mexico and serve the U.S. market?",
    "Could this Congressional action affect DAQO's ability to invest in a manufacturing facility or assembly line in the U.S.?",
]

for i, q in enumerate(mandatory_questions, 1):
    st.markdown(f"**Q{i}. {q}**")

st.caption("The decision is qualitative. Bills are not kept because of a numeric similarity score. The model must work through Gate 0 → Gate 4 and cite the operative evidence.")
api_key = os.getenv("OPENAI_API_KEY", "").strip()

if st.button("🧠 Run DAQO Gate Analysis"):
    if not st.session_state.doc_store:
        st.error("Build the document base first.")
    elif not api_key:
        st.error("OPENAI_API_KEY is missing from `.env`.")
    else:
        progress = st.progress(0)
        status = st.empty()

        def gate_cb(done, total, message):
            progress.progress(min(done / max(total, 1), 1.0))
            status.write(message)

        results, kept_docs = run_gate_analysis(documents=st.session_state.doc_store, questions=mandatory_questions, api_key=api_key, progress_callback=gate_cb)
        st.session_state.gate_results = results
        st.session_state.stage1_docs = kept_docs
        st.session_state.custom_results = {}
        st.session_state.stage_docs = {}
        progress.progress(1.0)
        status.empty()
        st.success(f"✅ Gate analysis complete. {len(kept_docs)} documents survived for the next stage.")

if st.session_state.gate_results:
    results = st.session_state.gate_results
    kept = [r for r in results if r["keep_document"]]
    dropped = [r for r in results if not r["keep_document"]]

    st.markdown(f"""
        <div class="card">
            <b>📌 Gate-analysis result</b><br>
            Reviewed <b>{len(results)}</b> documents.<br>
            ✅ <b>{len(kept)}</b> survived the DAQO tests.<br>
            🗑️ <b>{len(dropped)}</b> failed because the operative mechanism did not meaningfully reach DAQO.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### ✅ Bills that survived the DAQO gates")
    for r in kept:
        with st.expander(f"✅ {r['bill_number']} — {r['title']} | {r.get('bucket','Other')} | {r.get('daqo_actionability','Unknown')}"):
            st.markdown(f"""
                **Mechanism bucket:** {r.get('bucket','')}  
                **Legislative outcome:** {r.get('legislative_outcome','')}  
                **Binding / signal:** {r.get('binding_signal','')}  
                **Primary channel:** {r.get('primary_channel','')}  
                **DAQO actionability:** {r.get('daqo_actionability','')}
                """)
            st.markdown("#### Gate-by-gate reasoning")
            for gate in r.get("gates", []):
                icon = "🟢" if gate.get("pass") else "⚪"
                st.markdown(f"**{icon} {gate.get('gate','Gate')} — {gate.get('result','')}**")
                st.write(gate.get("reason", ""))
            st.markdown("#### The 3 mandatory questions")
            for qa in r.get("question_assessments", []):
                verdict = qa.get("verdict", "NO")
                icon = "🟢" if verdict == "YES" else ("🟠" if verdict == "POSSIBLE" else "⚪")
                st.markdown(f"**{icon} Q{qa.get('question_number')}: {verdict} — {qa.get('short_reason','')}**")
                st.write(qa.get("business_meaning", ""))
            if r.get("evidence_quotes"):
                st.markdown("#### Operative evidence")
                for ev in r["evidence_quotes"]:
                    st.markdown(f"- {ev}")
            st.caption("Sources used: " + ", ".join(r.get("sources_used", [])))

    with st.expander("🗑️ Show excluded bills"):
        for r in dropped:
            st.markdown(f"**{r['bill_number']} — {r['title']}**")
            st.write(r.get("overall_reason", ""))

st.divider()
st.subheader("🔎 Step 3 — Apply additional qualitative filters")
st.write("Each filter only receives documents that survived the previous stage. Use these boxes for risk, opportunity, tariff exposure, sourcing, investment constraints, grid-equipment policy, compliance, supply-chain issues, or any other qualitative research question.")

placeholders = [
    "Example: Does this bill create a material risk or opportunity for DAQO's transformer business?",
    "Example: Could this bill increase the cost of supplying transformers to the U.S. market?",
    "Example: Does this bill create an opportunity for U.S. transformer manufacturing that DAQO is actually eligible to use?",
    "Example: Could this bill create supply-chain, ownership, tariff or compliance constraints for DAQO?",
]

for stage in range(2, 6):
    st.markdown(f"### Filter {stage}")
    q = st.text_area(f"Qualitative question for Filter {stage}", key=f"custom_question_{stage}", placeholder=placeholders[stage - 2], height=88)

    if st.button(f"▶️ Run Filter {stage}", key=f"run_filter_{stage}"):
        input_docs = st.session_state.stage1_docs if stage == 2 else st.session_state.stage_docs.get(stage - 1, [])
        if not input_docs:
            st.warning("No documents survived the previous stage.")
        elif not q.strip():
            st.warning("Type a qualitative filter question first.")
        elif not api_key:
            st.error("OPENAI_API_KEY is missing from `.env`.")
        else:
            progress = st.progress(0)
            status = st.empty()

            def fcb(done, total, message):
                progress.progress(min(done / max(total, 1), 1.0))
                status.write(message)

            results, kept_docs = run_custom_filter(documents=input_docs, question=q, api_key=api_key, progress_callback=fcb)
            st.session_state.custom_results[stage] = results
            st.session_state.stage_docs[stage] = kept_docs
            for later in range(stage + 1, 6):
                st.session_state.custom_results.pop(later, None)
                st.session_state.stage_docs.pop(later, None)
            progress.progress(1.0)
            status.empty()
            st.success(f"✅ Filter {stage} complete. {len(kept_docs)} documents remain.")

    if stage in st.session_state.custom_results:
        rr = st.session_state.custom_results[stage]
        kk = [x for x in rr if x["keep_document"]]
        dd = [x for x in rr if not x["keep_document"]]
        st.markdown(f"""
            <div class="card">
                Filter {stage}: <b>{len(rr)}</b> reviewed → <b>{len(kk)}</b> saved → <b>{len(dd)}</b> excluded.
            </div>
            """, unsafe_allow_html=True)
        for r in kk:
            icon = "🟢" if r["verdict"] == "YES" else "🟠"
            with st.expander(f"{icon} {r['bill_number']} — {r['title']} — {r['verdict']}"):
                st.markdown(f"**Why kept:** {r['short_reason']}")
                st.write(r["business_meaning"])
                st.markdown(f"**Channel:** {r.get('channel','')}")
                st.markdown(f"**DAQO actionability:** {r.get('daqo_actionability','')}")
                if r.get("evidence_quotes"):
                    st.markdown("**Evidence:**")
                    for ev in r["evidence_quotes"]:
                        st.markdown(f"- {ev}")
        with st.expander(f"🗑️ Excluded by Filter {stage}"):
            for r in dd:
                st.markdown(f"**{r['bill_number']} — {r['title']}**")
                st.write(r["short_reason"])

st.divider()
st.subheader("🏆 Final document set")

final_docs = st.session_state.stage1_docs
last_stage = 1
for stage in range(2, 6):
    if stage in st.session_state.stage_docs:
        final_docs = st.session_state.stage_docs[stage]
        last_stage = stage

if final_docs:
    st.success(f"🎉 {len(final_docs)} documents remain after Filter {last_stage}.")
    labels = ["Document base", "DAQO gates"]
    values = [len(st.session_state.doc_store), len(st.session_state.stage1_docs)]
    for stage in range(2, 6):
        if stage in st.session_state.stage_docs:
            labels.append(f"Filter {stage}")
            values.append(len(st.session_state.stage_docs[stage]))
    funnel = pd.DataFrame({"Stage": labels, "Documents": values})
    fig = px.funnel(funnel, x="Documents", y="Stage", color="Stage", title="DAQO Research Funnel", color_discrete_sequence=px.colors.qualitative.Prism)
    st.plotly_chart(fig, use_container_width=True)
    final_table = pd.DataFrame([
        {"Bill": d["bill_number"], "Title": d["title"], "Status": d.get("status", ""), "Country": d.get("country", ""), "Sources": ", ".join(d.get("sources", []))}
        for d in final_docs
    ])
    st.dataframe(final_table, use_container_width=True, hide_index=True)

    if api_key and st.button("✨ Create final qualitative research summary"):
        summary = make_final_summary(
            documents=final_docs,
            mandatory_questions=mandatory_questions,
            custom_questions=[st.session_state.get(f"custom_question_{stage}", "") for stage in range(2, 6) if st.session_state.get(f"custom_question_{stage}", "").strip()],
            api_key=api_key,
        )
        st.markdown(f"""
            <div class="card">
                <h3>{summary.get('headline','Final assessment')}</h3>
                <p>{summary.get('summary','')}</p>
                <p><b>Risk themes:</b> {', '.join(summary.get('risks', [])) or 'None clearly identified'}</p>
                <p><b>Opportunity themes:</b> {', '.join(summary.get('opportunities', [])) or 'None clearly identified'}</p>
                <p><b>Most important channels:</b> {', '.join(summary.get('channels', [])) or 'None clearly identified'}</p>
                <p><b>Suggested next step:</b> {summary.get('recommended_action','')}</p>
            </div>
            """, unsafe_allow_html=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = save_results_csv(final_docs, OUTPUT_DIR / "final_documents.csv")
    st.download_button("⬇️ Download final document list", data=csv_path.read_bytes(), file_name="final_documents.csv", mime="text/csv")
else:
    st.info("Run the DAQO gate analysis first.")

st.divider()
st.caption("Method: operative evidence → mechanism → DAQO reach → trigger test → business channel → mandatory questions → sequential qualitative filters → final evidence-backed bill set")
