from pathlib import Path
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
st.markdown("""
<style>
.block-container{max-width:1500px;padding-top:1.2rem}.hero{border-radius:22px;padding:24px 28px;background:linear-gradient(135deg,#12263a,#3d7ea6);color:white;margin-bottom:18px}.card{border:1px solid rgba(130,140,160,.25);border-radius:16px;padding:16px 18px;margin:10px 0}.yes{border-left:6px solid #2ecc71}.no{border-left:6px solid #95a5a6}.risk{border-left:6px solid #e74c3c}.opp{border-left:6px solid #2ecc71}.small{color:#8792a2}
</style>
<div class="hero"><h1>🏛️ DAQO Congressional Risk & Opportunity RAG</h1><p>Qualitative Congressional content analysis for DAQO's three U.S.-market strategies: China export, Mexico manufacture, and U.S. manufacture.</p></div>
""", unsafe_allow_html=True)

@st.cache_data
def load_excel():
    if not DATA_PATH.exists(): return None
    d = pd.read_excel(DATA_PATH); d.columns=[str(c).strip() for c in d.columns]; return d

df=load_excel()
if df is None:
    st.error("The Excel file was not found at data/raw_data.xlsx"); st.stop()

def find_col(*names):
    norm={c.lower().replace(" ","").replace("-","").replace("_",""):c for c in df.columns}
    for n in names:
        k=n.lower().replace(" ","").replace("-","").replace("_","")
        if k in norm:return norm[k]
    for c in df.columns:
        if any(n.lower() in c.lower() for n in names):return c
    return None

cols={"bill_col":find_col("Bill Number","Bill"),"title_col":find_col("Title"),"status_col":find_col("Status"),"country_col":find_col("Country"),"type_col":find_col("Bill Type","Type"),"govtrack_col":find_col("GOVTRACK URL","GovTrack"),"congress_col":find_col("Secondary Source","congress.gov"),"congress_session_col":find_col("Congress")}

st.subheader("🧭 Research method")
st.info("The workbook's **Country** value is treated as research data. The model does **not** infer or replace the country.")
with st.expander("How each Congressional action is classified", expanded=True):
    st.markdown("""
**1 — Relevance first:** Is this meaningful evidence of a DAQO risk or opportunity? **YES / NO**.  
A proposal, unsuccessful bill or non-binding resolution can still be relevant. Failure to become law is **never by itself** a reason to exclude it.

**2 — Exactly one primary classification if relevant:**  
🔴 **Country Risk** · 🟢 **Country Opportunity** · 🔴 **Industry Risk** · 🟢 **Industry Opportunity** · 🔴 **Investment Risk** · 🟢 **Investment Opportunity**

**3 — Separate research descriptors:** policy stage, directness (**Direct / Emerging / Sentiment-Political Pressure**) and affected DAQO scenario(s).

**4 — Evidence-backed explanation:** Congressional action → policy/economic mechanism → DAQO risk/opportunity → affected scenario.
""")
with st.expander("Important industry boundary"):
    st.markdown("""
DAQO's study product is **transformers — HTS 8504**. Industry relevance includes transformers, substations, switchgear, transmission/distribution, grid infrastructure and directly relevant electrical equipment/components.  
**5G, EVs/batteries, semiconductors, pharma, aerospace and unrelated military equipment are not automatically DAQO industry.** They may still provide broader **Country Risk** evidence when the substantive mechanism supports that conclusion.
""")

st.divider(); st.subheader("📊 Dataset overview")
a,b,c=st.columns(3); a.metric("Congressional actions",len(df)); b.metric("Countries in dataset",df[cols["country_col"]].nunique() if cols["country_col"] else "—"); c.metric("With Congress.gov/GovTrack fields",len(df))
with st.expander("👀 View source dataset"): st.dataframe(df,use_container_width=True)

for k,v in {"docs":[],"build":None,"log":[],"results":[]}.items():
    if k not in st.session_state: st.session_state[k]=v

st.divider(); st.subheader("📚 Step 1 — Retrieve Congressional evidence")
st.write("The app checks the selected rows and tries both GovTrack and Congress.gov. It keeps source URLs and passages so the final classification can show its evidence.")
max_bills=st.number_input("How many Congressional actions should we scan?",1,max(1,len(df)),len(df),1)
if st.button("🌐 Build / Refresh Evidence Base",type="primary"):
    p=st.progress(0); msg=st.empty()
    def cb(done,total,text): p.progress(min(done/max(total,1),1.0)); msg.write(text)
    docs,summary,log=build_document_store(df.head(int(max_bills)),cols,cb)
    st.session_state.docs=docs; st.session_state.build=summary; st.session_state.log=log; st.session_state.results=[]
    p.progress(1.0); msg.empty(); st.success(f"Evidence base ready: {len(docs)} Congressional actions have readable source material.")
if st.session_state.build:
    s=st.session_state.build
    x1,x2,x3,x4=st.columns(4); x1.metric("Bills scanned",s["bills_scanned"]); x2.metric("Links checked",s["links_checked"]); x3.metric("Sources loaded",s["sources_loaded"]); x4.metric("Source errors",s["source_errors"])
    with st.expander("🔧 Source diagnostics"): st.dataframe(pd.DataFrame(st.session_state.log),use_container_width=True,hide_index=True)

st.divider(); st.subheader("🎯 Step 2 — Classify DAQO relevance, risk & opportunity")
st.caption("This is the qualitative analysis itself — not a later filter. Every action is first tested for relevance, then receives exactly one primary classification if relevant.")
st.markdown("**Strategic scenarios assessed:** 🇨🇳 CHINA-EXPORT · 🇲🇽 MEXICO-MANUFACTURE · 🇺🇸 US-MANUFACTURE")

# Works locally with .env and on Streamlit Community Cloud with Secrets.
api_key=os.getenv("OPENAI_API_KEY","").strip()
if not api_key:
    try: api_key=str(st.secrets.get("OPENAI_API_KEY","")).strip()
    except Exception: pass

if st.button("🧠 Run Qualitative DAQO Classification"):
    if not st.session_state.docs: st.error("Build the evidence base first.")
    elif not api_key: st.error("OPENAI_API_KEY is missing. For Streamlit Cloud, add OPENAI_API_KEY in the app's Secrets settings.")
    else:
        p=st.progress(0); msg=st.empty()
        def cb2(done,total,text): p.progress(min(done/max(total,1),1.0)); msg.write(text)
        st.session_state.results=run_classification(st.session_state.docs,api_key,cb2)
        p.progress(1.0); msg.empty(); st.success("Qualitative classification complete.")

if st.session_state.results:
    results=st.session_state.results
    rel=[r for r in results if r.get("relevant")=="YES"]; non=[r for r in results if r.get("relevant")!="YES"]
    st.markdown("### 📌 Research results")
    m1,m2,m3=st.columns(3); m1.metric("Reviewed",len(results)); m2.metric("Relevant",len(rel)); m3.metric("Not relevant",len(non))
    if rel:
        chart=pd.DataFrame(rel)["primary_classification"].value_counts().reset_index(); chart.columns=["Classification","Actions"]
        fig=px.bar(chart,x="Classification",y="Actions",color="Classification",title="Relevant Congressional actions by primary classification",color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(showlegend=False); st.plotly_chart(fig,use_container_width=True)
        scenario_rows=[]
        for r in rel:
            for sc in r.get("affected_scenarios",[]): scenario_rows.append({"Scenario":sc,"Classification":r.get("primary_classification")})
        if scenario_rows:
            sc=pd.DataFrame(scenario_rows).groupby(["Scenario","Classification"]).size().reset_index(name="Actions")
            fig2=px.bar(sc,x="Scenario",y="Actions",color="Classification",title="Which DAQO strategies are affected?",barmode="stack",color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(fig2,use_container_width=True)

    st.markdown("### ✅ Relevant Congressional actions")
    for r in rel:
        cls=r.get("primary_classification",""); icon="🟢" if "OPPORTUNITY" in cls else "🔴"
        with st.expander(f"{icon} {r['bill_number']} — {r['title']} | {cls}"):
            c1,c2,c3=st.columns(3); c1.markdown(f"**Country (dataset):** {r.get('country','')}"); c2.markdown(f"**Policy stage:** {r.get('policy_stage','')}"); c3.markdown(f"**Directness:** {r.get('directness','')}")
            st.markdown("**Affected scenario(s):** "+(", ".join(r.get("affected_scenarios",[])) or "None"))
            st.markdown(f"**Mechanism:** {r.get('mechanism','')}")
            st.markdown(f"**Analysis:** {r.get('analytical_summary','')}")
            ev=r.get("evidence",[])
            if ev:
                st.markdown("**Evidence**")
                for e in ev: st.info(f"“{e.get('quote','')}”\n\nSource: {e.get('source','')} — {e.get('why_it_matters','')}")
            if r.get("source_urls"):
                st.markdown("**Source links used:**")
                for u in r["source_urls"]: st.code(u)

    with st.expander(f"⚪ Not relevant ({len(non)})"):
        for r in non:
            st.markdown(f"**{r['bill_number']} — {r['title']}**  \n{r.get('analytical_summary','')}")
            st.divider()

    export=pd.DataFrame([{"Bill":r.get("bill_number"),"Title":r.get("title"),"Country":r.get("country"),"Relevant":r.get("relevant"),"Primary Classification":r.get("primary_classification"),"Policy Stage":r.get("policy_stage"),"Directness":r.get("directness"),"Affected Scenarios":"; ".join(r.get("affected_scenarios",[])),"Mechanism":r.get("mechanism"),"Analytical Summary":r.get("analytical_summary"),"Confidence":r.get("confidence")} for r in results])
    st.download_button("⬇️ Download classification results (CSV)",export.to_csv(index=False).encode("utf-8-sig"),"daqo_congressional_classification.csv","text/csv")

st.divider(); st.caption("Method: relevance first → one primary classification → policy stage → directness → affected DAQO scenario(s) → evidence-backed analytical summary")
