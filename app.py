from pathlib import Path
import os, pandas as pd, streamlit as st, plotly.express as px
from dotenv import load_dotenv
from src.rag_engine import build_document_store, run_classification

load_dotenv()
BASE=Path(__file__).resolve().parent
DATA_PATH=BASE/"data"/"raw_data.xlsx"
st.set_page_config(page_title="DAQO Congressional RAG",page_icon="🏛️",layout="wide")
st.title("🏛️ DAQO Congressional Risk & Opportunity RAG")
st.caption("Qualitative Congressional content analysis for DAQO's China-export, Mexico-manufacture and U.S.-manufacture strategies.")

@st.cache_data
def load_excel():
    if not DATA_PATH.exists(): return None
    d=pd.read_excel(DATA_PATH); d.columns=[str(c).strip() for c in d.columns]; return d
df=load_excel()
if df is None:
    st.error("Missing data/raw_data.xlsx"); st.stop()

def find_col(*names):
    norm={c.lower().replace(" ","").replace("-","").replace("_",""):c for c in df.columns}
    for n in names:
        k=n.lower().replace(" ","").replace("-","").replace("_","")
        if k in norm:return norm[k]
    for c in df.columns:
        if any(n.lower() in c.lower() for n in names):return c
    return None

cols={"bill_col":find_col("Bill Number","Bill"),"title_col":find_col("Title"),"status_col":find_col("Status"),
"country_col":find_col("Country"),"type_col":find_col("Bill Type","Type"),"govtrack_col":find_col("GOVTRACK URL","GovTrack"),
"congress_col":find_col("Secondary Source","congress.gov"),"congress_session_col":find_col("Congress")}

st.info("Country is taken from the Excel dataset. The AI does not infer or replace it.")
st.subheader("📊 Dataset overview")
c1,c2=st.columns(2); c1.metric("Congressional actions",len(df)); c2.metric("Countries",df[cols["country_col"]].nunique() if cols["country_col"] else "—")
with st.expander("View source data"):st.dataframe(df,use_container_width=True)

for k,v in {"docs":[],"summary":None,"log":[],"results":[]}.items():
    if k not in st.session_state:st.session_state[k]=v

st.divider(); st.subheader("📚 Step 1 — Retrieve Congressional evidence")
n=st.number_input("Actions to scan",1,max(1,len(df)),min(10,len(df)),1)
if st.button("🌐 Build Evidence Base",type="primary"):
    p=st.progress(0); msg=st.empty()
    def cb(done,total,text):p.progress(min(done/max(total,1),1.0));msg.write(text)
    docs,s,l=build_document_store(df.head(int(n)),cols,cb)
    st.session_state.docs=docs;st.session_state.summary=s;st.session_state.log=l;st.session_state.results=[]
    p.progress(1.0);msg.empty();st.success(f"{len(docs)} actions loaded.")
if st.session_state.summary:
    s=st.session_state.summary
    a,b,c,d=st.columns(4);a.metric("Bills",s["bills_scanned"]);b.metric("Links",s["links_checked"]);c.metric("Loaded",s["sources_loaded"]);d.metric("Errors",s["source_errors"])
    with st.expander("Source diagnostics"):st.dataframe(pd.DataFrame(st.session_state.log),use_container_width=True)

st.divider();st.subheader("🎯 Step 2 — Qualitative relevance & classification")
st.markdown("**First:** RELEVANT = YES/NO. **Then, if relevant:** exactly one of Country Risk/Opportunity, Industry Risk/Opportunity, or Investment Risk/Opportunity.")
st.markdown("Scenarios: 🇨🇳 **CHINA-EXPORT** · 🇲🇽 **MEXICO-MANUFACTURE** · 🇺🇸 **US-MANUFACTURE**")
key=os.getenv("OPENAI_API_KEY","").strip()
if not key:st.warning("Add OPENAI_API_KEY to a local .env file before running Step 2.")
if st.button("🧠 Run Qualitative Classification"):
    if not st.session_state.docs:st.error("Run Step 1 first.")
    elif not key:st.error("OPENAI_API_KEY missing from .env")
    else:
        p=st.progress(0);msg=st.empty()
        def cb2(done,total,text):p.progress(min(done/max(total,1),1.0));msg.write(text)
        st.session_state.results=run_classification(st.session_state.docs,key,cb2)
        p.progress(1.0);msg.empty();st.success("Classification complete.")

if st.session_state.results:
    res=st.session_state.results; rel=[r for r in res if r.get("relevant")=="YES"]; non=[r for r in res if r.get("relevant")!="YES"]
    a,b,c=st.columns(3);a.metric("Reviewed",len(res));b.metric("Relevant",len(rel));c.metric("Not relevant",len(non))
    if rel:
        x=pd.DataFrame(rel)["primary_classification"].value_counts().reset_index();x.columns=["Classification","Actions"]
        st.plotly_chart(px.bar(x,x="Classification",y="Actions",color="Classification",color_discrete_sequence=px.colors.qualitative.Bold),use_container_width=True)
    st.subheader("✅ Relevant Congressional actions")
    for r in rel:
        with st.expander(f"{r['bill_number']} — {r['title']} | {r.get('primary_classification','')}"):
            st.write("**Country:**",r.get("country",""));st.write("**Policy stage:**",r.get("policy_stage",""));st.write("**Directness:**",r.get("directness",""))
            st.write("**Affected scenarios:**",", ".join(r.get("affected_scenarios",[])));st.write("**Mechanism:**",r.get("mechanism",""));st.write("**Analysis:**",r.get("analytical_summary",""))
            for e in r.get("evidence",[]):st.info(f"Evidence: {e.get('quote','')}\n\nWhy it matters: {e.get('why_it_matters','')}")
    with st.expander(f"⚪ Not relevant ({len(non)})"):
        for r in non:st.markdown(f"**{r['bill_number']} — {r['title']}**");st.write(r.get("analytical_summary",""));st.divider()
