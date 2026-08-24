# DAQO Congressional RAG — Dual Source v6

Choose one source per run:
- **GovTrack Bill Page** → `GOVTRACK URL`
- **Congress.gov Full Bill Text** → `Secondary Source - full text (congress.gov)`

The original Excel column names are unchanged.

Flow: Excel → choose source → retrieve evidence → qualitative analysis → Relevant / Not Relevant / Analysis Failed → Country / Industry / Investment → Risk / Opportunity → evidence → Excel export.

v5 protections remain: failed AI calls are never counted as Not Relevant, and Excel export contains results, evidence, failures and methodology.

Congress.gov may block automated retrieval. Such cases are shown as `Source Retrieval Failed`; the app does not silently switch sources.

Local setup:
1. Put `raw_data.xlsx` in `data/`
2. Create `.env` with `OPENAI_API_KEY=your_key_here`
3. `pip install -r requirements.txt`
4. `streamlit run app.py`

Never commit `.env`.
