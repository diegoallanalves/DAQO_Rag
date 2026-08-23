# DAQO Congressional RAG — GovTrack-only v4

## Evidence source
This version uses only the Excel column `GOVTRACK URL`.

The `Secondary Source - full text (congress.gov)` column is ignored by the application and can remain in the workbook without affecting retrieval.

## Flow
Excel → GOVTRACK URL → retrieve evidence → qualitative Gate analysis → RELEVANT YES/NO → Country / Industry / Investment → Risk / Opportunity → evidence-backed explanation.

## Run locally
1. Put `raw_data.xlsx` inside `data/`.
2. Create `.env` in the project root:
   `OPENAI_API_KEY=your_key_here`
3. Run:
   `pip install -r requirements.txt`
4. Start:
   `streamlit run app.py`

Never commit `.env` to GitHub.
