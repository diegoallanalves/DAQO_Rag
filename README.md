# DAQO Congressional RAG — Congress.gov-only v3

This version removes `GOVTRACK URL` from the RAG retrieval workflow.

## Research flow
Excel dataset → Congress.gov full text → evidence retrieval → qualitative Gate 0–4 analysis → RELEVANT YES/NO → Country / Industry / Investment → Risk / Opportunity → evidence-backed explanation.

## Workbook
Place your workbook at `data/raw_data.xlsx`.

You may delete the `GOVTRACK URL` column. Keep the Congress.gov full-text column, such as `Secondary Source - full text (congress.gov)`.

## Local run
Create `.env` in the project root:
`OPENAI_API_KEY=your_key_here`

Then run:
`pip install -r requirements.txt`
`streamlit run app.py`

## Important source behavior
Congress.gov can return HTTP 403 to automated page requests. This version intentionally does NOT silently fall back to GovTrack. Blocked documents are flagged transparently so the research record remains defensible.

Test a small sample before attempting the full dataset.
