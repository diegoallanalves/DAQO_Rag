# DAQO Congressional RAG — GovTrack-only v5

New in v5:
- Failed API/model calls are classified as ANALYSIS FAILED, never Not Relevant.
- Dashboard shows Reviewed / Relevant / Not Relevant / Analysis Failed.
- Excel export includes All Results, Relevant, Not Relevant, Evidence, Failed Analysis, and Methodology.
- GovTrack remains the only evidence URL source.

Run locally:
1. Put raw_data.xlsx in data/.
2. Create .env with OPENAI_API_KEY=your_key_here
3. pip install -r requirements.txt
4. streamlit run app.py

Never commit .env.
