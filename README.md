# DAQO Congressional RAG — Local Test v2

## Run locally
1. Put `raw_data.xlsx` in `data/`.
2. Create `.env` in the project root:
   `OPENAI_API_KEY=your_key`
3. Install:
   `pip install -r requirements.txt`
4. Run:
   `streamlit run app.py`

## Retrieval behavior
The app checks both GovTrack and Congress.gov URLs from the Excel file.
Congress.gov can return HTTP 403 to automated requests even when the URL works in a browser.
The app now labels this as **Blocked by source — fallback used**, rather than a true application error.
If GovTrack loads, that bill remains ready for analysis. The original Congress.gov URL is retained for traceability.

Start with 3–10 bills during testing before running the full workbook.
