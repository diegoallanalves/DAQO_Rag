# DAQO Congressional RAG — Simple Congress + Country Filter

Simple workflow:

1. Select one or more Congress sessions.
2. Select one or more countries (China, Mexico, or both).
3. Choose GovTrack or Congress.gov as the evidence source.
4. Choose how many filtered actions to scan.
5. Build the evidence base.
6. Run qualitative AI classification.
7. Export the results to Excel.

There is no smart pre-screening, relevance-ranking, checkpoint, or processing-strategy layer.

The `.env` file is intentionally excluded. Keep `OPENAI_API_KEY` local or configure it in Streamlit Secrets.
