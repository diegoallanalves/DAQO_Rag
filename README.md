# DAQO Congressional RAG — Smart Pre-Screen v9

This version is optimized for large Congressional datasets.

## Pipeline

1. **Congress filter**
2. **Country filter**
3. **Evidence source selection**
4. **Local NLP / keyword pre-screen** — no OpenAI cost
5. **URL and duplicate quality checks**
6. **Local relevance ranking**
7. **Processing strategy**
   - Highest relevance first
   - Random sample
   - All filtered candidates
   - Resume unfinished / failed
8. **Evidence retrieval**
9. **OpenAI qualitative classification**
10. **Checkpoint results for resume**
11. **Excel export**

## Why it is faster

OpenAI no longer receives every row from the filtered dataset. Low-relevance,
duplicate, missing-URL, and invalid-URL records can be removed before retrieval.

## Local setup

Create `.env`:

```text
OPENAI_API_KEY=your_key_here
```

Install and run:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

`.env` is ignored and is not included in this ZIP.
