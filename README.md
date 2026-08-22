# DAQO Congressional Risk & Opportunity RAG

A qualitative RAG (Retrieval-Augmented Generation) research application for evaluating U.S. Congressional actions against DAQO's transformer and electrical power equipment business.

## DAQO context

The app treats DAQO as a manufacturer of:

- transformers;
- HTS 8504 power/electrical goods;
- grid and electrical power equipment;
- related power-electronics supply chains.

The research is therefore not simply a China keyword search.

A bill must plausibly reach DAQO through:

- product / industry;
- country / ownership;
- investment;
- or a genuine opportunity that DAQO is eligible to use.

## Gate methodology

### Gate 0 — Evidence
Use operative sections. Do not rely on the title, findings, purpose language, or whereas clauses as sufficient evidence by themselves.

### Gate 1 — Mechanism
Look for something that obliges, prohibits, taxes, funds, restricts, or conditions. Studies, reports, strategies and commissions normally fail. A non-binding resolution may still survive when it is a credible leading indicator for one of the DAQO channels.

### Gate 2 — Reach DAQO
Two routes:

**Product / industry** — transformers, HTS 8504, grid equipment, energy equipment, power electronics, and relevant components.

**Entity / country** — PRC-organised firms, Chinese-owned firms, Chinese-origin goods, or similarly defined entities.

If both fail, exclude the bill.

### Gate 3 — Trigger
DAQO must actually be able to satisfy the predicate. Carve-outs must be respected.

### Gate 4 — Channel

- **Country** — Chinese origin / ownership.
- **Industry** — energy, grid, transformers, inverters, solar, semiconductors, electrical equipment.
- **Investment** — affects the cost or feasibility of establishing a U.S. or Mexico manufacturing facility.
- **Opportunity** — DAQO receives a benefit and is actually eligible.

If PRC-owned firms are excluded from a benefit, the app treats it as an investment/eligibility risk rather than an opportunity.

## Three mandatory questions

1. Could this Congressional action affect DAQO's ability to manufacture in China and serve the U.S. market?
2. Could this Congressional action affect DAQO's ability to manufacture in Mexico and serve the U.S. market?
3. Could this Congressional action affect DAQO's ability to invest in a manufacturing facility or assembly line in the U.S.?

## Coding framework

Every surviving bill receives:

1. **What is it about?** — tariffs, forced labour, grid equipment, investment, sanctions, procurement, tax/incentive, ownership/entity restriction, trade/import, supply chain, etc.
2. **Did it go anywhere?** — Became law / Moved / Died-or-unclear.
3. **Can DAQO do anything about it?** — Yes / Yes-but-expensive / No.

## Sequential qualitative RAG funnel

```text
All Congressional documents
        ↓
Gate 0–4 analysis
        ↓
3 mandatory DAQO questions
        ↓
Save plausible bills
        ↓
Custom qualitative Filter 2
        ↓
Save survivors
        ↓
Custom Filter 3
        ↓
Save survivors
        ↓
Custom Filter 4
        ↓
Save survivors
        ↓
Custom Filter 5
        ↓
Final document set
```

Each custom filter only receives documents that survived the previous stage.

## Data file

The app expects your workbook here:

```text
data/raw_data.xlsx
```

The workbook is intentionally not included in this repository.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your OpenAI key to `.env`:

```text
OPENAI_API_KEY=your_real_key_here
```

Never commit `.env` to GitHub.

## Run

```powershell
streamlit run app.py
```

## Methodological note

The final relevance judgement is qualitative. The app does not retain or exclude bills based on a numeric similarity threshold.

A lightweight passage-selection helper is used only to reduce long bill text into a manageable evidence packet for LLM review. The actual decision is based on the pre-specified Gate 0–4 criteria and the three DAQO questions.

This makes the workflow easier to document, validate against a human-coded sample, and defend as LLM-assisted qualitative content analysis.
