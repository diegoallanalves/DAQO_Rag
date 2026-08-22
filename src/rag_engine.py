import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


def _safe(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def fetch_url_text(url: str, timeout: int = 35) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(
        ["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]
    ):
        tag.decompose()

    lines = []
    for line in soup.get_text("\n").splitlines():
        clean = re.sub(r"\s+", " ", line).strip()
        if clean:
            lines.append(clean)

    text = "\n".join(lines)

    if len(text) < 350:
        raise ValueError(
            "Too little readable text returned. The source may restrict automated access."
        )

    return text


def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 260) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end]

        if end < len(text):
            last_break = max(piece.rfind("\n"), piece.rfind(". "))
            if last_break > chunk_size * .6:
                end = start + last_break + 1
                piece = text[start:end]

        piece = piece.strip()

        if len(piece) >= 150:
            chunks.append(piece)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def _candidate_chunks(question: str, chunks: List[Dict], max_chunks: int = 10) -> List[Dict]:
    """
    Candidate retrieval only.

    This is deliberately NOT the final relevance decision.
    It is used only to select passages from long bills for qualitative LLM review.
    """
    q = question.lower()

    domain_terms = [
        "transformer", "transformers", "8504", "power", "electrical", "grid",
        "energy", "inverter", "solar", "semiconductor", "pcb", "battery",
        "china", "chinese", "prc", "mexico", "mexican", "united states",
        "import", "export", "tariff", "sanction", "procurement", "restriction",
        "ownership", "entity", "investment", "facility", "assembly", "factory",
        "manufacturing", "manufacture", "tax", "credit", "grant", "funding",
        "prohibit", "condition", "require", "shall", "duty", "origin"
    ]

    q_words = set(
        w for w in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'-]{2,}", q)
        if w not in {
            "the", "and", "for", "that", "this", "with", "from", "could", "would",
            "should", "have", "has", "are", "was", "were", "will", "its", "their",
            "they", "about", "what", "which", "how", "does", "can", "may", "might"
        }
    )

    scored = []

    for chunk in chunks:
        t = chunk["text"].lower()
        score = sum(1 for w in q_words if w in t)
        score += sum(1 for term in domain_terms if term in q and term in t)

        operative = ["shall", "prohibit", "require", "may not", "duty", "tax", "fund", "eligible", "ineligible", "condition"]
        score += sum(1 for term in operative if term in t)

        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:max_chunks]] or chunks[:max_chunks]


def build_document_store(
    df,
    column_map: Dict[str, str],
    progress_callback: Optional[Callable] = None,
) -> Tuple[List[Dict], Dict, List[Dict]]:
    documents = []
    link_log = []

    total = len(df)
    links_checked = 0
    sources_loaded = 0
    source_errors = 0

    for pos, (_, row) in enumerate(df.iterrows(), start=1):
        bill_number = _safe(row.get(column_map.get("bill_col"))) if column_map.get("bill_col") else ""
        title = _safe(row.get(column_map.get("title_col"))) if column_map.get("title_col") else ""

        if progress_callback:
            progress_callback(pos - 1, total, f"Reading {bill_number} — {title[:65]}")

        urls = []

        if column_map.get("govtrack_col"):
            u = _safe(row.get(column_map["govtrack_col"]))
            if u:
                urls.append(("GovTrack", u))

        if column_map.get("congress_col"):
            u = _safe(row.get(column_map["congress_col"]))
            if u:
                urls.append(("Congress.gov", u))

        source_texts = []
        source_names = []
        source_urls = []

        for source_name, url in urls:
            links_checked += 1

            try:
                text = fetch_url_text(url)
                source_texts.append({"source": source_name, "url": url, "text": text})
                source_names.append(source_name)
                source_urls.append(url)
                sources_loaded += 1

                link_log.append({
                    "Bill": bill_number,
                    "Source": source_name,
                    "Result": "✅ Loaded",
                    "URL": url,
                })

            except Exception as exc:
                source_errors += 1

                link_log.append({
                    "Bill": bill_number,
                    "Source": source_name,
                    "Result": f"⚠️ Failed: {str(exc)[:90]}",
                    "URL": url,
                })

        if source_texts:
            chunks = []

            for src in source_texts:
                for i, piece in enumerate(chunk_text(src["text"]), start=1):
                    chunks.append({
                        "source": src["source"],
                        "url": src["url"],
                        "chunk_id": i,
                        "text": piece,
                    })

            documents.append({
                "bill_number": bill_number,
                "title": title,
                "status": _safe(row.get(column_map.get("status_col"))) if column_map.get("status_col") else "",
                "country": _safe(row.get(column_map.get("country_col"))) if column_map.get("country_col") else "",
                "bill_type": _safe(row.get(column_map.get("type_col"))) if column_map.get("type_col") else "",
                "party": _safe(row.get(column_map.get("party_col"))) if column_map.get("party_col") else "",
                "sponsor": _safe(row.get(column_map.get("sponsor_col"))) if column_map.get("sponsor_col") else "",
                "congress": _safe(row.get(column_map.get("congress_session_col"))) if column_map.get("congress_session_col") else "",
                "sources": source_names,
                "source_urls": source_urls,
                "chunks": chunks,
            })

        if progress_callback:
            progress_callback(pos, total, f"Finished {bill_number}")

    summary = {
        "bills_scanned": total,
        "links_checked": links_checked,
        "sources_loaded": sources_loaded,
        "source_errors": source_errors,
        "documents_created": len(documents),
    }

    return documents, summary, link_log


def _llm_json(api_key: str, system: str, user: str, model: str = "gpt-4o-mini") -> Dict:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        temperature=.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    return json.loads(response.choices[0].message.content)


def run_gate_analysis(
    documents: List[Dict],
    questions: List[str],
    api_key: str,
    progress_callback: Optional[Callable] = None,
) -> Tuple[List[Dict], List[Dict]]:
    results = []
    kept_docs = []
    total = len(documents)

    system = """
You are conducting defensible LLM-assisted qualitative content analysis of U.S. Congressional actions.

COMPANY CONTEXT
DAQO is treated as a manufacturer of transformers / electrical power equipment that may manufacture in China or Mexico,
sell into the U.S. market, and potentially invest in U.S. manufacturing or assembly.

PRODUCT / INDUSTRY CONTEXT
Relevant product and industry concepts include transformers, HTS 8504 power-electrical goods, grid equipment,
power electronics, inverters, energy equipment, semiconductors and important electrical components.
A bill mentioning China is NOT automatically relevant.

EVIDENCE STANDARD
Gate 0: Use operative sections. Do not rely on the title, findings, purpose statements, or "whereas" clauses alone.

GATE 1 — MECHANISM
Ask whether the action obliges, prohibits, taxes, funds, restricts, conditions, or otherwise creates an operative mechanism.
A study/report/strategy/commission normally fails.
Exception: a non-binding resolution can still survive if its subject is a credible leading indicator on one of the DAQO channels.

GATE 2 — DAQO REACH
A mechanism reaches DAQO in either of two ways:
A) PRODUCT / INDUSTRY: it reaches transformers, HTS 8504 goods, power/electrical equipment, grid/energy equipment,
or materially relevant components/supply chains.
B) ENTITY / COUNTRY: it reaches PRC-organised, Chinese-owned, Chinese-origin, or similarly defined entities/goods regardless of product.
If it fails both, it should normally be excluded.

GATE 3 — TRIGGER / PREDICATE
DAQO must be capable of satisfying the factual predicate.
Example: "any entity organised under PRC law" can plausibly reach a PRC-organised company.
Example: "persons militarising South China Sea features" does not plausibly describe a transformer manufacturer.
Respect carve-outs. If importation of goods is expressly excluded from sanctions, shipment risk should not be claimed unless another section independently reaches it.

GATE 4 — BUSINESS CHANNEL
Classify the primary effect:
- Country: Chinese origin or ownership.
- Industry: energy, grid, transformers, inverters, solar, semiconductors, electrical equipment. Industry beats Country if the provision directly names the product/industry.
- Investment: constrains or changes the cost/feasibility of establishing a U.S. or Mexico plant.
- Opportunity: DAQO benefits and is actually eligible.
If a benefit excludes PRC-owned firms, it is NOT an opportunity for DAQO; treat it as an investment/eligibility risk.

RESEARCH CODING
Also assign:
1. Bucket: tariffs, forced labour, grid equipment, investment, sanctions, procurement, tax/incentive, ownership/entity restriction, trade/import, supply chain, or other.
2. Legislative outcome: Became law / Moved / Died-or-unclear.
3. DAQO actionability: Yes / Yes-but-expensive / No.
4. Binding/signal: Binding policy action / Potentially binding / Political signal.

MANDATORY QUESTIONS
Q1 China manufacturing + U.S. market.
Q2 Mexico manufacturing + U.S. market.
Q3 U.S. manufacturing/assembly investment.

VERDICTS
YES = clear qualitative evidence.
POSSIBLE = plausible or indirect effect, but uncertain/conditional.
NO = operative evidence does not meaningfully support the connection.

KEEP RULE
Keep the document if at least one mandatory question is YES or POSSIBLE AND the gate analysis shows a plausible DAQO pathway.
A pure title/keyword match is not enough.

Return ONLY valid JSON:
{
  "bucket": "one bucket",
  "legislative_outcome": "Became law | Moved | Died-or-unclear",
  "daqo_actionability": "Yes | Yes-but-expensive | No",
  "binding_signal": "Binding policy action | Potentially binding | Political signal",
  "primary_channel": "Country | Industry | Investment | Opportunity | None",
  "gates": [
    {"gate":"Gate 0","pass":true,"result":"PASS","reason":"short"},
    {"gate":"Gate 1","pass":true,"result":"PASS","reason":"short"},
    {"gate":"Gate 2","pass":true,"result":"PASS","reason":"short"},
    {"gate":"Gate 3","pass":true,"result":"PASS","reason":"short"},
    {"gate":"Gate 4","pass":true,"result":"PASS","reason":"short"}
  ],
  "question_assessments": [
    {
      "question_number": 1,
      "verdict": "YES | POSSIBLE | NO",
      "short_reason": "short",
      "business_meaning": "plain English"
    }
  ],
  "evidence_quotes": ["short operative evidence snippet 1", "snippet 2"],
  "keep_document": true,
  "overall_reason": "short overall reason"
}
""".strip()

    combined_q = "\n".join(f"Q{i+1}: {q}" for i, q in enumerate(questions))

    for pos, doc in enumerate(documents, start=1):
        if progress_callback:
            progress_callback(pos - 1, total, f"Running gates on {doc['bill_number']} — {doc['title'][:60]}")

        retrieval_question = (
            "DAQO transformer manufacturer China Mexico United States market investment "
            "grid electrical power equipment ownership import export tariff sanctions opportunity "
            + " ".join(questions)
        )

        chunks = _candidate_chunks(retrieval_question, doc["chunks"], max_chunks=12)

        evidence = "\n\n".join(
            f"""[EVIDENCE {i}]
Source: {c['source']}
URL: {c['url']}
Text:
{c['text']}
"""
            for i, c in enumerate(chunks, start=1)
        )

        user = f"""
BILL METADATA
Bill: {doc['bill_number']}
Title: {doc['title']}
Type: {doc['bill_type']}
Status: {doc['status']}
Country tag: {doc['country']}
Congress: {doc.get('congress','')}

MANDATORY QUESTIONS
{combined_q}

RETRIEVED BILL TEXT
{evidence}

Apply Gate 0 through Gate 4 carefully.
Do not keep the bill just because its title or findings sound relevant.
""".strip()

        try:
            analysis = _llm_json(api_key, system, user)
        except Exception as exc:
            analysis = {
                "bucket": "Other",
                "legislative_outcome": "Died-or-unclear",
                "daqo_actionability": "No",
                "binding_signal": "Political signal",
                "primary_channel": "None",
                "gates": [],
                "question_assessments": [],
                "evidence_quotes": [],
                "keep_document": False,
                "overall_reason": f"Analysis failed: {exc}",
            }

        verdicts = {str(x.get("verdict", "")).upper() for x in analysis.get("question_assessments", [])}
        gate2 = next((g for g in analysis.get("gates", []) if g.get("gate") == "Gate 2"), {})
        gate3 = next((g for g in analysis.get("gates", []) if g.get("gate") == "Gate 3"), {})
        plausible_path = bool(gate2.get("pass")) and bool(gate3.get("pass"))
        keep = bool({"YES", "POSSIBLE"}.intersection(verdicts)) and plausible_path

        record = {
            "bill_number": doc["bill_number"],
            "title": doc["title"],
            "status": doc["status"],
            "country": doc["country"],
            "sources_used": doc["sources"],
            "bucket": analysis.get("bucket", "Other"),
            "legislative_outcome": analysis.get("legislative_outcome", "Died-or-unclear"),
            "daqo_actionability": analysis.get("daqo_actionability", "No"),
            "binding_signal": analysis.get("binding_signal", "Political signal"),
            "primary_channel": analysis.get("primary_channel", "None"),
            "gates": analysis.get("gates", []),
            "question_assessments": analysis.get("question_assessments", []),
            "evidence_quotes": analysis.get("evidence_quotes", []),
            "keep_document": keep,
            "overall_reason": analysis.get("overall_reason", ""),
        }

        results.append(record)
        if keep:
            kept_docs.append(doc)
        if progress_callback:
            progress_callback(pos, total, f"Finished {doc['bill_number']}")

    return results, kept_docs


def run_custom_filter(
    documents: List[Dict],
    question: str,
    api_key: str,
    progress_callback: Optional[Callable] = None,
) -> Tuple[List[Dict], List[Dict]]:
    results = []
    kept_docs = []
    total = len(documents)

    system = """
You are applying a qualitative policy filter to Congressional documents that already survived a DAQO transformer-industry gate.

DAQO CONTEXT
DAQO is treated as a transformer / electrical power equipment manufacturer with possible China and Mexico production,
U.S. market exposure, and potential U.S. investment.

Do not use a numeric score.

Use operative evidence and classify:
YES = clear meaningful evidence supports the filter question.
POSSIBLE = plausible/conditional connection.
NO = supplied evidence does not meaningfully support the filter question.

Also classify the main channel:
Country | Industry | Investment | Opportunity | Trade | Supply chain | Compliance | Other

DAQO actionability:
Yes | Yes-but-expensive | No

Keep YES or POSSIBLE. Exclude NO.

Return ONLY valid JSON:
{
  "verdict": "YES | POSSIBLE | NO",
  "short_reason": "short reason",
  "business_meaning": "plain-English implication",
  "channel": "one channel",
  "daqo_actionability": "Yes | Yes-but-expensive | No",
  "evidence_quotes": ["short operative snippet 1", "snippet 2"],
  "keep_document": true
}
""".strip()

    for pos, doc in enumerate(documents, start=1):
        if progress_callback:
            progress_callback(pos - 1, total, f"Applying custom filter to {doc['bill_number']}")

        chunks = _candidate_chunks(question, doc["chunks"], max_chunks=10)
        evidence = "\n\n".join(
            f"""[EVIDENCE {i}]
Source: {c['source']}
URL: {c['url']}
Text:
{c['text']}
"""
            for i, c in enumerate(chunks, start=1)
        )

        user = f"""
BILL
{doc['bill_number']} — {doc['title']}
Type: {doc['bill_type']}
Status: {doc['status']}

QUALITATIVE FILTER
{question}

EVIDENCE
{evidence}

Decide whether this document remains in the research set.
""".strip()

        try:
            analysis = _llm_json(api_key, system, user)
        except Exception as exc:
            analysis = {
                "verdict": "NO",
                "short_reason": f"Analysis failed: {exc}",
                "business_meaning": "",
                "channel": "Other",
                "daqo_actionability": "No",
                "evidence_quotes": [],
                "keep_document": False,
            }

        verdict = str(analysis.get("verdict", "NO")).upper()
        keep = verdict in {"YES", "POSSIBLE"}

        record = {
            "bill_number": doc["bill_number"],
            "title": doc["title"],
            "verdict": verdict,
            "short_reason": analysis.get("short_reason", ""),
            "business_meaning": analysis.get("business_meaning", ""),
            "channel": analysis.get("channel", "Other"),
            "daqo_actionability": analysis.get("daqo_actionability", "No"),
            "evidence_quotes": analysis.get("evidence_quotes", []),
            "keep_document": keep,
        }

        results.append(record)
        if keep:
            kept_docs.append(doc)
        if progress_callback:
            progress_callback(pos, total, f"Finished {doc['bill_number']}")

    return results, kept_docs


def make_final_summary(
    documents: List[Dict],
    mandatory_questions: List[str],
    custom_questions: List[str],
    api_key: str,
) -> Dict:
    system = """
Summarize a final qualitative Congressional policy document set for DAQO, a transformer/electrical power equipment manufacturer.

Return ONLY valid JSON:
{
  "headline": "short headline",
  "summary": "3 to 5 simple sentences",
  "risks": ["risk 1", "risk 2"],
  "opportunities": ["opportunity 1", "opportunity 2"],
  "channels": ["Country", "Industry", "Investment"],
  "recommended_action": "one practical next step"
}

Do not invent facts. Preserve uncertainty.
""".strip()

    docs_text = "\n".join(
        f"- {d['bill_number']} | {d['title']} | {d['status']} | {d['country']}"
        for d in documents
    )

    user = f"""
MANDATORY QUESTIONS
{chr(10).join('- ' + q for q in mandatory_questions)}

ADDITIONAL FILTERS
{chr(10).join('- ' + q for q in custom_questions) if custom_questions else 'None'}

FINAL DOCUMENTS
{docs_text}
""".strip()

    return _llm_json(api_key, system, user)


def save_results_csv(documents: List[Dict], output_path: Path) -> Path:
    rows = []

    for d in documents:
        rows.append({
            "Bill Number": d["bill_number"],
            "Title": d["title"],
            "Status": d.get("status", ""),
            "Country": d.get("country", ""),
            "Bill Type": d.get("bill_type", ""),
            "Party": d.get("party", ""),
            "Sponsor": d.get("sponsor", ""),
            "Congress": d.get("congress", ""),
            "Sources": ", ".join(d.get("sources", [])),
            "Source URLs": " | ".join(d.get("source_urls", [])),
        })

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
