import json
import re
from typing import Dict, List, Tuple, Callable, Optional

import requests
from bs4 import BeautifulSoup


def _safe(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def fetch_url_text(url: str, timeout: int = 35) -> str:
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
        tag.decompose()
    text = "\n".join(re.sub(r"\s+", " ", x).strip() for x in soup.get_text("\n").splitlines() if x.strip())
    if len(text) < 350:
        raise ValueError("Too little readable text returned")
    return text


def chunk_text(text: str, chunk_size: int = 1900, overlap: int = 250) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    out, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if len(piece) >= 150:
            out.append(piece)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return out


def build_document_store(df, column_map: Dict[str, str], progress_callback: Optional[Callable] = None) -> Tuple[List[Dict], Dict, List[Dict]]:
    documents, log = [], []
    loaded = errors = checked = 0
    total = len(df)
    for pos, (_, row) in enumerate(df.iterrows(), 1):
        bill = _safe(row.get(column_map.get("bill_col"))) if column_map.get("bill_col") else ""
        title = _safe(row.get(column_map.get("title_col"))) if column_map.get("title_col") else ""
        if progress_callback:
            progress_callback(pos - 1, total, f"Reading {bill} — {title[:60]}")
        urls = []
        for key, name in [("govtrack_col", "GovTrack"), ("congress_col", "Congress.gov")]:
            col = column_map.get(key)
            if col:
                u = _safe(row.get(col))
                if u:
                    urls.append((name, u))
        chunks, sources, source_urls = [], [], []
        for source, url in urls:
            checked += 1
            try:
                text = fetch_url_text(url)
                loaded += 1
                sources.append(source); source_urls.append(url)
                for i, piece in enumerate(chunk_text(text), 1):
                    chunks.append({"source": source, "url": url, "chunk_id": i, "text": piece})
                log.append({"Bill": bill, "Source": source, "Result": "Loaded", "URL": url})
            except Exception as exc:
                errors += 1
                log.append({"Bill": bill, "Source": source, "Result": f"Failed: {str(exc)[:90]}", "URL": url})
        if chunks:
            documents.append({
                "bill_number": bill, "title": title,
                "status": _safe(row.get(column_map.get("status_col"))) if column_map.get("status_col") else "",
                "country": _safe(row.get(column_map.get("country_col"))) if column_map.get("country_col") else "",
                "bill_type": _safe(row.get(column_map.get("type_col"))) if column_map.get("type_col") else "",
                "congress": _safe(row.get(column_map.get("congress_session_col"))) if column_map.get("congress_session_col") else "",
                "sources": sources, "source_urls": source_urls, "chunks": chunks,
            })
        if progress_callback:
            progress_callback(pos, total, f"Finished {bill}")
    return documents, {"bills_scanned": total, "links_checked": checked, "sources_loaded": loaded, "source_errors": errors, "documents_created": len(documents)}, log


def _candidate_chunks(doc: Dict, max_chunks: int = 14) -> List[Dict]:
    terms = ["transformer", "8504", "electrical", "grid", "transmission", "distribution", "substation", "china", "chinese", "prc", "mexico", "usmca", "tariff", "duty", "import", "export", "forced labor", "sanction", "investment", "cfIUS", "ownership", "manufacturing", "procurement", "domestic content", "buy america", "shall", "prohibit", "require", "eligible", "fund", "credit"]
    scored = []
    for c in doc["chunks"]:
        t = c["text"].lower()
        score = sum(2 if x in ["transformer", "8504", "electrical", "grid", "cfIUS"] else 1 for x in terms if x.lower() in t)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:max_chunks]]


def _llm_json(api_key: str, system: str, user: str, model: str = "gpt-4o-mini") -> Dict:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    r = client.chat.completions.create(model=model, temperature=0.05, response_format={"type": "json_object"}, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return json.loads(r.choices[0].message.content)


SYSTEM_PROMPT = r'''
You are performing LLM-assisted qualitative content analysis for the research project “DAQO’s Strategic Approach to Serving the U.S. Market”. Follow these rules exactly.

CONTEXT
DAQO is a Chinese electrical-equipment manufacturer. Main product: transformers, HTS heading 8504. Strategic scenarios: CHINA-EXPORT (manufacture in China, export to U.S.); MEXICO-MANUFACTURE (manufacture in Mexico, serve U.S., including USMCA advantages); US-MANUFACTURE (establish U.S. manufacturing/assembly).
IMPORTANT: Country is supplied in dataset metadata. NEVER infer, replace or classify country.

FIRST: RELEVANCE
Decide RELEVANT YES/NO. Relevant means meaningful evidence of a risk or opportunity affecting DAQO's strategic environment. Relevance does NOT require enactment, binding force, success, direct naming of DAQO/transformers, or immediate economic effect. Proposed, unsuccessful and non-binding actions may be relevant as emerging policy or political pressure. If NO, primary_classification must be NOT RELEVANT and stop substantive classification.

IF RELEVANT: EXACTLY ONE PRIMARY CLASSIFICATION
COUNTRY RISK | COUNTRY OPPORTUNITY | INDUSTRY RISK | INDUSTRY OPPORTUNITY | INVESTMENT RISK | INVESTMENT OPPORTUNITY.
Use the PRINCIPAL mechanism. Secondary mechanisms belong only in summary.

COUNTRY RISK: broader country-linked political/regulatory/economic/trade/supply-chain/geopolitical risk: tariffs, duties, origin treatment, broad imports, forced-labor trade policy, broad sanctions, restrictions on country firms, decoupling, diversification, elevated security/economic treatment.
COUNTRY OPPORTUNITY: broader geographic/trade/manufacturing-location advantage: tariff relief, favorable trade treatment, USMCA, North American manufacturing, Mexico advantages, relocation from China.
INDUSTRY RISK: ONLY direct connection to DAQO's actual industry/product/inputs/market: transformers/8504, substations, switchgear, transmission/distribution, grid infrastructure, electrical equipment, relevant renewable/new-energy electrical infrastructure/components/procurement. Energy alone is not enough.
INDUSTRY OPPORTUNITY: directly creates/expands demand or market conditions for transformers/electrical equipment: grid modernization/buildout, transmission, substations, power infrastructure, relevant renewable/new-energy infrastructure.
INVESTMENT RISK: makes establishment/ownership/acquisition/finance/expansion of relevant manufacturing more difficult: CFIUS, investment screening, Chinese acquisition/ownership restrictions, capital/eligibility/land/JV restrictions.
INVESTMENT OPPORTUNITY: makes manufacturing investment more attractive/feasible/financeable: nearshoring/relocation/manufacturing/FDI incentives, financing/support. Build/Buy America can be investment opportunity when U.S. establishment is the principal pathway; if directly about transformer/electrical procurement, use INDUSTRY.

BOUNDARY
5G/telecom, EV/batteries, semiconductors, pharma, aerospace, unrelated military equipment are NOT DAQO industry. They may still indicate COUNTRY RISK if they meaningfully evidence broader country treatment. Political criticism alone without meaningful trade/investment/manufacturing/supply-chain/economic/DAQO-market connection is NOT RELEVANT.

NON-BINDING/SENTIMENT
Never reject merely because non-binding/unsuccessful/proposed. Sentiment can be relevant when it evidences pressure or emerging direction on economic restrictions, investment, supply chains, trade, domestic/North American manufacturing, or DAQO's market.

FORCED LABOR / COVID
Human-rights condemnation alone: not relevant. Forced-labor action tied to economic/trade/import/sanctions: usually COUNTRY RISK unless transformer/electrical-specific. COVID blame alone: not relevant; sanctions, Chinese investment/goods restrictions, diversification, decoupling, reshoring or company restrictions can be COUNTRY RISK.

POLICY STAGE (separate field)
Proposed | Passed one chamber | Passed Congress | Enacted | Implemented / Enforced | Non-binding resolution | Other.
Do not use stage to determine relevance by itself.

DIRECTNESS
DIRECT | EMERGING | SENTIMENT / POLITICAL PRESSURE.

AFFECTED SCENARIOS
Select only meaningfully affected scenarios: CHINA-EXPORT | MEXICO-MANUFACTURE | US-MANUFACTURE. Do not automatically assign all three.

EVIDENCE
Prefer substantive/operative provisions. For non-binding actions, substantive calls/sense/resolution language can evidence sentiment. Do not rely on title alone. Do not claim causation between early and later actions unless evidence establishes it.

ANALYTICAL SUMMARY
For every action, 2–5 concise substantive sentences: what it does; why keyword screening may have flagged it where identifiable; why relevant/not; why classification; mechanism; affected scenario(s). For relevant actions show pathway: CONGRESSIONAL ACTION → POLICY/ECONOMIC MECHANISM → DAQO RISK/OPPORTUNITY → AFFECTED SCENARIO.

Return ONLY JSON with this exact shape:
{
 "relevant":"YES|NO",
 "primary_classification":"COUNTRY RISK|COUNTRY OPPORTUNITY|INDUSTRY RISK|INDUSTRY OPPORTUNITY|INVESTMENT RISK|INVESTMENT OPPORTUNITY|NOT RELEVANT",
 "policy_stage":"Proposed|Passed one chamber|Passed Congress|Enacted|Implemented / Enforced|Non-binding resolution|Other",
 "directness":"DIRECT|EMERGING|SENTIMENT / POLITICAL PRESSURE",
 "affected_scenarios":["CHINA-EXPORT","MEXICO-MANUFACTURE","US-MANUFACTURE"],
 "mechanism":"short description",
 "evidence":[{"quote":"short evidence excerpt","source":"GovTrack|Congress.gov","why_it_matters":"short explanation"}],
 "analytical_summary":"2-5 sentences",
 "confidence":"HIGH|MEDIUM|LOW"
}
'''


def run_classification(documents: List[Dict], api_key: str, progress_callback: Optional[Callable] = None) -> List[Dict]:
    results = []
    total = len(documents)
    for pos, doc in enumerate(documents, 1):
        if progress_callback:
            progress_callback(pos - 1, total, f"Classifying {doc['bill_number']} — {doc['title'][:55]}")
        chunks = _candidate_chunks(doc)
        evidence = "\n\n".join(f"[PASSAGE {i}] Source={c['source']} URL={c['url']}\n{c['text']}" for i, c in enumerate(chunks, 1))
        user = f'''DATASET METADATA\nBill: {doc['bill_number']}\nTitle: {doc['title']}\nBill type: {doc['bill_type']}\nStatus: {doc['status']}\nCountry (USE AS GIVEN; DO NOT INFER): {doc['country']}\nCongress: {doc['congress']}\n\nRETRIEVED DOCUMENT PASSAGES\n{evidence}\n\nClassify this one Congressional action under the research rules. Exactly one primary classification if relevant.'''
        try:
            a = _llm_json(api_key, SYSTEM_PROMPT, user)
        except Exception as exc:
            a = {"relevant":"NO", "primary_classification":"NOT RELEVANT", "policy_stage":"Other", "directness":"EMERGING", "affected_scenarios":[], "mechanism":"Analysis failed", "evidence":[], "analytical_summary":f"Analysis failed: {exc}", "confidence":"LOW"}
        relevant = str(a.get("relevant", "NO")).upper() == "YES"
        if not relevant:
            a["primary_classification"] = "NOT RELEVANT"
            a["affected_scenarios"] = []
        a.update({"bill_number": doc["bill_number"], "title": doc["title"], "country": doc["country"], "status": doc["status"], "bill_type": doc["bill_type"], "congress": doc["congress"], "source_urls": doc["source_urls"]})
        results.append(a)
        if progress_callback:
            progress_callback(pos, total, f"Finished {doc['bill_number']}")
    return results
