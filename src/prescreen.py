
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


TOPIC_GROUPS = {
    "China / PRC": [
        "china", "chinese", "prc", "people's republic of china",
        "ccp", "communist party of china",
    ],
    "Transformers / Grid / Electrical": [
        "transformer", "transformers", "hts 8504", "8504",
        "grid", "transmission", "distribution", "substation",
        "switchgear", "electrical equipment", "power equipment",
    ],
    "Tariffs / Trade / Imports": [
        "tariff", "tariffs", "duty", "duties", "trade restriction",
        "import restriction", "imports", "customs", "normal trade relations",
        "most favored nation", "most-favored-nation",
    ],
    "Sanctions / Export Controls": [
        "sanction", "sanctions", "export control", "export controls",
        "entity list", "restricted entity", "designation",
    ],
    "Supply Chain / Reshoring": [
        "supply chain", "reshoring", "onshoring", "nearshoring",
        "domestic manufacturing", "domestic content", "buy america",
        "buy american", "reduce reliance", "reduce dependence",
    ],
    "Investment / CFIUS / Ownership": [
        "cfius", "foreign investment", "investment restriction",
        "foreign ownership", "chinese-owned", "prc-owned",
        "covered transaction", "acquisition",
    ],
    "Forced Labor / Uyghur": [
        "forced labor", "uyghur", "xinjiang", "uflpa",
        "forced labour",
    ],
    "Mexico / USMCA": [
        "mexico", "mexican", "usmca", "north american",
        "north america", "nearshore", "nearshoring",
    ],
    "Semiconductors / Critical Inputs": [
        "semiconductor", "semiconductors", "chip", "chips",
        "critical mineral", "critical minerals", "battery",
        "pcb", "printed circuit board",
    ],
    "Solar / Energy": [
        "solar", "photovoltaic", "pv", "polysilicon",
        "renewable energy", "energy infrastructure",
    ],
}


def normalize_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def valid_http_url(value: str) -> bool:
    text = normalize_text(value)
    if not text:
        return False
    try:
        parsed = urlparse(text)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def row_text(row: pd.Series, columns: list[str]) -> str:
    return " ".join(normalize_text(row.get(c)) for c in columns if c).lower()


def topic_hits(text: str, selected_topics: list[str]) -> tuple[list[str], list[str]]:
    groups, terms = [], []
    for topic in selected_topics:
        matched = [t for t in TOPIC_GROUPS.get(topic, []) if t in text]
        if matched:
            groups.append(topic)
            terms.extend(matched)
    # preserve order / remove duplicates
    return list(dict.fromkeys(groups)), list(dict.fromkeys(terms))


def local_relevance_score(
    row: pd.Series,
    text_columns: list[str],
    selected_topics: list[str],
    source_col: str | None,
) -> tuple[int, list[str], list[str], str]:
    """
    Transparent 0-100 local screening score.
    No API call is made here.
    """
    text = row_text(row, text_columns)
    groups, terms = topic_hits(text, selected_topics)

    score = 0
    # Topic breadth + explicit keyword evidence
    score += min(len(groups) * 15, 60)
    score += min(len(terms) * 3, 24)

    # Higher weighting for DAQO's core subject.
    core_terms = [
        "transformer", "8504", "electrical equipment", "grid",
        "china", "chinese", "prc", "mexico", "usmca",
    ]
    score += min(sum(1 for t in core_terms if t in text) * 4, 16)

    source_status = "Valid URL"
    if source_col:
        url = normalize_text(row.get(source_col))
        if not url:
            source_status = "Missing URL"
            score = 0
        elif not valid_http_url(url):
            source_status = "Invalid URL"
            score = 0

    return min(score, 100), groups, terms, source_status


def band(score: int) -> str:
    if score >= 70:
        return "Very High"
    if score >= 50:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"


def prescreen_dataframe(
    df: pd.DataFrame,
    *,
    text_columns: list[str],
    selected_topics: list[str],
    source_col: str | None,
    bill_col: str | None,
) -> pd.DataFrame:
    out = df.copy()

    scores, groups_all, terms_all, statuses = [], [], [], []
    for _, row in out.iterrows():
        score, groups, terms, status = local_relevance_score(
            row, text_columns, selected_topics, source_col
        )
        scores.append(score)
        groups_all.append("; ".join(groups))
        terms_all.append("; ".join(terms))
        statuses.append(status)

    out["PreScreen Score"] = scores
    out["PreScreen Band"] = [band(s) for s in scores]
    out["PreScreen Topics"] = groups_all
    out["PreScreen Terms"] = terms_all
    out["Source Quality"] = statuses

    if bill_col:
        out["Duplicate Bill"] = out.duplicated(subset=[bill_col], keep="first")
    else:
        out["Duplicate Bill"] = False

    return out


def make_checkpoint_key(
    bill: str,
    source_url: str,
    source_name: str,
) -> str:
    payload = f"{normalize_text(bill)}|{normalize_text(source_url)}|{source_name}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
