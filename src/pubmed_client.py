from __future__ import annotations

"""
pubmed_client.py

Lightweight PubMed E-utilities client with:
- ESearch for PMID retrieval
- ESummary for stable article metadata
- EFetch for abstracts, MeSH terms, publication types, and language
- JSON caching for reproducible downstream analysis
"""

from pathlib import Path
import json
import time
import hashlib
import random
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# -----------------------------------------------------------------------------
# NCBI E-utilities settings
# -----------------------------------------------------------------------------
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

ESUMMARY_CHUNK_SIZE = 10
EFETCH_CHUNK_SIZE = 10
DEFAULT_TIMEOUT = 60
DEFAULT_RPS = 3.0
DEFAULT_SORT = "relevance"
REQUEST_RETRY_TOTAL = 5
REQUEST_BACKOFF_FACTOR = 0.8
REQUEST_STATUS_FORCELIST = (429, 500, 502, 503, 504)


# -----------------------------------------------------------------------------
# HTTP session and retry handling
# -----------------------------------------------------------------------------
def _make_session() -> requests.Session:
    """
    Create a requests session configured with retries for transient failures.
    """
    session = requests.Session()
    retry = Retry(
        total=REQUEST_RETRY_TOTAL,
        connect=REQUEST_RETRY_TOTAL,
        read=REQUEST_RETRY_TOTAL,
        status=REQUEST_RETRY_TOTAL,
        backoff_factor=REQUEST_BACKOFF_FACTOR,
        status_forcelist=REQUEST_STATUS_FORCELIST,
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION = _make_session()


def _robust_get(url: str, *, params: Dict[str, Any], timeout: int) -> requests.Response:
    """
    Send a GET request with an additional manual retry layer for transient
    connection errors that may not be fully handled by urllib3 retries.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, REQUEST_RETRY_TOTAL + 1):
        try:
            return _SESSION.get(url, params=params, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            sleep_seconds = min(
                30.0,
                (2 ** (attempt - 1)) * REQUEST_BACKOFF_FACTOR + random.random(),
            )
            time.sleep(sleep_seconds)

    if last_error is None:
        raise RuntimeError("Request failed without a captured exception.")

    raise last_error


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def _hash_key(term: str, retmax: int, sort: str) -> str:
    text = f"{term}||{int(retmax)}||{sort}"
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _sleep_rate_limit(last_time: float, requests_per_second: float) -> float:
    if requests_per_second <= 0:
        return time.time()

    min_interval = 1.0 / max(requests_per_second, 0.1)
    now = time.time()
    elapsed = now - last_time

    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    return time.time()


def _chunked(items: List[str], chunk_size: int) -> List[List[str]]:
    size = max(1, int(chunk_size))
    return [items[index:index + size] for index in range(0, len(items), size)]


def _deduplicate_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    deduplicated_values: List[str] = []

    for value in values:
        if value not in seen:
            deduplicated_values.append(value)
            seen.add(value)

    return deduplicated_values


def _build_common_request_params(
    email: str,
    api_key: Optional[str],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"email": email}
    if api_key:
        params["api_key"] = api_key
    return params


# -----------------------------------------------------------------------------
# PubMed response parsers
# -----------------------------------------------------------------------------
def _parse_esummary_response(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = data.get("result", {}) or {}
    uids = result.get("uids", []) or []

    records: List[Dict[str, Any]] = []
    for uid in uids:
        record = result.get(str(uid), {}) or {}
        authors = record.get("authors") or []

        records.append(
            {
                "pmid": str(uid),
                "title": record.get("title", "") or "",
                "pubdate": record.get("pubdate", "") or "",
                "source": record.get("source", "") or "",
                "authors": ", ".join(
                    author.get("name", "")
                    for author in authors[:10]
                    if author.get("name")
                ),
                "elocationid": record.get("elocationid", "") or "",
                "abstract": "",
                "mesh_terms": [],
                "pub_types": [],
                "language": "",
            }
        )
    return records


def _parse_efetch_response(
    xml_text: str,
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]], Dict[str, str]]:
    root = ET.fromstring(xml_text)

    abstract_by_pmid: Dict[str, str] = {}
    mesh_terms_by_pmid: Dict[str, List[str]] = {}
    publication_types_by_pmid: Dict[str, List[str]] = {}
    language_by_pmid: Dict[str, str] = {}

    for article in root.findall(".//PubmedArticle"):
        pmid_element = article.find(".//MedlineCitation/PMID")
        if pmid_element is None or not (pmid_element.text or "").strip():
            continue

        pmid = pmid_element.text.strip()

        abstract_sections: List[str] = []
        for abstract_text in article.findall(".//Article/Abstract/AbstractText"):
            label = abstract_text.attrib.get("Label")
            text = "".join(abstract_text.itertext()).strip()
            if not text:
                continue
            abstract_sections.append(f"{label}: {text}" if label else text)
        abstract_by_pmid[pmid] = " ".join(abstract_sections).strip()

        mesh_terms = [
            (element.text or "").strip()
            for element in article.findall(
                ".//MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName"
            )
            if (element.text or "").strip()
        ]
        mesh_terms_by_pmid[pmid] = _deduplicate_preserve_order(mesh_terms)

        publication_types = [
            (element.text or "").strip()
            for element in article.findall(".//Article/PublicationTypeList/PublicationType")
            if (element.text or "").strip()
        ]
        publication_types_by_pmid[pmid] = _deduplicate_preserve_order(publication_types)

        language_element = article.find(".//Article/Language")
        language_by_pmid[pmid] = (
            (language_element.text or "").strip()
            if language_element is not None
            else ""
        )

    return (
        abstract_by_pmid,
        mesh_terms_by_pmid,
        publication_types_by_pmid,
        language_by_pmid,
    )


# -----------------------------------------------------------------------------
# PubMed E-utilities operations
# -----------------------------------------------------------------------------
def esearch_pubmed(
    term: str,
    retmax: int,
    email: str,
    api_key: Optional[str] = None,
    sort: str = DEFAULT_SORT,
    rps: float = DEFAULT_RPS,
    timeout: int = DEFAULT_TIMEOUT,
    _last_t: float = 0.0,
) -> tuple[List[str], float]:
    """
    Run a PubMed ESearch query and return PMIDs.
    """
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": str(int(retmax)),
        "sort": sort,
        **_build_common_request_params(email=email, api_key=api_key),
    }

    last_request_time = _sleep_rate_limit(_last_t, rps)
    response = _robust_get(
        f"{EUTILS_BASE_URL}/esearch.fcgi",
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    pmids = data.get("esearchresult", {}).get("idlist", []) or []
    pmids = [str(pmid) for pmid in pmids if str(pmid).strip()]
    return pmids, last_request_time


def esummary_pubmed(
    pmids: List[str],
    email: str,
    api_key: Optional[str] = None,
    rps: float = DEFAULT_RPS,
    timeout: int = DEFAULT_TIMEOUT,
    _last_t: float = 0.0,
) -> tuple[List[Dict[str, Any]], float]:
    """
    Retrieve metadata for a PMID list using chunked PubMed ESummary.
    """
    if not pmids:
        return [], _last_t

    all_records: List[Dict[str, Any]] = []
    last_request_time = _last_t

    for pmid_chunk in _chunked(pmids, ESUMMARY_CHUNK_SIZE):
        params = {
            "db": "pubmed",
            "id": ",".join(str(pmid) for pmid in pmid_chunk),
            "retmode": "json",
            **_build_common_request_params(email=email, api_key=api_key),
        }

        last_request_time = _sleep_rate_limit(last_request_time, rps)
        response = _robust_get(
            f"{EUTILS_BASE_URL}/esummary.fcgi",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()

        data = response.json()
        all_records.extend(_parse_esummary_response(data))

    return all_records, last_request_time


def efetch_enrich(
    pmids: List[str],
    email: str,
    api_key: Optional[str] = None,
    rps: float = DEFAULT_RPS,
    timeout: int = DEFAULT_TIMEOUT,
    _last_t: float = 0.0,
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]], Dict[str, str], float]:
    """
    Retrieve abstracts, MeSH terms, publication types, and language using
    chunked PubMed EFetch.
    """
    if not pmids:
        return {}, {}, {}, {}, _last_t

    abstract_map: Dict[str, str] = {}
    mesh_map: Dict[str, List[str]] = {}
    publication_type_map: Dict[str, List[str]] = {}
    language_map: Dict[str, str] = {}
    last_request_time = _last_t

    for pmid_chunk in _chunked(pmids, EFETCH_CHUNK_SIZE):
        params = {
            "db": "pubmed",
            "id": ",".join(str(pmid) for pmid in pmid_chunk),
            "retmode": "xml",
            **_build_common_request_params(email=email, api_key=api_key),
        }

        last_request_time = _sleep_rate_limit(last_request_time, rps)
        response = _robust_get(
            f"{EUTILS_BASE_URL}/efetch.fcgi",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()

        (
            chunk_abstract_map,
            chunk_mesh_map,
            chunk_publication_type_map,
            chunk_language_map,
        ) = _parse_efetch_response(response.text)

        abstract_map.update(chunk_abstract_map)
        mesh_map.update(chunk_mesh_map)
        publication_type_map.update(chunk_publication_type_map)
        language_map.update(chunk_language_map)

    return abstract_map, mesh_map, publication_type_map, language_map, last_request_time


# -----------------------------------------------------------------------------
# Record enrichment and cache I/O
# -----------------------------------------------------------------------------
def _attach_efetch_fields_to_records(
    records: List[Dict[str, Any]],
    abstract_map: Dict[str, str],
    mesh_map: Dict[str, List[str]],
    publication_type_map: Dict[str, List[str]],
    language_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    enriched_records: List[Dict[str, Any]] = []

    for record in records:
        enriched_record = dict(record)
        pmid = str(enriched_record.get("pmid", "")).strip()
        enriched_record["abstract"] = abstract_map.get(pmid, "") or ""
        enriched_record["mesh_terms"] = mesh_map.get(pmid, []) or []
        enriched_record["pub_types"] = publication_type_map.get(pmid, []) or []
        enriched_record["language"] = language_map.get(pmid, "") or ""
        enriched_records.append(enriched_record)

    return enriched_records


def _read_cached_payload(cache_file: Path) -> Dict[str, Any]:
    with open(cache_file, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_cached_payload(cache_file: Path, payload: Dict[str, Any]) -> None:
    with open(cache_file, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Public cached search pipeline
# -----------------------------------------------------------------------------
def pubmed_search_with_cache(
    term: str,
    retmax: int,
    cache_dir: str | Path,
    email: str,
    api_key: Optional[str] = None,
    rps: float = DEFAULT_RPS,
    sort: str = DEFAULT_SORT,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Run a PubMed search pipeline with local JSON caching.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cache_key = _hash_key(term=term, retmax=int(retmax), sort=str(sort))
    cache_file = cache_path / f"pubmed_{cache_key}.json"

    if cache_file.exists():
        return _read_cached_payload(cache_file)

    last_request_time = 0.0

    pmids, last_request_time = esearch_pubmed(
        term=term,
        retmax=retmax,
        email=email,
        api_key=api_key,
        sort=sort,
        rps=rps,
        timeout=timeout,
        _last_t=last_request_time,
    )

    summary_records, last_request_time = esummary_pubmed(
        pmids=pmids,
        email=email,
        api_key=api_key,
        rps=rps,
        timeout=timeout,
        _last_t=last_request_time,
    )

    (
        abstract_map,
        mesh_map,
        publication_type_map,
        language_map,
        last_request_time,
    ) = efetch_enrich(
        pmids=pmids,
        email=email,
        api_key=api_key,
        rps=rps,
        timeout=timeout,
        _last_t=last_request_time,
    )

    records = _attach_efetch_fields_to_records(
        records=summary_records,
        abstract_map=abstract_map,
        mesh_map=mesh_map,
        publication_type_map=publication_type_map,
        language_map=language_map,
    )

    payload = {
        "term": term,
        "retmax": int(retmax),
        "pmids": pmids,
        "records": records,
        "fetched_at": time.strftime("%Y-%m-%d"),
        "sort": sort,
    }

    _write_cached_payload(cache_file, payload)
    return payload
