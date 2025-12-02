# fetch_and_clean_grants.py
# Pulls grant opportunities from Grants.gov's search2 endpoint and writes:
#   - grants_opps_raw.csv
#   - grants_opps_clean.csv

from __future__ import annotations

import time
import json
from typing import Any, Dict, List
import requests
import pandas as pd

API_URL = "https://api.grants.gov/v1/api/search2"

KEYWORDS = [
    "semiconductor",
    "microelectronics",
    "advanced packaging",
    "chip manufacturing",
    "CHIPS",
]

# Includes posted + forecasted (and optionally closed so you have more data to cluster)
OPP_STATUSES = "posted|forecasted|closed"

PAGE_SIZE = 200          # try 100-500 if you want
MAX_RECORDS_PER_KEYWORD = 3000  # safety cap


def _post_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(API_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if int(data.get("errorcode", 0)) != 0:
        raise RuntimeError(f"API error: {data.get('msg')} | payload={payload}")
    return data


def fetch_all_for_keyword(keyword: str) -> List[Dict[str, Any]]:
    start = 0
    out: List[Dict[str, Any]] = []
    seen = set()

    while True:
        payload = {
            "rows": PAGE_SIZE,
            "keyword": keyword,
            "oppStatuses": OPP_STATUSES,
            # These are documented fields; empty string means "no filter"
            "oppNum": "",
            "eligibilities": "",
            "agencies": "",
            "aln": "",
            "fundingCategories": "",
            # Pagination: response echoes startRecordNum; endpoint accepts it in practice
            "startRecordNum": start,
        }

        data = _post_search(payload)
        payload_data = data.get("data", {}) or {}
        hits = payload_data.get("oppHits", []) or []
        hit_count = int(payload_data.get("hitCount", 0) or 0)

        if not hits:
            break

        for h in hits:
            # Prefer stable ID, else fallback to opportunity number
            k = h.get("id") or h.get("number")
            if k not in seen:
                h["_search_keyword"] = keyword
                out.append(h)
                seen.add(k)

        start += PAGE_SIZE
        if start >= hit_count:
            break
        if len(out) >= MAX_RECORDS_PER_KEYWORD:
            break

        time.sleep(0.2)  # be polite

    return out


def main() -> None:
    all_hits: List[Dict[str, Any]] = []
    for kw in KEYWORDS:
        hits = fetch_all_for_keyword(kw)
        print(f"Keyword={kw!r} -> {len(hits)} rows")
        all_hits.extend(hits)

    if not all_hits:
        raise SystemExit("No results returned. Try broader keywords (ex: 'manufacturing', 'electronics').")

    # Raw
    raw_df = pd.DataFrame(all_hits)
    raw_df.to_csv("grants_opps_raw.csv", index=False)
    print(f"Saved grants_opps_raw.csv ({len(raw_df)} rows)")

    # Clean
    df = raw_df.copy()

    # Normalize columns we care about (API returns these in oppHits)
    rename_map = {
        "id": "opp_id",
        "number": "opp_number",
        "title": "title",
        "agencyCode": "agency_code",
        "agencyName": "agency_name",
        "openDate": "open_date",
        "closeDate": "close_date",
        "oppStatus": "opp_status",
        "docType": "doc_type",
        "alnist": "aln_list",
        "_search_keyword": "search_keyword",
    }
    # Keep existing cols, just rename ones we know
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Dates are MM/DD/YYYY in examples
    for col in ["open_date", "close_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="%m/%d/%Y")

    # ALN list: ensure it's always a list, then create a pipe-joined string (easy for CSV + later processing)
    if "aln_list" in df.columns:
        def as_list(x):
            if isinstance(x, list):
                return x
            if pd.isna(x) or x is None:
                return []
            # sometimes lists get stringified; try to recover
            if isinstance(x, str) and x.strip().startswith("["):
                try:
                    return json.loads(x)
                except Exception:
                    return [x]
            return [str(x)]

        df["aln_list"] = df["aln_list"].apply(as_list)
        df["aln_joined"] = df["aln_list"].apply(lambda xs: "|".join(map(str, xs)))

    # Basic string cleanup
    for col in ["opp_number", "title", "agency_code", "agency_name", "opp_status", "doc_type", "search_keyword"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # Deduplicate by opp_id if present, else opp_number
    dedup_key = "opp_id" if "opp_id" in df.columns else "opp_number"
    df = df.drop_duplicates(subset=[dedup_key])

    # Keep a tidy set of columns
    keep_cols = [c for c in [
        "opp_id", "opp_number", "title",
        "agency_code", "agency_name",
        "open_date", "close_date",
        "opp_status", "doc_type",
        "aln_joined", "search_keyword",
    ] if c in df.columns]
    sort_cols = [c for c in ["open_date", "agency_name", "agency_code", "title"] if c in df.columns]
    df = df[keep_cols].sort_values(by=sort_cols, na_position="last")


    df.to_csv("grants_opps_clean.csv", index=False)
    print(f"Saved grants_opps_clean.csv ({len(df)} rows)")
    print("Quick sanity checks:")
    print("  Unique agencies:", df["agency_name"].nunique() if "agency_name" in df.columns else "n/a")
    print("  Date range:", df["open_date"].min(), "to", df["open_date"].max())


if __name__ == "__main__":
    main()
