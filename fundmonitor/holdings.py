"""Parse the client AUM-by-scheme workbook and match schemes to the AMFI universe."""
from __future__ import annotations

import difflib

import pandas as pd

from .amfi import base_name, norm

try:  # optional, faster fuzzy matching
    from rapidfuzz import fuzz, process
except Exception:  # noqa: BLE001
    process = None

COLS = ["sno", "scheme", "r3m", "r6m", "r1y", "r3y", "r5y", "r10y", "rsi",
        "invested", "div_paid", "div_reinv", "aum", "gain", "abs_ret", "weight"]


def parse_aum_file(file) -> pd.DataFrame:
    raw = pd.read_excel(file, header=None, sheet_name=0)
    hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("S.No.")]
    if len(hdr) == 0:
        raise ValueError("Could not find the 'S.No.' header row - is this an AUM By Scheme report?")
    body = raw.iloc[hdr[0] + 2:, :16].copy()
    body.columns = COLS
    body = body[pd.to_numeric(body["sno"], errors="coerce").notna()].copy()
    for c in COLS[2:]:
        body[c] = pd.to_numeric(body[c], errors="coerce")
    body["scheme"] = body["scheme"].astype(str).str.strip()
    return body.reset_index(drop=True)


def aggregate_holdings(df: pd.DataFrame, min_aum: float = 1.0) -> pd.DataFrame:
    """Collapse IDCW/Growth rows of the same scheme into one holding."""
    d = df[df["aum"].fillna(0) > min_aum].copy()
    d["base"] = d["scheme"].map(base_name)
    d["norm"] = d["base"].map(norm)
    agg = d.groupby("norm").agg(
        scheme=("base", "first"),
        aum=("aum", "sum"),
        invested=("invested", "sum"),
        gain=("gain", "sum"),
        rows=("scheme", "count"),
    ).reset_index()
    agg["weight_pct"] = agg["aum"] / agg["aum"].sum() * 100
    return agg.sort_values("aum", ascending=False).reset_index(drop=True)


def match_holdings(h: pd.DataFrame, uni: pd.DataFrame) -> pd.DataFrame:
    """Attach AMFI code / AMC / group to each holding (Regular-plan Growth variant)."""
    cand = uni[(uni["plan"].isin(["Regular", "NA"])) & uni["is_growth"] & ~uni["segregated"]]
    cand = cand.sort_values("last_date").drop_duplicates("norm", keep="last")
    by_norm = cand.set_index("norm")
    norms = list(by_norm.index)
    out = []
    for n in h["norm"]:
        method, key = "exact", n if n in by_norm.index else None
        if key is None:
            if process is not None:
                m = process.extractOne(n, norms, scorer=fuzz.token_sort_ratio, score_cutoff=90)
                key = m[0] if m else None
            else:
                m = difflib.get_close_matches(n, norms, n=1, cutoff=0.9)
                key = m[0] if m else None
            method = "fuzzy"
        if key is None:
            out.append((None, None, None, None, None, "unmatched"))
        else:
            r = by_norm.loc[key]
            out.append((int(r["code"]), r["name"], r["amc_short"], r["group"], r["subcat"], method))
    m = pd.DataFrame(out, columns=["code", "amfi_name", "amc", "group", "subcat", "match"], index=h.index)
    return pd.concat([h, m], axis=1)
