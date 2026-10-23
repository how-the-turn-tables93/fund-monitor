"""AMFI NAVAll parser: scheme universe with AMC, category and plan metadata."""
from __future__ import annotations

import io
import re

import pandas as pd
import requests

AMFI_URLS = [
    "https://portal.amfiindia.com/spages/NAVAll.txt",
    "https://www.amfiindia.com/spages/NAVAll.txt",
]

# (regex on lower-cased category text, canonical sub-category). Order matters.
_CANON = [
    (r"large\s*(&|and)\s*mid", "Large & Mid Cap"),
    (r"large cap", "Large Cap"),
    (r"mid cap", "Mid Cap"),
    (r"small cap", "Small Cap"),
    (r"flexi", "Flexi Cap"),
    (r"multi cap", "Multi Cap"),
    (r"elss|tax saver", "ELSS"),
    (r"focused", "Focused"),
    (r"dividend yield|contra|value", "Value/Contra"),  # one group: too few funds each on their own
    (r"sectoral|thematic", "Sectoral/Thematic"),
    (r"aggressive hybrid", "Aggressive Hybrid"),
    (r"balanced advantage|dynamic asset", "Balanced Advantage"),
    (r"multi asset", "Multi Asset Allocation"),
    (r"equity savings", "Equity Savings"),
    (r"arbitrage", "Arbitrage"),
    (r"conservative hybrid", "Conservative Hybrid"),
    (r"balanced hybrid", "Balanced Hybrid"),
    (r"banking and psu", "Banking & PSU Debt"),
    (r"corporate bond", "Corporate Bond"),
    (r"credit risk", "Credit Risk"),
    (r"dynamic bond|dynamic term", "Dynamic Bond"),
    (r"10.year constant|gilt", "Gilt"),
    (r"liquid", "Liquid"),
    (r"overnight", "Overnight"),
    (r"money market", "Money Market"),
    (r"ultra short", "Ultra Short Duration"),
    (r"low duration|low term", "Low Duration"),
    (r"medium to long", "Medium to Long Duration"),
    (r"short duration|short term", "Short Duration"),
    (r"medium duration|medium term", "Medium Duration"),
    (r"long duration|long term", "Long Duration"),
    (r"floating", "Floater"),
    (r"retirement", "Retirement"),
    (r"children", "Children's"),
    (r"gold etf", "Gold ETF"),
    (r"silver etf", "Silver ETF"),
    (r"overseas", "Overseas FoF"),
    (r"fof|fund of funds", "Domestic FoF"),
    (r"index funds?", "Index Fund"),
    (r"etf", "Other ETF"),
]
_ASSET = [
    (r"equity scheme", "Equity"),
    (r"hybrid", "Hybrid"),
    (r"debt|income", "Debt"),
    (r"solution", "Solution"),
]

_OPT_BAD = re.compile(r"idcw|dividend|payout|reinvest|bonus|periodic|annual|weekly|monthly|quarterly|daily|half", re.I)
_SEG_PLAN = re.compile(
    r"^(regular|direct|retail|institutional)( plan)?$|^(growth|idcw|dividend)( option| plan)?$"
    r"|idcw|payout|reinvest|(monthly|weekly|daily|quarterly|annual|half[ -]?yearly|periodic) (idcw|dividend)",
    re.I,
)


# spelling variants seen between RTA reports and the AMFI file
_SPELL = [
    (r"\b(large|mid|small|flexi|multi|micro) cap\b", r"\1cap"),
    (r"\bhealth care\b", "healthcare"),
    (r"\bfund of funds?\b", "fof"),
    (r"\bliquid plan\b", "liquid fund"),
    (r"\blong short\b", "longshort"),
    (r"\bcompulsory lock in\b", "compulsory lockin"),
]


def norm(name: str) -> str:
    s = str(name).lower().replace("&", " and ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = " ".join(s.split())
    for a, b in _SPELL:
        s = re.sub(a, b, s)
    return s


def base_name(full: str) -> str:
    """Strip plan/option segments ('- Regular Plan - Growth') from a scheme name."""
    parts = [p.strip() for p in re.split(r"\s+-\s+", str(full))]
    keep = [p for p in parts if not _SEG_PLAN.search(p)]
    return " - ".join(keep) if keep else str(full)


def canon_category(header: str) -> tuple[str, str]:
    m = re.match(r".*?\((.*)\)\s*$", header)
    text = (m.group(1) if m else header).replace("’", "'")
    low = text.lower()
    asset = next((a for rx, a in _ASSET if re.search(rx, low)), "Other")
    if "index fund" in low or "etf" in low or "fof" in low or "fund of funds" in low:
        asset = "Passive/FoF"
    sub = next((c for rx, c in _CANON if re.search(rx, low)), text.split(" - ")[-1].strip())
    return asset, sub


_THEMES = [
    (r"bank|financ|bfsi", "Banking & Financial Services"),
    (r"pharma|health|medic", "Pharma & Healthcare"),
    (r"tech|digital|\bit\b|information", "Technology"),
    (r"infra", "Infrastructure"),
    (r"psu|public sector|cpse", "PSU"),
    (r"consum|fmcg", "Consumption"),
    (r"energy|power|oil|gas", "Energy & Power"),
    (r"manufactur|make in india|defence|industr", "Manufacturing & Defence"),
    (r"\bmnc\b|multinational", "MNC"),
    (r"auto|mobility|transport", "Auto & Mobility"),
    (r"real estate|housing", "Real Estate"),
    (r"esg|sustain|climate", "ESG"),
]


def theme_of(name: str) -> str:
    low = name.lower()
    for rx, t in _THEMES:
        if re.search(rx, low):
            return t
    return "Diversified Thematic"


_INDEX_KEY = re.compile(
    r"(nifty|sensex|bse|s and p|s&p|crisil|msci|nasdaq|nyse|hang seng|gold|silver|"
    r"fang|g sec|sdl|ibx|overnight|liquid)", re.I)


def tracked_index(name: str) -> str:
    n = norm(name)
    m = _INDEX_KEY.search(n)
    t = n[m.start():] if m else n
    t = re.sub(r"\b(index|etf|fof|fund of funds?|fund|exchange traded|plan|growth|direct|regular)\b", " ", t)
    return " ".join(t.split()).title() or "Other"


def _fetch_text() -> str:
    last = None
    for url in AMFI_URLS:
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.content.decode("utf-8", errors="ignore")
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"Could not download AMFI NAV file: {last}")


def load_universe(text: str | None = None) -> pd.DataFrame:
    text = text or _fetch_text()
    cat_header, amc = None, None
    rows = []
    for line in io.StringIO(text):
        line = line.strip()
        if not line:
            continue
        if ";" not in line:
            if re.match(r"^(Open Ended|Close Ended|Interval Fund)", line):
                cat_header = line
            elif re.search(r"mutual fund$", line, re.I):
                amc = line
            continue
        p = line.split(";")
        if len(p) < 8 or not p[0].strip().isdigit():
            continue
        rows.append((int(p[0]), p[3].strip(), p[4].strip(), p[5].strip(), p[6], p[7], cat_header, amc))
    df = pd.DataFrame(rows, columns=["code", "name", "plan_col", "option", "nav", "date", "header", "amc"])
    df["last_date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df[df["header"].fillna("").str.startswith("Open Ended")].copy()
    low_plan = (df["plan_col"] + " " + df["name"] + " " + df["option"]).str.lower()
    df["plan"] = "NA"
    df.loc[low_plan.str.contains("regular"), "plan"] = "Regular"
    df.loc[low_plan.str.contains("direct"), "plan"] = "Direct"
    df["is_growth"] = (~df["option"].str.contains(_OPT_BAD)
                       & ~df["name"].str.contains(r"idcw|payout|reinvest|bonus|dividend(?! yield)", case=False))
    df["segregated"] = df["name"].str.contains("segregated", case=False)
    cat = df["header"].map(canon_category)
    df["asset"] = [c[0] for c in cat]
    df["subcat"] = [c[1] for c in cat]
    df["base"] = df["name"].map(base_name)
    df["norm"] = df["base"].map(norm)
    df["group"] = df["subcat"]
    sect = df["subcat"] == "Sectoral/Thematic"
    df.loc[sect, "group"] = "Sectoral/Thematic: " + df.loc[sect, "base"].map(theme_of)
    passive = df["subcat"].isin(["Index Fund", "Other ETF"])
    df.loc[passive, "group"] = "Passive: " + df.loc[passive, "base"].map(tracked_index)
    df["amc_short"] = df["amc"].fillna("").str.replace(r"\s*mutual fund$", "", regex=True, case=False)
    return df.drop(columns=["nav", "date"]).reset_index(drop=True)
