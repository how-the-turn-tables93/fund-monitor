"""Benchmark mapping.

Official benchmark TRI series (NSE / BSE / CRISIL) are not freely downloadable by
scripts, so each benchmark is proxied by the NAV of a low-cost passive fund or ETF
that tracks the same index (public via AMFI / mfapi.in). A passive fund trails its
TRI by roughly its expense ratio (~0.1-0.3% p.a.), which is immaterial for ranking
active funds. Where no passive proxy exists (e.g. liquid / money-market funds) the
category median is used instead and flagged as such.

Edit SUB_SPECS / THEME_SPECS / PROXIES to change the mapping.
"""
from __future__ import annotations

import re

import pandas as pd

_EX = (r"next|value|equal|alpha|momentum|quality|low vol|volatility|arbitrage|hybrid|top|shariah|"
       r"dividend|enhanced|factor|junior|fof|fund of fund|covered|multi ?factor|growth sectors|esg|"
       r"leader|private|psu|ex |\d+ \d+ \d+|maturity|target|2\d{3}")

# key -> (label, include regex, exclude regex, allowed subcats)
_PASSIVE = ("Index Fund", "Other ETF")
PROXIES: dict[str, tuple[str, str, str, tuple]] = {
    "n50": ("Nifty 50 TRI", r"\bnifty 50\b", _EX, _PASSIVE),
    "n100": ("Nifty 100 TRI", r"\bnifty 100\b", _EX, _PASSIVE),
    "n500": ("Nifty 500 TRI", r"\bnifty 500\b", _EX, _PASSIVE),
    "mid150": ("Nifty Midcap 150 TRI", r"\bnifty midcap 150\b", _EX, _PASSIVE),
    "small250": ("Nifty Smallcap 250 TRI", r"\bnifty smallcap 250\b", _EX, _PASSIVE),
    "lm250": ("Nifty LargeMidcap 250 TRI", r"\bnifty large ?midcap 250\b", _EX, _PASSIVE),
    "bank": ("Nifty Bank TRI", r"\bnifty bank\b", r"fof|fund of fund|private|psu", _PASSIVE),
    "finserv": ("Nifty Financial Services TRI", r"\bnifty fin(ancial)? ?services?\b",
                r"fof|fund of fund|ex bank|25 35", _PASSIVE),
    "it": ("Nifty IT TRI", r"\bnifty it\b", r"fof|fund of fund", _PASSIVE),
    "pharma": ("Nifty Pharma/Healthcare TRI", r"\bnifty (pharma|healthcare)\b", r"fof|fund of fund", _PASSIVE),
    "cons": ("Nifty India Consumption TRI", r"\bnifty india consumption\b", r"fof|fund of fund", _PASSIVE),
    "mfg": ("Nifty India Manufacturing TRI", r"\bnifty india (manufacturing|defence)\b", r"fof|fund of fund", _PASSIVE),
    "infra": ("Nifty Infrastructure TRI", r"\bnifty (india )?infra(structure)?\b", r"fof|fund of fund", _PASSIVE),
    "psu": ("PSU index TRI", r"\b(bse psu|nifty psu|nifty cpse|cpse)\b", r"fof|fund of fund|bank", _PASSIVE),
    "energy": ("Nifty Energy/Oil & Gas TRI", r"\bnifty (energy|oil and gas)\b", r"fof|fund of fund", _PASSIVE),
    "auto": ("Nifty Auto TRI", r"\bnifty auto\b", r"fof|fund of fund", _PASSIVE),
    "corp": ("Corporate-bond index", r"\b(corporate bond|aaa|banking and psu|bharat bond)\b",
             r"fof|fund of fund|maturity|target|2\d{3}|etf fof", _PASSIVE),
    "gilt": ("Gilt / G-Sec index", r"\b(gilt|g sec|10 year|sdl)\b",
             r"fof|fund of fund|maturity|target|2\d{3}", _PASSIVE),
    "gold": ("Domestic gold price", r"\bgold\b", r"fof|fund of fund|silver|sector", ("Gold ETF",)),
    "silver": ("Domestic silver price", r"\bsilver\b", r"fof|fund of fund|gold", ("Silver ETF",)),
}

_EQ = ["n500", "n50"]
_DEBT = ["corp", "gilt"]

# group (canonical sub-category) -> spec
#   ("proxy", [keys in preference order])  or  ("blend", [(keys, weight), ...], label)  or  ("median",)
SUB_SPECS: dict[str, tuple] = {
    "Large Cap": ("proxy", ["n100", "n50"]),
    "Mid Cap": ("proxy", ["mid150"]),
    "Small Cap": ("proxy", ["small250"]),
    "Large & Mid Cap": ("proxy", ["lm250", "n500"]),
    "Flexi Cap": ("proxy", ["n500"]),
    "Multi Cap": ("proxy", ["n500"]),
    "ELSS": ("proxy", ["n500"]),
    "Focused": ("proxy", ["n500"]),
    "Value/Contra": ("proxy", ["n500"]),
    "Aggressive Hybrid": ("blend", [(_EQ, .65), (_DEBT, .35)], "65% Nifty 500 + 35% debt index"),
    "Balanced Advantage": ("blend", [(_EQ, .50), (_DEBT, .50)], "50% Nifty 50/500 + 50% debt index"),
    "Balanced Hybrid": ("blend", [(_EQ, .50), (_DEBT, .50)], "50% Nifty 50/500 + 50% debt index"),
    "Multi Asset Allocation": ("blend", [(_EQ, .50), (_DEBT, .25), (["gold"], .25)],
                               "50% equity + 25% debt + 25% gold"),
    "Equity Savings": ("blend", [(_EQ, .35), (_DEBT, .65)], "35% equity + 65% debt index"),
    "Conservative Hybrid": ("blend", [(_EQ, .15), (_DEBT, .85)], "15% equity + 85% debt index"),
    "Corporate Bond": ("proxy", ["corp"]),
    "Banking & PSU Debt": ("proxy", ["corp"]),
    "Gilt": ("proxy", ["gilt"]),
    "Long Duration": ("proxy", ["gilt"]),
    "Dynamic Bond": ("proxy", ["gilt", "corp"]),
    "Medium to Long Duration": ("proxy", ["corp", "gilt"]),
    "Gold ETF": ("proxy", ["gold"]),
    "Silver ETF": ("proxy", ["silver"]),
}
THEME_SPECS: dict[str, tuple] = {
    "Banking & Financial Services": ("proxy", ["finserv", "bank"]),
    "Pharma & Healthcare": ("proxy", ["pharma"]),
    "Technology": ("proxy", ["it"]),
    "Infrastructure": ("proxy", ["infra"]),
    "PSU": ("proxy", ["psu"]),
    "Consumption": ("proxy", ["cons"]),
    "Energy & Power": ("proxy", ["energy"]),
    "Manufacturing & Defence": ("proxy", ["mfg"]),
    "Auto & Mobility": ("proxy", ["auto"]),
    "Diversified Thematic": ("proxy", ["n500"]),
}


def spec_for(group: str) -> tuple:
    if group.startswith("Sectoral/Thematic: "):
        return THEME_SPECS.get(group.split(": ", 1)[1], ("median",))
    return SUB_SPECS.get(group, ("median",))


def spec_keys(spec: tuple) -> list[str]:
    if spec[0] == "proxy":
        return list(spec[1])
    if spec[0] == "blend":
        return [k for keys, _ in spec[1] for k in keys]
    return []


def candidate_codes(uni: pd.DataFrame, key: str, maxn: int = 5) -> list[int]:
    label, inc, exc, subs = PROXIES[key]
    latest = uni["last_date"].max()
    c = uni[uni["subcat"].isin(subs) & uni["plan"].isin(["Direct", "NA"]) & uni["is_growth"]
            & ~uni["segregated"] & (uni["last_date"] >= latest - pd.Timedelta(days=10))]
    ok = c["norm"].map(lambda n: bool(re.search(inc, n)) and not re.search(exc, n))
    return sorted(c[ok]["code"].tolist())[:maxn]


def resolve(uni: pd.DataFrame, navs: dict, keys) -> dict[str, tuple]:
    """key -> (code, name, series): the candidate with the longest NAV history."""
    out = {}
    names = uni.drop_duplicates("code").set_index("code")["name"]
    for k in dict.fromkeys(keys):
        best = None
        for code in candidate_codes(uni, k):
            s = navs.get(code)
            if s is not None and (best is None or s.index[0] < best[2].index[0]):
                best = (code, names.get(code, str(code)), s)
        if best:
            out[k] = best
    return out
