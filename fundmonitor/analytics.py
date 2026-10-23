"""Peer-rank, benchmark and consistency analytics on a monthly date grid."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import benchmarks as bm
from .navs import fetch_many

N_GRID = 61      # as_of minus 0..60 months
N_ROLL = 36      # rolling 1Y windows measured (monthly steps)
MIN_PEERS = 5    # minimum peers for a percentile rank to be meaningful
TOL = 0.0025     # excess return smaller than 0.25pp (~passive tracking noise) counts as "in line"
HORIZONS = {"3m": 3, "6m": 6, "1y": 12, "3y": 36, "5y": 60}
STALE_DAYS = 7


# ----------------------------------------------------------------- grid helpers
def grid_dates(as_of: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([as_of - pd.DateOffset(months=k) for k in range(N_GRID)])


def to_grid(s: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    asc = dates.sort_values()
    v = np.array(s.reindex(s.index.union(asc)).ffill().reindex(asc), dtype=float)  # writable copy (pandas 3)
    v[np.asarray(asc < s.index[0])] = np.nan
    return v[::-1]  # index k = as_of - k months


def returns_from_grid(P: np.ndarray) -> dict:
    P = np.atleast_2d(P)
    out = {}
    for h, m in HORIZONS.items():
        r = P[:, 0] / P[:, m] - 1
        out[h] = (1 + r) ** (12 / m) - 1 if m > 12 else r
    out["roll"] = P[:, :N_ROLL] / P[:, 12:12 + N_ROLL] - 1
    return out


def pct_rank(M: np.ndarray) -> np.ndarray:
    """0 = worst, 1 = best, per column (or for a 1-D vector)."""
    one_d = M.ndim == 1
    df = pd.DataFrame(M.reshape(len(M), -1))
    cnt = df.notna().sum()
    p = (df.rank(method="average") - 1).div((cnt - 1).replace(0, np.nan))
    p.loc[:, cnt < MIN_PEERS] = np.nan
    p = p.to_numpy()
    return p[:, 0] if one_d else p


def blend_grid(parts: list[tuple[np.ndarray, float]]) -> np.ndarray:
    r = sum(w * (P[:-1] / P[1:] - 1) for P, w in parts)       # monthly, k = 0..59 (newest first)
    out = np.full(N_GRID, np.nan)
    out[-1] = 100.0
    for k in range(N_GRID - 2, -1, -1):
        out[k] = out[k + 1] * (1 + r[k])
    return out


# ---------------------------------------------------------------- group metrics
def group_metrics(P: np.ndarray, bench: dict | None) -> dict:
    """Metrics for every fund in one peer group. `bench` = returns dict for the benchmark
    (1-row arrays) or None to use the peer median."""
    R = returns_from_grid(P)
    if bench is None:
        B = {k: np.nanmedian(v, axis=0) if k == "roll" else np.nanmedian(v) for k, v in R.items()}
    else:
        B = {k: (v[0] if k == "roll" else v[0]) for k, v in bench.items()}
    m = {"n_peers": np.full(len(P), len(P))}
    for h in HORIZONS:
        m[f"ret_{h}"] = R[h]
        m[f"bench_{h}"] = np.full(len(P), B[h])
        m[f"ex_{h}"] = R[h] - B[h]
        m[f"pct_{h}"] = pct_rank(R[h])
    pct = pct_rank(R["roll"])
    exc = R["roll"] - B["roll"][None, :]
    v_rank, v_exc = ~np.isnan(pct), ~np.isnan(exc)
    n_rank = v_rank.sum(1)
    n_exc = v_exc.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m["n_win"] = n_rank
        m["sh_bh"] = ((pct < 0.5) & v_rank).sum(1) / n_rank
        m["sh_bq"] = ((pct < 0.25) & v_rank).sum(1) / n_rank
        m["sh_th"] = ((pct >= 0.5) & v_rank).sum(1) / n_rank
        m["sh_tq"] = ((pct >= 0.75) & v_rank).sum(1) / n_rank
        m["sh_lag"] = ((exc < -TOL) & v_exc).sum(1) / n_exc
        m["sh_lead"] = ((exc > TOL) & v_exc).sum(1) / n_exc
    p5 = np.where(np.isnan(m["pct_5y"]), m["pct_3y"], m["pct_5y"])
    m["score"] = (0.30 * m["pct_3y"] + 0.25 * m["pct_1y"] + 0.15 * p5 + 0.30 * m["sh_th"])
    m["_roll_pct"], m["_roll_exc"] = pct, exc
    return m


@dataclass
class Result:
    as_of: pd.Timestamp
    metrics: pd.DataFrame                      # one row per peer fund (all groups)
    rolls: dict                                # code -> (pct[36], excess[36])
    bench_info: pd.DataFrame                   # per group: benchmark label / source
    holdings: pd.DataFrame
    notes: list = field(default_factory=list)


def run_analysis(holdings: pd.DataFrame, uni: pd.DataFrame, progress=None) -> Result:
    """holdings: output of holdings.match_holdings. Fetches peers + benchmark proxies and computes metrics."""
    say = progress or (lambda frac, msg: None)
    matched = holdings[holdings["code"].notna()]
    groups = sorted(matched["group"].unique())
    latest = uni["last_date"].max()
    peers = uni[uni["group"].isin(groups) & uni["plan"].isin(["Regular", "NA"]) & uni["is_growth"]
                & ~uni["segregated"] & (uni["last_date"] >= latest - pd.Timedelta(days=STALE_DAYS))]
    peers = peers.sort_values("last_date").drop_duplicates("norm", keep="last")
    held_codes = set(matched["code"].astype(int))
    peers = pd.concat([peers, uni[uni["code"].isin(held_codes) & ~uni["code"].isin(peers["code"])]])

    specs = {g: bm.spec_for(g) for g in groups}
    need_keys = sorted({k for s in specs.values() for k in bm.spec_keys(s)})
    bench_cands = {k: bm.candidate_codes(uni, k) for k in need_keys}
    all_codes = list(peers["code"]) + [c for v in bench_cands.values() for c in v]

    say(0.02, f"Downloading NAV history for {len(set(all_codes))} funds…")
    navs = fetch_many(all_codes, progress=lambda i, n: say(0.02 + 0.80 * i / n, f"NAV history {i}/{n}"))
    say(0.85, "Computing rankings and benchmark comparisons…")

    as_of = max(s.index[-1] for c, s in navs.items() if c in set(peers["code"]))
    # most common latest date among peers is the safest as_of (avoids a single lagging fund)
    last_dates = pd.Series({c: s.index[-1] for c, s in navs.items() if c in set(peers["code"])})
    as_of = last_dates.mode().max() if not last_dates.empty else as_of
    dates = grid_dates(as_of)
    resolved = bm.resolve(uni, navs, need_keys)

    frames, rolls, info = [], {}, []
    for g in groups:
        pg = peers[peers["code"].isin(navs)]
        pg = pg[pg["group"] == g]
        pg = pg[[navs[c].index[-1] >= as_of - pd.Timedelta(days=STALE_DAYS) for c in pg["code"]]]
        if pg.empty:
            continue
        P = np.vstack([to_grid(navs[c], dates) for c in pg["code"]])
        bench, label, source = None, "Peer median (no passive proxy available)", "median"
        if g.startswith("Passive"):
            label = "Peer median (same tracked index)"
        spec = specs[g]
        if spec[0] == "proxy":
            for k in spec[1]:
                if k in resolved:
                    bench = returns_from_grid(to_grid(resolved[k][2], dates))
                    label, source = bm.PROXIES[k][0], resolved[k][1]
                    break
        elif spec[0] == "blend":
            parts = []
            for keys, w in spec[1]:
                k = next((k for k in keys if k in resolved), None)
                if k is None:
                    parts = None
                    break
                parts.append((to_grid(resolved[k][2], dates), w))
            if parts:
                bench = returns_from_grid(blend_grid(parts))
                label = spec[2]
                source = " | ".join(next(resolved[k][1] for k in keys if k in resolved) for keys, _ in spec[1])
        if bench is not None and np.isnan(bench["1y"][0]):
            bench, label, source = None, "Peer median (proxy lacked history)", "median"
        met = group_metrics(P, bench)
        roll_pct, roll_exc = met.pop("_roll_pct"), met.pop("_roll_exc")
        df = pd.DataFrame(met)
        df.insert(0, "code", pg["code"].to_numpy())
        df.insert(1, "fund", pg["name"].to_numpy())
        df.insert(2, "amc", pg["amc_short"].to_numpy())
        df.insert(3, "group", g)
        df.insert(4, "asset", pg["asset"].to_numpy())
        df["first_nav"] = [navs[c].index[0] for c in pg["code"]]
        frames.append(df)
        for i, c in enumerate(pg["code"]):
            rolls[int(c)] = (roll_pct[i], roll_exc[i])
        info.append({"group": g, "benchmark": label, "proxy_fund": source, "peers": len(pg)})
    metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    say(1.0, "Done")
    return Result(as_of, metrics, rolls, pd.DataFrame(info), holdings)


# --------------------------------------------------------------- classification
@dataclass
class Params:
    mode: str = "half"            # "half" -> bottom/top half, "quartile" -> bottom/top quartile
    thr_rank_half: float = 0.70   # share of rolling windows in bottom/top half
    thr_rank_quartile: float = 0.50
    thr_bench: float = 0.70       # share of rolling windows below/above benchmark
    min_windows: int = 24         # months of rolling-1Y history required to call a fund "consistent"
    exclude_cash: bool = True     # skip liquid / overnight / money-market (ranking is noise there)


CASH_GROUPS = {"Liquid", "Overnight", "Money Market", "Ultra Short Duration", "Arbitrage"}


def quartile_label(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    return "Q1 (top)" if p >= 0.75 else "Q2" if p >= 0.5 else "Q3" if p >= 0.25 else "Q4 (bottom)"


def classify(df: pd.DataFrame, p: Params) -> pd.Series:
    q = p.mode == "quartile"
    thr = p.thr_rank_quartile if q else p.thr_rank_half
    low = (lambda x: x < 0.25) if q else (lambda x: x < 0.5)
    high = (lambda x: x >= 0.75) if q else (lambda x: x >= 0.5)
    sh_low = df["sh_bq"] if q else df["sh_bh"]
    sh_high = df["sh_tq"] if q else df["sh_th"]
    enough = df["n_win"] >= p.min_windows
    rec_ok = df["pct_6m"].notna() & df["pct_1y"].notna()

    cons_under = enough & (sh_low >= thr) & (df["sh_lag"] >= p.thr_bench) & ~(df["ex_3y"] > TOL)
    cons_out = enough & (sh_high >= thr) & (df["sh_lead"] >= p.thr_bench) & ~(df["ex_3y"] < -TOL)
    rec_under = rec_ok & low(df["pct_6m"]) & low(df["pct_1y"]) & (df["ex_6m"] < -TOL) & (df["ex_1y"] < -TOL)
    rec_out = rec_ok & high(df["pct_6m"]) & high(df["pct_1y"]) & (df["ex_6m"] > TOL) & (df["ex_1y"] > TOL)

    s = pd.Series("Neutral", index=df.index)
    s[rec_out] = "Recent outperformer"
    s[cons_out] = "Consistent outperformer"
    s[rec_under] = "Recent underperformer (watch)"
    s[cons_under] = "Consistent underperformer (exit)"
    s[df["pct_1y"].isna() & df["ret_1y"].notna()] = "Not ranked (too few peers)"
    if p.exclude_cash:
        s[df["group"].isin(CASH_GROUPS)] = "Not assessed (cash-like)"
    return s


# ----------------------------------------------------------------- replacements
_CORE_EQ = {"Large Cap", "Large & Mid Cap", "Mid Cap", "Small Cap", "Flexi Cap", "Multi Cap", "ELSS",
            "Focused", "Value/Contra"}
_HYBRID = {"Aggressive Hybrid", "Balanced Advantage", "Multi Asset Allocation", "Balanced Hybrid", "Equity Savings"}
_DEBT_LONG = {"Corporate Bond", "Banking & PSU Debt", "Gilt", "Dynamic Bond", "Medium Duration", "Short Duration",
              "Medium to Long Duration", "Long Duration", "Floater", "Credit Risk"}
# categories an AMC-level alternative may come from when the AMC has no better fund in the same category
FAMILY = {g: tuple(s) for s in (_CORE_EQ, _HYBRID, _DEBT_LONG) for g in s}


def _why(r: pd.Series) -> str:
    bits = []
    if pd.notna(r["ret_3y"]):
        bits.append(f"3Y {r['ret_3y']*100:.1f}% p.a. ({r['ex_3y']*100:+.1f}% vs bm)")
    if pd.notna(r["sh_th"]):
        bits.append(f"top-half in {r['sh_th']*100:.0f}% of windows")
    return "; ".join(bits)


def suggest_replacement(metrics: pd.DataFrame, code: int, min_windows: int = 24) -> dict:
    me = metrics[metrics["code"] == code]
    if me.empty:
        return {}
    me = me.iloc[0]
    pool = metrics[(metrics["code"] != code) & (metrics["n_win"] >= min_windows) & metrics["score"].notna()]
    good = pool[(pool["sh_bh"] < 0.5) & ~(pool["ex_3y"] < 0)]

    def best(df):
        df = df[df["score"] > (me["score"] if pd.notna(me["score"]) else -1)]
        return df.sort_values("score", ascending=False).head(1)

    out = {"same_cat_fund": "", "same_cat_amc": "", "same_cat_why": "",
           "same_amc_fund": "", "same_amc_cat": "", "same_amc_why": ""}
    cat = best(good[good["group"] == me["group"]])
    if not cat.empty:
        r = cat.iloc[0]
        out.update(same_cat_fund=r["fund"], same_cat_amc=r["amc"], same_cat_why=_why(r))
    amc_pool = good[good["amc"] == me["amc"]]
    same = best(amc_pool[amc_pool["group"] == me["group"]])
    fam = FAMILY.get(me["group"])
    if same.empty and fam:  # fall back to the AMC's best fund in a comparable category family
        same = best(amc_pool[amc_pool["group"].isin(fam)])
    if not same.empty:
        r = same.iloc[0]
        out.update(same_amc_fund=r["fund"], same_amc_cat=r["group"], same_amc_why=_why(r))
    return out
