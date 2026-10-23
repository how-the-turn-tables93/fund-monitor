"""Peer-rank, benchmark and consistency analytics on a monthly date grid."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import benchmarks as bm
from .navs import fetch_many

warnings.filterwarnings("ignore", message="All-NaN slice")
N_GRID = 61     # as_of minus 0..60 months
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
    out["roll6"] = P[:, :N_ROLL] / P[:, 6:6 + N_ROLL] - 1                    # rolling 6M windows
    out["roll3"] = (P[:, :N_GRID - 36] / P[:, 36:]) ** (1 / 3) - 1           # rolling 3Y windows (annualised)
    return out


def _dd_values(v: np.ndarray) -> tuple[float, float]:
    """(max drawdown, point return) of a value path."""
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return np.nan, np.nan
    return float((v / np.maximum.accumulate(v) - 1).min()), float(v[-1] / v[0] - 1)


def dd_window(s: pd.Series, as_of: pd.Timestamp, days: int) -> tuple[float, float]:
    start = as_of - pd.Timedelta(days=days)
    w = pd.concat([s[s.index <= start].iloc[-1:], s[(s.index > start) & (s.index <= as_of)]])
    return _dd_values(w.to_numpy())


def dd_blend(parts: list[tuple[pd.Series, float]], as_of: pd.Timestamp, days: int) -> tuple[float, float]:
    start = as_of - pd.Timedelta(days=days)
    cols = []
    for s, w in parts:
        x = pd.concat([s[s.index <= start].iloc[-1:], s[(s.index > start) & (s.index <= as_of)]])
        cols.append(x.pct_change() * w)
    r = pd.concat(cols, axis=1).sort_index().ffill().fillna(0).sum(axis=1)
    return _dd_values(np.concatenate([[1.0], (1 + r.iloc[1:]).cumprod().to_numpy()]))


def _shares(pct: np.ndarray, min_n: int = 1) -> dict:
    v = ~np.isnan(pct)
    n = v.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = {"n": n,
             "bh": ((pct < 0.5) & v).sum(1) / n, "bq": ((pct < 0.25) & v).sum(1) / n,
             "th": ((pct >= 0.5) & v).sum(1) / n, "tq": ((pct >= 0.75) & v).sum(1) / n}
    for k in ("bh", "bq", "th", "tq"):
        d[k] = np.where(n >= min_n, d[k], np.nan)
    return d


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
def group_metrics(P: np.ndarray, bench: dict | None, dd: dict | None = None,
                  bench_dd: dict | None = None) -> dict:
    """Metrics for every fund in one peer group. `bench` = returns dict for the benchmark
    (1-row arrays) or None to use the peer median. `dd` = per-fund drawdown arrays
    {'dd_1m','ret_1m','dd_3m'}; `bench_dd` = benchmark drawdowns (None -> peer median)."""
    R = returns_from_grid(P)
    if bench is None:
        B = {k: np.nanmedian(v, axis=0) if v.ndim == 2 else np.nanmedian(v) for k, v in R.items()}
    else:
        B = {k: v[0] for k, v in bench.items()}
    m = {"n_peers": np.full(len(P), len(P))}
    if dd is not None:
        for w in ("1m", "3m"):
            x = dd[f"dd_{w}"]
            bx = np.nanmedian(x) if bench_dd is None else bench_dd[f"dd_{w}"]
            m[f"dd_{w}"] = x
            m[f"bdd_{w}"] = np.full(len(P), bx)
            m[f"ddp_{w}"] = pct_rank(x)          # 0 = deepest fall in the category
        m["ret_1m"] = dd["ret_1m"]
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
    for tag, key, min_n in (("6", "roll6", 12), ("3y", "roll3", 12)):
        sh = _shares(pct_rank(R[key]), min_n)
        m[f"n_{tag}"] = sh["n"]
        for k in ("bh", "bq", "th", "tq"):
            m[f"sh{tag}_{k}"] = sh[k]
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
        bench, label, source, bdd = None, "Peer median (no passive proxy available)", "median", None
        if g.startswith("Passive"):
            label = "Peer median (same tracked index)"
        spec = specs[g]
        if spec[0] == "proxy":
            for k in spec[1]:
                if k in resolved:
                    bench = returns_from_grid(to_grid(resolved[k][2], dates))
                    label, source = bm.PROXIES[k][0], resolved[k][1]
                    bdd = {"dd_1m": dd_window(resolved[k][2], as_of, 30)[0],
                           "dd_3m": dd_window(resolved[k][2], as_of, 91)[0]}
                    break
        elif spec[0] == "blend":
            parts, daily = [], []
            for keys, w in spec[1]:
                k = next((k for k in keys if k in resolved), None)
                if k is None:
                    parts = None
                    break
                parts.append((to_grid(resolved[k][2], dates), w))
                daily.append((resolved[k][2], w))
            if parts:
                bdd = {"dd_1m": dd_blend(daily, as_of, 30)[0], "dd_3m": dd_blend(daily, as_of, 91)[0]}
                bench = returns_from_grid(blend_grid(parts))
                label = spec[2]
                source = " | ".join(next(resolved[k][1] for k in keys if k in resolved) for keys, _ in spec[1])
        if bench is not None and np.isnan(bench["1y"][0]):
            bench, label, source, bdd = None, "Peer median (proxy lacked history)", "median", None
        w1 = [dd_window(navs[c], as_of, 30) for c in pg["code"]]
        w3 = [dd_window(navs[c], as_of, 91) for c in pg["code"]]
        dd = {"dd_1m": np.array([x[0] for x in w1]), "ret_1m": np.array([x[1] for x in w1]),
              "dd_3m": np.array([x[0] for x in w3])}
        met = group_metrics(P, bench, dd, bdd)
        roll_pct, roll_exc = met.pop("_roll_pct"), met.pop("_roll_exc")
        df = pd.DataFrame(met)
        df.insert(0, "code", pg["code"].to_numpy())
        df.insert(1, "fund", pg["name"].to_numpy())
        df.insert(2, "amc", pg["amc_short"].to_numpy())
        df.insert(3, "group", g)
        df.insert(4, "asset", pg["asset"].to_numpy())
        df["first_nav"] = [navs[c].index[0] for c in pg["code"]]
        df["age_m"] = (as_of - df["first_nav"]).dt.days / 30.44
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

    # New funds: too little history for rolling windows. Soft flag from whichever of 3M/6M/1Y exist.
    new = ~enough & df["age_m"].notna()
    avail = df[["pct_3m", "pct_6m", "pct_1y"]]
    n_av = avail.notna().sum(axis=1)
    all_low = ((avail.fillna(1) < 0.25) | avail.isna()).all(axis=1)
    all_high = ((avail.fillna(1) >= 0.75) | avail.isna()).all(axis=1)
    s[new] = "New fund - neutral"
    s[new & (df["age_m"] < 6)] = "New fund - deploying (<6M)"
    s[new & (df["age_m"] < 3)] = "New fund - too new (<3M)"
    s[new & (n_av >= 2) & all_high] = "New fund - strong"
    s[new & (n_av >= 2) & all_low] = "New fund - watch"
    s[df["age_m"].isna()] = "No NAV data"
    if p.exclude_cash:
        s[df["group"].isin(CASH_GROUPS) & ~s.eq("No NAV data")] = "Not assessed (cash-like)"
    return s


def momentum_badge(df: pd.DataFrame, status: pd.Series) -> pd.Series:
    """Short-term (3M & 6M) context for the long-term lists."""
    rec = ((df["pct_3m"] >= 0.5) & (df["pct_6m"] >= 0.5) & (df["ex_3m"] > TOL) & (df["ex_6m"] > TOL))
    fade = ((df["pct_3m"] < 0.5) & (df["pct_6m"] < 0.5) & (df["ex_3m"] < -TOL) & (df["ex_6m"] < -TOL))
    out = pd.Series("", index=df.index)
    out[status == "Consistent underperformer (exit)"] = "Still weak"
    out[(status == "Consistent underperformer (exit)") & rec] = "Recovering"
    out[status == "Consistent outperformer"] = "Holding up"
    out[(status == "Consistent outperformer") & fade] = "Fading"
    return out


def sharp_fall(df: pd.DataFrame, window: str = "1m", min_dd: float = 0.03) -> pd.Series:
    """Deep drawdown in the category's worst 10% and worse than the benchmark's drawdown."""
    return (df[f"ddp_{window}"] < 0.10) & (df[f"dd_{window}"] < df[f"bdd_{window}"]) & (df[f"dd_{window}"] <= -min_dd)


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
