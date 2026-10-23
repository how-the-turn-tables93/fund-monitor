"""Bibhab Capital - Fund Performance Monitor (Streamlit)."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from fundmonitor import amfi, analytics, holdings
from fundmonitor.analytics import (Params, classify, momentum_badge, quartile_label, sharp_fall,
                                   suggest_replacement)

ROOT = Path(__file__).parent
DARK, MID, LIGHT, PALE = "#006874", "#0099AB", "#9ADBDF", "#E6F5F6"
RED, GREEN, AMBER, PURPLE = "#B4372D", "#1E8E5A", "#C98400", "#6B4C9A"

st.set_page_config(page_title="Fund Performance Monitor | Bibhab Capital",
                   page_icon=Image.open(ROOT / "assets" / "mark.png"), layout="wide")

st.markdown(f"""
<style>
  h1, h2, h3 {{ color: {DARK}; }}
  .block-container {{ padding-top: 1.4rem; max-width: 1500px; }}
  .tagline {{ color: {DARK}; font-size: 1.05rem; margin-top: -0.6rem; }}
  .kpi {{ background: {PALE}; border-radius: 14px; padding: 14px 18px; border-left: 6px solid var(--c); }}
  .kpi .t {{ font-size: .78rem; color: #3b5559; text-transform: uppercase; letter-spacing: .04em; }}
  .kpi .n {{ font-size: 1.8rem; font-weight: 700; color: {DARK}; line-height: 1.15; }}
  .kpi .s {{ font-size: .8rem; color: #3b5559; }}
  button[data-baseweb="tab"] {{ font-weight: 600; }}
  div[data-testid="stSidebar"] {{ background: {PALE}; }}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=6 * 3600, show_spinner="Loading AMFI scheme universe…")
def get_universe() -> pd.DataFrame:
    return amfi.load_universe()


def header():
    c1, c2 = st.columns([1, 4], vertical_alignment="center")
    c1.image(str(ROOT / "assets" / "logo.png"), width=210)
    c2.markdown("# Fund Performance Monitor")
    c2.markdown('<div class="tagline">Peer-quartile and benchmark tracking of client scheme holdings, '
                'built on public AMFI / mfapi.in data</div>', unsafe_allow_html=True)


def kpi(col, title, n, sub, color):
    col.markdown(f'<div class="kpi" style="--c:{color}"><div class="t">{title}</div>'
                 f'<div class="n">{n}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ tables
def build_table(frame: pd.DataFrame, p: Params, kind: str, metrics: pd.DataFrame) -> pd.DataFrame:
    """kind: exit | watch | out_recent | out_cons | new | all"""
    q = p.mode == "quartile"
    lo, hi = ("bottom-quartile", "top-quartile") if q else ("bottom-half", "top-half")
    t = frame.copy()
    t["Fund"], t["AMC"], t["Category"] = t["fund"], t["amc"], t["group"]
    t["AUM (₹ Cr)"] = t["aum"] / 1e7
    t["Weight %"] = t["weight_pct"]
    t["1Y %"], t["1Y vs BM"] = t["ret_1y"] * 100, t["ex_1y"] * 100
    t["3Y % p.a."], t["3Y vs BM"] = t["ret_3y"] * 100, t["ex_3y"] * 100
    t["3M vs BM"], t["6M vs BM"] = t["ex_3m"] * 100, t["ex_6m"] * 100
    t["3M %"], t["6M %"] = t["ret_3m"] * 100, t["ret_6m"] * 100
    for h in ("3m", "6m", "1y", "3y"):
        t[f"{h.upper()} quartile"] = t[f"pct_{h}"].map(quartile_label)
    t["Momentum"] = momentum_badge(t, t["status"])
    t["Status"] = t["status"]
    t["Age (months)"] = t["age_m"]
    t["1M max DD %"] = t["dd_1m"] * 100
    t[f"% 1Y windows {lo}"] = (t["sh_bq"] if q else t["sh_bh"]) * 100
    t["% 1Y windows below BM"] = t["sh_lag"] * 100
    t[f"% 6M windows {lo}"] = (t["sh6_bq"] if q else t["sh6_bh"]) * 100
    t[f"% 3Y windows {lo}"] = (t["sh3y_bq"] if q else t["sh3y_bh"]) * 100
    t[f"% 1Y windows {hi}"] = (t["sh_tq"] if q else t["sh_th"]) * 100
    t["% 1Y windows above BM"] = t["sh_lead"] * 100
    t[f"% 6M windows {hi}"] = (t["sh6_tq"] if q else t["sh6_th"]) * 100
    t[f"% 3Y windows {hi}"] = (t["sh3y_tq"] if q else t["sh3y_th"]) * 100
    t["Benchmark"] = t["benchmark"]

    ident = ["Fund", "AMC", "Category", "AUM (₹ Cr)", "Weight %"]
    long_ret = ["1Y %", "1Y vs BM", "3Y % p.a.", "3Y vs BM"]
    short = ["3M vs BM", "6M vs BM"]
    rep_cols = ["Replace: best in category", "…why", "Replace: same AMC", "…category", "…why "]
    if kind in ("exit", "watch", "new"):
        rows = t if kind != "new" else t[t["status"] == "New fund - watch"]
        keys = ["same_cat_fund", "same_cat_why", "same_amc_fund", "same_amc_cat", "same_amc_why"]
        rep = pd.DataFrame([suggest_replacement(metrics, int(c), p.min_windows) for c in rows["code"]],
                           index=rows.index).reindex(columns=keys)
        t["Replace: best in category"] = rep["same_cat_fund"].replace("", "- none better found -")
        t["…why"], t["Replace: same AMC"] = rep["same_cat_why"], rep["same_amc_fund"].replace("", "- none better found -")
        t["…category"], t["…why "] = rep["same_amc_cat"], rep["same_amc_why"]
    if kind == "exit":
        cols = ident + ["Momentum"] + long_ret + short + ["1Y quartile", "3Y quartile",
                f"% 1Y windows {lo}", "% 1Y windows below BM", f"% 6M windows {lo}", f"% 3Y windows {lo}"] + rep_cols
    elif kind == "watch":
        cols = ident + long_ret + short + ["1Y quartile", "3Y quartile", f"% 1Y windows {lo}",
                "% 1Y windows below BM", f"% 6M windows {lo}"] + rep_cols
    elif kind == "out_recent":
        cols = ident + long_ret + short + ["1Y quartile", "3Y quartile", f"% 1Y windows {hi}", "% 1Y windows above BM"]
    elif kind == "out_cons":
        cols = ident + ["Momentum"] + long_ret + short + ["1Y quartile", "3Y quartile", f"% 1Y windows {hi}",
                "% 1Y windows above BM", f"% 6M windows {hi}", f"% 3Y windows {hi}"]
    elif kind == "new":
        cols = ident + ["Status", "Age (months)", "3M %", "3M quartile", "6M %", "6M quartile", "1Y %",
                        "1Y quartile", "1Y vs BM"] + rep_cols
    else:
        cols = ident + ["Status", "Momentum", "1Y %", "1Y vs BM", "3Y % p.a.", "3Y vs BM", "1Y quartile",
                        "3Y quartile", "1M max DD %"]
    cols += ["Benchmark"]
    out = t[cols].sort_values("AUM (₹ Cr)", ascending=False).reset_index(drop=True)
    if kind == "new":
        out[rep_cols] = out[rep_cols].fillna("")
        out["_o"] = out["Status"].map({"New fund - watch": 0, "New fund - strong": 1}).fillna(2)
        out = out.sort_values(["_o", "AUM (₹ Cr)"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)
        if (out["Status"] != "New fund - watch").all():
            out = out.drop(columns=rep_cols)
        else:
            blank = out["Status"] != "New fund - watch"
            for c in rep_cols:
                out.loc[blank, c] = ""
    return out


def show_table(df: pd.DataFrame, key: str):
    num = {c: st.column_config.NumberColumn(format="%.1f") for c in df.columns
           if c.endswith("vs BM") or c.endswith(" %") or c in ("3Y % p.a.", "Weight %", "Age (months)", "1M max DD %")
           or c.startswith("Max DD") or c.startswith("Benchmark DD") or c.startswith("Return")}
    num["AUM (₹ Cr)"] = st.column_config.NumberColumn(format="%.2f")
    for c in df.columns:
        if c.startswith("% "):
            num[c] = st.column_config.ProgressColumn(c, format="%.0f%%", min_value=0, max_value=100)
    st.dataframe(df, hide_index=True, width="stretch", column_config=num, key=key,
                 height=min(60 + 36 * len(df), 620))


def to_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            for i, col in enumerate(d.columns, 1):
                ws.column_dimensions[ws.cell(1, i).column_letter].width = min(48, max(12, len(str(col)) + 2))
    return buf.getvalue()


# ------------------------------------------------------------------ drill-down
def drilldown(res: analytics.Result, code: int, name: str):
    if code not in res.rolls:
        st.info("No rolling history available for this fund.")
        return
    pct, exc = res.rolls[code]
    dates = analytics.grid_dates(res.as_of)[:analytics.N_ROLL][::-1]
    pct, exc = pct[::-1] * 100, exc[::-1] * 100
    c1, c2 = st.columns(2)
    f1 = go.Figure()
    f1.add_hrect(y0=0, y1=25, fillcolor=RED, opacity=.08, line_width=0)
    f1.add_hrect(y0=75, y1=100, fillcolor=GREEN, opacity=.08, line_width=0)
    f1.add_hline(y=50, line_dash="dot", line_color="#889")
    f1.add_trace(go.Scatter(x=dates, y=pct, mode="lines+markers", line=dict(color=DARK, width=3), name="Percentile"))
    f1.update_layout(title="Rolling 1Y return - percentile in peer category (100 = best)", height=360,
                     yaxis=dict(range=[0, 100]), margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white")
    c1.plotly_chart(f1, width="stretch")
    f2 = go.Figure(go.Bar(x=dates, y=exc, marker_color=[GREEN if v >= 0 else RED for v in np.nan_to_num(exc)]))
    f2.update_layout(title="Rolling 1Y return minus benchmark (% points)", height=360,
                     margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white")
    c2.plotly_chart(f2, width="stretch")


# ------------------------------------------------------------------ main
header()
st.sidebar.image(str(ROOT / "assets" / "mark.png"), width=46)
st.sidebar.markdown("### Upload holdings")
up = st.sidebar.file_uploader("AUM By Scheme report (.xlsx)", type=["xlsx", "xls"])
st.sidebar.markdown("### Rules")
mode = st.sidebar.radio("Rank test", ["Bottom / top half", "Bottom / top quartile"],
                        help="Which part of the sub-category counts as under/outperformance.")
p = Params(mode="quartile" if "quartile" in mode else "half")
if p.mode == "half":
    p.thr_rank_half = st.sidebar.slider("Consistency: % of rolling 1Y windows in bottom/top half", 50, 100, 70, 5) / 100
else:
    p.thr_rank_quartile = st.sidebar.slider("Consistency: % of rolling 1Y windows in bottom/top quartile", 25, 100, 50, 5) / 100
p.thr_bench = st.sidebar.slider("Consistency: % of windows below/above benchmark", 50, 100, 70, 5) / 100
p.min_windows = st.sidebar.slider("Min. months of rolling history", 12, 36, 24, 3,
                                  help="Funds younger than this go to the New funds tab instead.")
p.exclude_cash = st.sidebar.checkbox("Skip liquid / overnight / money-market / arbitrage", True)
min_aum = st.sidebar.number_input("Ignore holdings below (₹)", 0, 10_000_000, 100_000, 10_000,
                                  help="Tiny residual balances add noise to the lists.")
st.sidebar.markdown("### Sharp-fall alert")
dd_win = st.sidebar.radio("Drawdown window", ["1M", "3M"], horizontal=True).lower()
min_dd = st.sidebar.slider("Min. drawdown to flag (%)", 1, 15, 3) / 100

demo = os.environ.get("FM_DEMO_FILE")  # local testing only: preload a workbook without uploading
if up is not None:
    data = up.getvalue()
elif demo and Path(demo).exists():
    data = Path(demo).read_bytes()
else:
    st.info("Upload the **AUM By Scheme** report in the sidebar to begin. The app matches each scheme to AMFI, "
            "pulls NAV history for every peer in its sub-category, and ranks it against peers and its benchmark.")
    st.stop()

key = hashlib.md5(data).hexdigest() + str(min_aum)
if st.session_state.get("key") != key:
    try:
        raw = holdings.parse_aum_file(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the file: {e}")
        st.stop()
    uni = get_universe()
    h = holdings.aggregate_holdings(raw, min_aum)
    h = holdings.match_holdings(h, uni)
    bar = st.progress(0.0, "Starting…")
    res = analytics.run_analysis(h, uni, progress=lambda f, m: bar.progress(min(f, 1.0), m))
    bar.empty()
    st.session_state.update(key=key, res=res)
res: analytics.Result = st.session_state["res"]

# join holdings -> metrics -> status
H = res.holdings.copy()
H["code"] = H["code"].astype("Int64")
M = res.metrics.merge(res.bench_info[["group", "benchmark"]], on="group", how="left")
J = H[H["code"].notna()].merge(M, on="code", how="left", suffixes=("_h", ""), validate="m:1")
J["code"] = J["code"].astype(int)
J["status"] = classify(J, p)
J["sharp"] = sharp_fall(J, dd_win, min_dd) & ~J["status"].isin(["No NAV data"])
total_aum = H["aum"].sum()
st.caption(f"NAV as of **{res.as_of:%d %b %Y}** · {len(H)} holdings (₹{total_aum / 1e7:,.0f} Cr) · "
           f"{int(H['code'].notna().sum())} matched to AMFI · {len(res.metrics):,} peer funds ranked")

LISTS = [("Consistent underperformer (exit)", "Exit list", RED, "exit"),
         ("Recent underperformer (watch)", "Watch list", AMBER, "watch"),
         ("Recent outperformer", "Recent outperformers", MID, "out_recent"),
         ("Consistent outperformer", "Consistent outperformers", GREEN, "out_cons")]
cnt = lambda s: int((J["status"] == s).sum())  # noqa: E731
kc = st.columns(6)
for c, (s, title, color, _) in zip(kc, LISTS):
    sub = J[J["status"] == s]
    kpi(c, title, len(sub), f"₹{sub['aum'].sum() / 1e7:,.1f} Cr · {sub['aum'].sum() / total_aum * 100:.1f}% of AUM", color)
newf = J[J["status"] == "New fund - watch"]
kpi(kc[4], "New funds on watch", len(newf), f"₹{newf['aum'].sum() / 1e7:,.1f} Cr · {newf['aum'].sum() / total_aum * 100:.1f}% of AUM", PURPLE)
sf = J[J["sharp"]]
kpi(kc[5], f"Sharp falls ({dd_win.upper()})", len(sf), f"₹{sf['aum'].sum() / 1e7:,.1f} Cr · {sf['aum'].sum() / total_aum * 100:.1f}% of AUM", "#444")
st.write("")

n_new = int(J["status"].isin(["New fund - watch", "New fund - strong"]).sum())
tabs = st.tabs([f"🔴 Exit ({cnt(LISTS[0][0])})", f"🟠 Watch ({cnt(LISTS[1][0])})",
                f"🔵 Recent outperformers ({cnt(LISTS[2][0])})", f"🟢 Consistent outperformers ({cnt(LISTS[3][0])})",
                f"🆕 New funds ({n_new})", f"📉 Sharp falls ({len(sf)})",
                "🔍 Fund drill-down", "📋 All holdings", "ℹ️ Method & data"])
captions = [
    "Persistently in the lower tier of their sub-category **and** behind their benchmark across rolling 1Y windows, "
    "with 3Y return also behind the benchmark. **Momentum** badge: *Recovering* = 3M and 6M both top-half and ahead of "
    "the benchmark; *Still weak* otherwise. Recovering funds stay on the list so you can decide case by case.",
    "Not (yet) persistent, but both 6M and 1Y returns sit in the lower tier of the sub-category and behind the "
    "benchmark. Monitor closely.",
    "6M and 1Y returns in the upper tier of the sub-category and ahead of the benchmark, without a long enough track "
    "record of leadership.",
    "Persistently in the upper tier of their sub-category and ahead of their benchmark across rolling 1Y windows. "
    "**Momentum** badge: *Fading* = 3M and 6M both bottom-half and behind the benchmark.",
]
exports = {}
for i, (s, title, color, kind) in enumerate(LISTS):
    with tabs[i]:
        st.caption(captions[i])
        sub = J[J["status"] == s]
        if sub.empty:
            st.success("No funds in this list under the current rules.")
            continue
        tbl = build_table(sub, p, kind, res.metrics)
        exports[title] = tbl
        show_table(tbl, f"t{i}")

with tabs[4]:
    st.caption(f"Funds younger than {p.min_windows + 12} months (too short for rolling windows). **Watch** = every available "
               "horizon of 3M / 6M / 1Y is in the bottom quartile of the category (needs at least two horizons, i.e. "
               "6M+ of history); **Strong** = all in the top quartile. A soft flag, not an exit signal: launch-phase cash "
               "drag and the small sample make early ranks noisy. Funds under 6 months are shown for reference only.")
    show_all = st.checkbox("Also show new funds that are neither watch nor strong", False)
    nf = J[J["status"].str.startswith("New fund")]
    if not show_all:
        nf = nf[nf["status"].isin(["New fund - watch", "New fund - strong"])]
    if nf.empty:
        st.success("No new funds flagged.")
    else:
        tbl = build_table(nf, p, "new", res.metrics)
        exports["New funds"] = tbl
        show_table(tbl, "tnew")

with tabs[5]:
    st.caption(f"Holdings whose worst peak-to-trough NAV fall over the last {'30 days' if dd_win == '1m' else '3 months'} "
               f"is at least {min_dd * 100:.0f}%, in the worst 10% of their category **and** deeper than their "
               "benchmark's drawdown. The relative tests stop a market-wide fall from flagging everything. "
               "Change window and depth in the sidebar.")
    show_all_dd = st.checkbox("Show all holdings with a drawdown figure", False)
    d = J[J["dd_" + dd_win].notna() & (J["status"] != "No NAV data")]
    if not show_all_dd:
        d = d[d["sharp"]]
    if d.empty:
        st.success("No sharp falls under the current settings.")
    else:
        w = dd_win.upper()
        tbl = pd.DataFrame({
            "Fund": d["fund"], "AMC": d["amc"], "Category": d["group"], "AUM (₹ Cr)": d["aum"] / 1e7,
            "Weight %": d["weight_pct"], f"Max DD {w} %": d["dd_" + dd_win] * 100,
            f"Benchmark DD {w} %": d["bdd_" + dd_win] * 100,
            f"Return {w} %": (d["ret_1m"] if dd_win == "1m" else d["ret_3m"]) * 100,
            "Category rank (0 = deepest fall)": (d["ddp_" + dd_win] * 100).round(0),
            "Status": d["status"], "Momentum": momentum_badge(d, d["status"]), "Benchmark": d["benchmark"],
        }).sort_values(["AUM (₹ Cr)"], ascending=False).reset_index(drop=True)
        exports[f"Sharp falls {w}"] = tbl
        show_table(tbl, "tfall")

with tabs[6]:
    options = J[J["fund"].notna()].sort_values("aum", ascending=False)
    names = dict(zip(options["code"].astype(int), options["fund"]))
    pick = st.selectbox("Fund", list(names), format_func=lambda c: names[c])
    r = options[options["code"] == pick].iloc[0]
    st.markdown(f"**{r['fund']}** · {r['amc']} · {r['group']} · status: **{r['status']}**  \nBenchmark: {r['benchmark']}")
    drilldown(res, int(pick), r["fund"])
    hz = pd.DataFrame({"Horizon": ["3M", "6M", "1Y", "3Y p.a.", "5Y p.a."],
                       "Fund %": [r[f"ret_{k}"] * 100 for k in analytics.HORIZONS],
                       "Benchmark %": [r[f"bench_{k}"] * 100 for k in analytics.HORIZONS],
                       "Excess (pp)": [r[f"ex_{k}"] * 100 for k in analytics.HORIZONS],
                       "Peer percentile (100 = best)": [r[f"pct_{k}"] * 100 for k in analytics.HORIZONS]}).round(1)
    st.dataframe(hz, hide_index=True)
    st.caption(f"Max drawdown 1M: {r['dd_1m'] * 100:.1f}% (benchmark {r['bdd_1m'] * 100:.1f}%) · "
               f"3M: {r['dd_3m'] * 100:.1f}% (benchmark {r['bdd_3m'] * 100:.1f}%)")
with tabs[7]:
    allt = build_table(J[J["fund"].notna()], p, "all", res.metrics)
    exports["All holdings"] = allt
    show_table(allt, "tall")
with tabs[8]:
    st.markdown("""
**Universe & matching.** Schemes in the uploaded file are collapsed to one holding per scheme (IDCW/Growth rows
merged, tiny/zero balances dropped) and matched by name to the AMFI NAV file. Peers are all *Regular-plan Growth*
open-ended schemes in the same AMFI sub-category (Sectoral/Thematic funds are split by theme; passive funds are
grouped by tracked index). Closed/merged schemes with stale NAVs are excluded.

**Rolling consistency.** For each of the last 36 month-ends the app computes every fund's trailing 1Y return, its
percentile rank among category peers, and its excess over the benchmark. *Consistent* means the fund was in the
lower (or upper) tier in at least the chosen share of those windows **and** behind (or ahead of) the benchmark in at
least the chosen share, **and** its 3Y annualised return is on the same side of the benchmark. Shares of rolling
**6M** windows (36) and rolling **3Y** windows (up to 25) are shown as extra columns for context, not as hard rules.

**Momentum badges.** On the Exit list, *Recovering* = 3M and 6M both top-half in the category and ahead of the benchmark.
On Consistent outperformers, *Fading* = 3M and 6M both bottom-half and behind the benchmark.

**Recent.** 6M and 1Y returns both in the lower (upper) tier of the category and both behind (ahead of) the benchmark,
for funds not already on a consistent list and with enough history.

**New funds.** Funds without enough history for rolling windows are assessed separately from 3M / 6M / 1Y point
returns versus peers over the same dates (see the New funds tab).

**Sharp falls.** Maximum peak-to-trough NAV decline over 30 days (or 3 months): flagged when at least the chosen depth,
in the worst 10% of the category and worse than the benchmark's own drawdown.

**Replacement.** For flagged funds the app proposes (a) the best-scoring fund in the same sub-category and (b) the best
fund from the same AMC (same sub-category when available, otherwise a comparable category). Candidates must have enough
history, be in the top half of the category in most rolling windows and not trail the benchmark over 3Y. Score = 30% 3Y rank
+ 25% 1Y rank + 15% 5Y rank + 30% share of rolling windows in the top half. This is a screening aid, not advice; check
portfolio fit, tax and exit loads before switching.
""")
    st.markdown("**Benchmarks used**")
    st.dataframe(res.bench_info.rename(columns={"group": "Category", "benchmark": "Benchmark", "proxy_fund": "Proxy fund(s) (public NAV)", "peers": "Peers"}),
                 hide_index=True, width="stretch")
    st.caption("Official TRI series are not freely scriptable, so each benchmark is proxied by a passive fund/ETF tracking "
               "the same index (≈0.1-0.3% p.a. below TRI). Blends approximate CRISIL hybrid composites with monthly "
               "rebalancing. Categories with no proxy use the peer median. Edit `fundmonitor/benchmarks.py` to change.")
    um = res.holdings[res.holdings["code"].isna()]
    st.markdown(f"**Unmatched schemes ({len(um)})**")
    if len(um):
        st.dataframe(um[["scheme", "aum"]].rename(columns={"scheme": "Scheme", "aum": "AUM (₹)"}), hide_index=True)
    else:
        st.caption("Every holding was matched to an AMFI scheme.")

if exports:
    st.sidebar.download_button("⬇ Download results (Excel)", to_excel(exports), "fund_monitor_results.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
