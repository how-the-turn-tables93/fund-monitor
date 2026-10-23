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
from fundmonitor.analytics import Params, classify, quartile_label, suggest_replacement

ROOT = Path(__file__).parent
DARK, MID, LIGHT, PALE = "#006874", "#0099AB", "#9ADBDF", "#E6F5F6"
RED, GREEN, AMBER = "#B4372D", "#1E8E5A", "#C98400"

st.set_page_config(page_title="Fund Performance Monitor | Bibhab Capital",
                   page_icon=Image.open(ROOT / "assets" / "mark.png"), layout="wide")

st.markdown(f"""
<style>
  h1, h2, h3 {{ color: {DARK}; }}
  .block-container {{ padding-top: 1.4rem; max-width: 1500px; }}
  .tagline {{ color: {DARK}; font-size: 1.05rem; margin-top: -0.6rem; }}
  .kpi {{ background: {PALE}; border-radius: 14px; padding: 14px 18px; border-left: 6px solid var(--c); }}
  .kpi .t {{ font-size: .82rem; color: #3b5559; text-transform: uppercase; letter-spacing: .04em; }}
  .kpi .n {{ font-size: 1.9rem; font-weight: 700; color: {DARK}; line-height: 1.15; }}
  .kpi .s {{ font-size: .85rem; color: #3b5559; }}
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
def build_table(frame: pd.DataFrame, p: Params, replacements: bool, metrics: pd.DataFrame) -> pd.DataFrame:
    q = p.mode == "quartile"
    t = frame.copy()
    t["Fund"] = t["fund"]
    t["AMC"] = t["amc"]
    t["Category"] = t["group"]
    t["AUM (₹ Cr)"] = t["aum"] / 1e7
    t["Weight %"] = t["weight_pct"]
    t["1Y %"] = t["ret_1y"] * 100
    t["1Y vs BM"] = t["ex_1y"] * 100
    t["3Y % p.a."] = t["ret_3y"] * 100
    t["3Y vs BM"] = t["ex_3y"] * 100
    t["1Y quartile"] = t["pct_1y"].map(quartile_label)
    t["3Y quartile"] = t["pct_3y"].map(quartile_label)
    low_lbl = "bottom-quartile" if q else "bottom-half"
    t[f"% windows {low_lbl}"] = (t["sh_bq"] if q else t["sh_bh"]) * 100
    t["% windows below BM"] = t["sh_lag"] * 100
    t[f"% windows top-{'quartile' if q else 'half'}"] = (t["sh_tq"] if q else t["sh_th"]) * 100
    t["% windows above BM"] = t["sh_lead"] * 100
    t["Benchmark"] = t["benchmark"]
    cols = ["Fund", "AMC", "Category", "AUM (₹ Cr)", "Weight %", "1Y %", "1Y vs BM", "3Y % p.a.", "3Y vs BM",
            "1Y quartile", "3Y quartile"]
    if replacements:
        cols += [f"% windows {low_lbl}", "% windows below BM"]
        rep = [suggest_replacement(metrics, int(c), p.min_windows) for c in t["code"]]
        rd = pd.DataFrame(rep, index=t.index)
        t["Replace: best in category"] = rd["same_cat_fund"].replace("", "- none better found -")
        t["…why"] = rd["same_cat_why"]
        t["Replace: same AMC"] = rd["same_amc_fund"].replace("", "- none better found -")
        t["…category"] = rd["same_amc_cat"]
        t["…why "] = rd["same_amc_why"]
        cols += ["Replace: best in category", "…why", "Replace: same AMC", "…category", "…why "]
    else:
        cols += [f"% windows top-{'quartile' if q else 'half'}", "% windows above BM"]
    cols += ["Benchmark"]
    return t[cols].sort_values("AUM (₹ Cr)", ascending=False).reset_index(drop=True)


def show_table(df: pd.DataFrame, key: str):
    num = {c: st.column_config.NumberColumn(format="%.1f") for c in df.columns
           if c.startswith("%") or c.endswith("vs BM") or c in ("1Y %", "3Y % p.a.", "Weight %")}
    num["AUM (₹ Cr)"] = st.column_config.NumberColumn(format="%.2f")
    for c in df.columns:
        if c.startswith("% windows"):
            num[c] = st.column_config.ProgressColumn(c, format="%.0f%%", min_value=0, max_value=100)
    st.dataframe(df, hide_index=True, use_container_width=True, column_config=num, key=key,
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
    c1.plotly_chart(f1, use_container_width=True)
    f2 = go.Figure(go.Bar(x=dates, y=exc, marker_color=[GREEN if v >= 0 else RED for v in np.nan_to_num(exc)]))
    f2.update_layout(title="Rolling 1Y return minus benchmark (% points)", height=360,
                     margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white")
    c2.plotly_chart(f2, use_container_width=True)


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
p.min_windows = st.sidebar.slider("Min. months of rolling history", 12, 36, 24, 3)
p.exclude_cash = st.sidebar.checkbox("Skip liquid / overnight / money-market / arbitrage", True)
min_aum = st.sidebar.number_input("Ignore holdings below (₹)", 0, 10_000_000, 100_000, 10_000,
                                  help="Tiny residual balances add noise to the lists.")

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
J["status"] = classify(J.assign(group=J["group"]), p)
J.loc[J["ret_1y"].isna(), "status"] = "Not enough history"
total_aum = H["aum"].sum()
st.caption(f"NAV as of **{res.as_of:%d %b %Y}** · {len(H)} holdings (₹{total_aum / 1e7:,.0f} Cr) · "
           f"{int(H['code'].notna().sum())} matched to AMFI · {len(res.metrics):,} peer funds ranked")

LISTS = [("Consistent underperformer (exit)", "Exit list", RED, True),
         ("Recent underperformer (watch)", "Watch list", AMBER, True),
         ("Recent outperformer", "Recent outperformers", MID, False),
         ("Consistent outperformer", "Consistent outperformers", GREEN, False)]
cols = st.columns(4)
for c, (s, title, color, _) in zip(cols, LISTS):
    sub = J[J["status"] == s]
    kpi(c, title, len(sub), f"₹{sub['aum'].sum() / 1e7:,.1f} Cr · {sub['aum'].sum() / total_aum * 100:.1f}% of AUM", color)
st.write("")

tabs = st.tabs([f"🔴 Exit ({(J['status'] == LISTS[0][0]).sum()})", f"🟠 Watch ({(J['status'] == LISTS[1][0]).sum()})",
                f"🔵 Recent outperformers ({(J['status'] == LISTS[2][0]).sum()})",
                f"🟢 Consistent outperformers ({(J['status'] == LISTS[3][0]).sum()})",
                "🔍 Fund drill-down", "📋 All holdings", "ℹ️ Method & data"])
captions = {
    0: "Persistently in the lower tier of their sub-category **and** persistently behind their benchmark "
       "across rolling 1Y windows, with 3Y return also behind the benchmark. Candidates to exit.",
    1: "Not (yet) persistent, but both 6M and 1Y returns sit in the lower tier of the sub-category and behind the "
       "benchmark. Monitor closely.",
    2: "6M and 1Y returns in the upper tier of the sub-category and ahead of the benchmark, without a long enough track record of leadership.",
    3: "Persistently in the upper tier of their sub-category and ahead of their benchmark across rolling 1Y windows.",
}
exports = {}
for i, (s, title, color, rep) in enumerate(LISTS):
    with tabs[i]:
        st.caption(captions[i])
        sub = J[J["status"] == s]
        if sub.empty:
            st.success("No funds in this list under the current rules.")
            continue
        tbl = build_table(sub, p, rep, res.metrics)
        exports[title] = tbl
        show_table(tbl, f"t{i}")
with tabs[4]:
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
with tabs[5]:
    allt = build_table(J, p, False, res.metrics)
    allt.insert(0, "Status", J.sort_values("aum", ascending=False)["status"].to_numpy()[:len(allt)])
    exports["All holdings"] = allt
    show_table(allt, "tall")
with tabs[6]:
    st.markdown("""
**Universe & matching.** Schemes in the uploaded file are collapsed to one holding per scheme (IDCW/Growth rows
merged, tiny/zero balances dropped) and matched by name to the AMFI NAV file. Peers are all *Regular-plan Growth*
open-ended schemes in the same AMFI sub-category (Sectoral/Thematic funds are split by theme; passive funds are
grouped by tracked index). Closed/merged schemes with stale NAVs are excluded.

**Rolling consistency.** For each of the last 36 month-ends the app computes every fund's trailing 1Y return, its
percentile rank among category peers, and its excess over the benchmark. *Consistent* means the fund was in the
lower (or upper) tier in at least the chosen share of those windows **and** behind (or ahead of) the benchmark in at
least the chosen share, **and** its 3Y annualised return is on the same side of the benchmark.

**Recent.** 6M and 1Y returns both in the lower (upper) tier of the category and both behind (ahead of) the benchmark,
for funds not already on a consistent list.

**Replacement.** For flagged funds the app proposes (a) the best-scoring fund in the same sub-category and (b) the best
fund from the same AMC (same sub-category when available, otherwise same asset class). Candidates must have enough history,
be in the top half of the category in most rolling windows and not trail the benchmark over 3Y. Score = 30% 3Y rank + 25% 1Y
rank + 15% 5Y rank + 30% share of rolling windows in the top half. This is a screening aid, not advice; check portfolio fit,
AUM, tax and exit loads before switching.
""")
    st.markdown("**Benchmarks used**")
    st.dataframe(res.bench_info.rename(columns={"group": "Category", "benchmark": "Benchmark", "proxy_fund": "Proxy fund(s) (public NAV)", "peers": "Peers"}),
                 hide_index=True, use_container_width=True)
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
