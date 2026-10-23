# Bibhab Capital – Fund Performance Monitor

Upload the **AUM By Scheme** report and get four watch-lists of the schemes your clients hold:

| List | Meaning |
|---|---|
| 🔴 **Exit** (consistent underperformers) | In the bottom half (or quartile) of their sub-category **and** behind their benchmark in most rolling 1Y windows over the last 3 years; 3Y return also behind benchmark |
| 🟠 **Watch** (recent underperformers) | 6M and 1Y returns both in the lower tier of the sub-category and behind the benchmark, not yet persistent |
| 🟢 **Consistent outperformers** | Mirror of Exit |
| 🔵 **Recent outperformers** | Mirror of Watch |

Exit and Watch rows come with a suggested **replacement**: the best fund in the same sub-category, and the best fund
from the same AMC (same sub-category if possible, else a comparable one). A fund drill-down shows its rolling peer
percentile and excess-vs-benchmark history. Everything can be downloaded as Excel.

## Data (all public, no API keys)
* **AMFI `NAVAll.txt`** – scheme universe, AMC, SEBI sub-category.
* **mfapi.in** – full NAV history for every scheme (holdings and all category peers).
* **Benchmarks** – NSE/BSE TRI series can't be pulled by scripts, so each benchmark is proxied by the NAV of a passive
  fund/ETF tracking that index (a few bps–0.3% p.a. below TRI). Hybrid benchmarks are monthly-rebalanced blends; categories
  with no proxy (liquid, money market, ESG, MNC…) use the peer median and are labelled so in the *Method & data* tab.
  Mapping lives in `fundmonitor/benchmarks.py`.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
First run downloads ~1,600 NAV histories (~1 min); they are cached on disk for 18 h.

## Deploy
1. Push this folder to a new GitHub repo (`app.py` at the repo root).
2. On [share.streamlit.io](https://share.streamlit.io) choose *New app*, pick the repo, main file `app.py`.
3. The uploaded workbook is processed in memory only; nothing is stored. The repo contains no client data.

## Method in brief
* Holdings are collapsed to one row per scheme (IDCW/Growth merged) and matched by name to AMFI; the Regular-plan Growth
  series is analysed. Unmatched schemes (e.g. SIF long-short funds not in the AMFI file) are listed in *Method & data*.
* Peers = Regular-Growth open-ended schemes in the same sub-category (Value/Contra/Dividend Yield pooled; Sectoral split by
  theme; passive funds grouped by tracked index). Fewer than 5 peers → not ranked.
* For each of the last 36 month-ends: trailing-1Y return → percentile among peers and excess vs benchmark.
  Excess within ±0.25pp counts as "in line".
* Thresholds (rank test half/quartile, % of windows, min history, small-holding cutoff) are sidebar controls.
* Replacement score = 30% 3Y rank + 25% 1Y rank + 15% 5Y rank + 30% share of windows in top half; the candidate must
  have enough history, beat the benchmark over 3Y and be top-half in most windows.

> Screening aid only, not investment advice. Check portfolio overlap, AUM, taxation and exit load before switching.
