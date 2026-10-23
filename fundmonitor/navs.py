"""NAV history from mfapi.in (public), with an on-disk cache and parallel fetching."""
from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

API = "https://api.mfapi.in/mf/{code}"
CACHE = Path(os.environ.get("FM_CACHE", Path(tempfile.gettempdir()) / "bibhab_nav_cache"))
TTL_HOURS = 18


def _path(code: int) -> Path:
    return CACHE / f"{code}.csv"


def _fresh(p: Path) -> bool:
    return p.exists() and (time.time() - p.stat().st_mtime) < TTL_HOURS * 3600


def fetch_nav(code: int, session: requests.Session | None = None) -> pd.Series | None:
    p = _path(code)
    if _fresh(p):
        try:
            s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
            return s
        except Exception:  # noqa: BLE001
            pass
    sess = session or requests
    for attempt in range(4):
        try:
            r = sess.get(API.format(code=code), timeout=30)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if not data:
                    return None
                s = pd.Series(
                    [float(x["nav"]) for x in data],
                    index=pd.to_datetime([x["date"] for x in data], format="%d-%m-%Y"),
                    name=code,
                ).sort_index()
                s = s[~s.index.duplicated(keep="last")]
                CACHE.mkdir(parents=True, exist_ok=True)
                s.to_csv(p)
                return s
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def fetch_many(codes, progress=None, workers: int = 16) -> dict[int, pd.Series]:
    codes = list(dict.fromkeys(int(c) for c in codes))
    out: dict[int, pd.Series] = {}
    sess = requests.Session()
    sess.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=workers))
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_nav, c, sess): c for c in codes}
        for i, f in enumerate(as_completed(futs), 1):
            s = f.result()
            if s is not None and len(s) > 20:
                out[futs[f]] = s
            if progress:
                progress(i, len(codes))
    return out
