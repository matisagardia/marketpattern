from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import math, os, pickle, time, httpx

app = FastAPI(title="Market Pattern Analyzer API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

AV_KEY = os.environ.get("AV_KEY", "")

# ── CACHE ─────────────────────────────────────────────────────────
_MEM = {}
_MEM_TTL  = 1800
_DISK_TTL = 86400
_DISK     = "/tmp/av_cache"
os.makedirs(_DISK, exist_ok=True)

def _cget(key):
    now = time.time()
    if key in _MEM and now - _MEM[key][0] < _MEM_TTL:
        return _MEM[key][1].copy()
    path = f"{_DISK}/{key}.pkl"
    if os.path.exists(path) and (now - os.path.getmtime(path)) < _DISK_TTL:
        try:
            with open(path,"rb") as f: df = pickle.load(f)
            _MEM[key] = (now, df)
            return df.copy()
        except Exception: pass
    return None

def _cset(key, df):
    _MEM[key] = (time.time(), df)
    try:
        with open(f"{_DISK}/{key}.pkl","wb") as f: pickle.dump(df, f)
    except Exception: pass

# ── ALPHA VANTAGE FETCH ───────────────────────────────────────────
async def _av_fetch(ticker: str, function: str, extra: dict = {}) -> dict:
    if not AV_KEY:
        raise HTTPException(status_code=500, detail="AV_KEY not set. Add it in Render → Environment Variables.")
    params = {"function": function, "symbol": ticker, "apikey": AV_KEY, "datatype": "json"}
    params.update(extra)
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get("https://www.alphavantage.co/query", params=params)
    data = r.json()
    if "Error Message" in data:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found.")
    if "Note" in data or "Information" in data:
        msg = data.get("Note") or data.get("Information","")
        raise HTTPException(status_code=429, detail=f"Alpha Vantage rate limit: {msg}")
    return data

def _parse_ts(data: dict, ts_key: str, close_key: str) -> pd.DataFrame:
    ts = data.get(ts_key, {})
    if not ts:
        raise HTTPException(status_code=500, detail=f"No time series in response. Keys: {list(data.keys())}")
    rows = []
    for date_str, vals in ts.items():
        rows.append({
            "date":  pd.to_datetime(date_str),
            "open":  float(vals.get("1. open", 0)),
            "high":  float(vals.get("2. high", 0)),
            "low":   float(vals.get("3. low", 0)),
            "close": float(vals.get(close_key, 0)),
        })
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df.dropna()

# ── GET HISTORY: weekly for long history, daily for recent ────────
async def get_history(ticker: str, years: int, prefer_daily: bool = False) -> pd.DataFrame:
    """
    Free tier strategy:
    - Weekly data: full history (~20yr), good for drawdown/seasonality/streak
    - Daily compact: last ~100 trading days, good for dayfall/volatility
    prefer_daily=True → use daily compact (shorter but daily granularity)
    """
    tk = ticker.upper().strip()
    suffix = "daily" if prefer_daily else "weekly"
    key = f"{tk}_{years}_{suffix}"

    cached = _cget(key)
    if cached is not None:
        return cached

    if prefer_daily:
        # Daily compact: ~100 most recent trading days
        data = await _av_fetch(tk, "TIME_SERIES_DAILY", {"outputsize": "compact"})
        df = _parse_ts(data, "Time Series (Daily)", "4. close")
    else:
        # Weekly: full history, free
        data = await _av_fetch(tk, "TIME_SERIES_WEEKLY")
        df = _parse_ts(data, "Weekly Time Series", "4. close")

    # Slice to requested years
    cutoff = datetime.today() - timedelta(days=years * 365 + 60)
    df = df[df.index >= cutoff]
    if len(df) < 10:
        raise HTTPException(status_code=404, detail=f"Not enough data for '{tk}' ({len(df)} rows).")

    _cset(key, df)
    return df.copy()

# ── HELPERS ───────────────────────────────────────────────────────
def safe(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))): return 0.0
    return round(float(v), 4)

def forward_return(df: pd.DataFrame, idx: int, horizon_days: int) -> Optional[float]:
    entry_date = df.index[idx]
    target_date = entry_date + timedelta(days=horizon_days)
    future = df[df.index >= target_date]
    if future.empty: return None
    ep = df.iloc[idx]["close"]
    if ep == 0: return None
    return round((future.iloc[0]["close"] / ep - 1) * 100, 2)

def summary_stats(returns: list) -> dict:
    if not returns:
        return {"event_count":0,"avg_forward_return":0.0,"median_forward_return":0.0,
                "pct_positive":0.0,"best_return":0.0,"worst_return":0.0,"std_dev":0.0}
    arr = np.array(returns)
    return {"event_count":len(returns),"avg_forward_return":safe(np.mean(arr)),
            "median_forward_return":safe(np.median(arr)),"pct_positive":safe(np.mean(arr>0)*100),
            "best_return":safe(np.max(arr)),"worst_return":safe(np.min(arr)),"std_dev":safe(np.std(arr))}

# ── TICKER SEARCH ─────────────────────────────────────────────────
@app.get("/search")
async def search_ticker(q: str):
    if len(q) < 1: return {"results": []}
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        headers = {"User-Agent": "Mozilla/5.0"}
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(url, params={"q":q,"quotesCount":10,"newsCount":0}, headers=headers)
            data = r.json()
        results = []
        for item in data.get("quotes", []):
            if item.get("quoteType") in {"EQUITY","ETF","INDEX","MUTUALFUND"}:
                results.append({"symbol":item.get("symbol",""),
                                 "name":item.get("longname") or item.get("shortname",""),
                                 "exchange":item.get("exchDisp",item.get("exchange","")),
                                 "type":item.get("quoteType","")})
        return {"results": results}
    except Exception as e:
        return {"results": [], "error": str(e)}

# ── DRAWDOWN (weekly data — full history) ─────────────────────────
@app.get("/drawdown")
async def drawdown(ticker: str, threshold: float = 30, horizon: int = 365, years: int = 10):
    ticker = ticker.upper().strip()
    df = await get_history(ticker, years, prefer_daily=False)
    closes = df["close"]
    events = []
    peak = closes.iloc[0]
    in_episode = False
    for i in range(1, len(closes)):
        price = closes.iloc[i]
        if not in_episode:
            if price > peak: peak = price
        dd_pct = (price / peak - 1) * 100
        if dd_pct <= -threshold:
            if not in_episode:
                in_episode = True
                fwd = forward_return(df, i, horizon)
                events.append({"date":df.index[i].strftime("%Y-%m-%d"),"price":safe(price),
                               "drawdown_pct":safe(dd_pct),"forward_return_pct":safe(fwd) if fwd is not None else None})
        else:
            if in_episode:
                peak = price
                in_episode = False
    valid = [e["forward_return_pct"] for e in events if e["forward_return_pct"] is not None]
    return {"ticker":ticker,"analysis_type":"drawdown","threshold":threshold,
            "horizon_days":horizon,"years":years,"events":events,"summary":summary_stats(valid)}

# ── DAY FALL (daily compact — recent ~100 days) ───────────────────
@app.get("/dayfall")
async def dayfall(ticker: str, threshold: float = 5, horizon: int = 90, years: int = 1):
    ticker = ticker.upper().strip()
    df = await get_history(ticker, years, prefer_daily=True)
    df = df.copy(); df["day_ret"] = df["close"].pct_change() * 100
    events = []
    for i in range(1, len(df)):
        day_ret = df.iloc[i]["day_ret"]
        if pd.isna(day_ret) or day_ret > -threshold: continue
        fwd = forward_return(df, i, horizon)
        events.append({"date":df.index[i].strftime("%Y-%m-%d"),"price":safe(df.iloc[i]["close"]),
                       "day_return_pct":safe(day_ret),"forward_return_pct":safe(fwd) if fwd is not None else None})
    valid = [e["forward_return_pct"] for e in events if e["forward_return_pct"] is not None]
    stats = summary_stats(valid)
    stats["avg_day_drop"] = safe(np.mean([e["day_return_pct"] for e in events])) if events else 0.0
    return {"ticker":ticker,"analysis_type":"dayfall","threshold":threshold,
            "horizon_days":horizon,"years":years,"events":events,"summary":stats}

# ── STREAK (weekly) ───────────────────────────────────────────────
@app.get("/streak")
async def streak(ticker: str, min_days: int = 3, horizon: int = 90, years: int = 10):
    ticker = ticker.upper().strip()
    df = await get_history(ticker, years, prefer_daily=False)
    df = df.copy(); df["ret"] = df["close"].pct_change()
    events = []; streak_len = 0
    for i in range(1, len(df)):
        ret = df.iloc[i]["ret"]
        if pd.isna(ret): streak_len = 0; continue
        if ret < 0: streak_len += 1
        else:
            if streak_len >= min_days:
                fwd = forward_return(df, i-1, horizon)
                events.append({"date":df.index[i-1].strftime("%Y-%m-%d"),"price":safe(df.iloc[i-1]["close"]),
                               "streak_length":streak_len,"forward_return_pct":safe(fwd) if fwd is not None else None})
            streak_len = 0
    valid = [e["forward_return_pct"] for e in events if e["forward_return_pct"] is not None]
    stats = summary_stats(valid)
    stats["avg_streak_length"] = safe(np.mean([e["streak_length"] for e in events])) if events else 0.0
    return {"ticker":ticker,"analysis_type":"streak","min_days":min_days,
            "horizon_days":horizon,"years":years,"events":events,"summary":stats}

# ── VOLATILITY (daily compact) ────────────────────────────────────
@app.get("/volatility")
async def volatility(ticker: str, window: int = 10, percentile: int = 85, horizon: int = 30, years: int = 1):
    ticker = ticker.upper().strip()
    df = await get_history(ticker, years, prefer_daily=True)
    df = df.copy(); df["ret"] = df["close"].pct_change()
    df["vol"] = df["ret"].rolling(window).std() * np.sqrt(252) * 100
    df = df.dropna(subset=["vol"])
    if df.empty: raise HTTPException(status_code=400, detail="Not enough data for volatility.")
    thr = float(np.percentile(df["vol"].values, percentile))
    events = []
    for i in range(len(df)):
        v = df.iloc[i]["vol"]
        if v >= thr:
            fwd = forward_return(df, i, horizon)
            events.append({"date":df.index[i].strftime("%Y-%m-%d"),"price":safe(df.iloc[i]["close"]),
                           "volatility":safe(v),"forward_return_pct":safe(fwd) if fwd is not None else None})
    deduped, last_d = [], None
    for e in events:
        d = datetime.strptime(e["date"],"%Y-%m-%d")
        if last_d is None or (d-last_d).days >= 5:
            deduped.append(e); last_d = d
    valid = [e["forward_return_pct"] for e in deduped if e["forward_return_pct"] is not None]
    stats = summary_stats(valid); stats["vol_threshold"] = safe(thr)
    return {"ticker":ticker,"analysis_type":"volatility","window":window,"percentile":percentile,
            "horizon_days":horizon,"years":years,"events":deduped,"summary":stats}

# ── SEASONALITY (weekly → monthly) ───────────────────────────────
@app.get("/seasonality")
async def seasonality(ticker: str, years: int = 10):
    ticker = ticker.upper().strip()
    df = await get_history(ticker, years, prefer_daily=False)
    monthly = df["close"].resample("MS").last().pct_change() * 100
    monthly = monthly.dropna()
    names = ["ENE","FEB","MAR","ABR","MAY","JUN","JUL","AGO","SEP","OCT","NOV","DIC"]
    result = []
    for m in range(1,13):
        vals = monthly[monthly.index.month==m].tolist()
        result.append({"month":m,"month_name":names[m-1],
                       "avg_return":safe(np.mean(vals)) if vals else 0.0,
                       "pct_positive":safe(np.mean([v>0 for v in vals])*100) if vals else 0.0,
                       "sample_count":len(vals)})
    best = max(result,key=lambda x:x["avg_return"])
    worst = min(result,key=lambda x:x["avg_return"])
    return {"ticker":ticker,"analysis_type":"seasonality","years":years,
            "monthly_data":result,"best_month":best["month_name"],"worst_month":worst["month_name"]}

# ── CANDLES (weekly OHLC) ─────────────────────────────────────────
@app.get("/candles")
async def candles(ticker: str, years: int = 10, events: str = ""):
    ticker = ticker.upper().strip()
    df = await get_history(ticker, years, prefer_daily=False)
    monthly = df.resample("MS").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    candle_list = [{"time":row.Index.strftime("%Y-%m-%d"),"open":safe(row.open),"high":safe(row.high),
                    "low":safe(row.low),"close":safe(row.close)} for row in monthly.itertuples()]
    event_list = []
    if events:
        for ds in events.split(","):
            ds = ds.strip()
            if not ds: continue
            try:
                d = datetime.strptime(ds,"%Y-%m-%d")
                ms = d.replace(day=1).strftime("%Y-%m-%d")
                near = df[df.index >= d]
                price = safe(near.iloc[0]["close"]) if not near.empty else 0.0
                event_list.append({"time":ms,"label":ds,"price":price})
            except Exception: pass
    return {"ticker":ticker,"candles":candle_list,"events":event_list}

# ── COMPARE ───────────────────────────────────────────────────────
@app.get("/compare")
async def compare(tickers: str, event: str = "drawdown30", horizon: int = 365, years: int = 10):
    results = []
    for t in tickers.split(","):
        t = t.strip().upper()
        if not t: continue
        try:
            if event.startswith("drawdown"):
                r = await drawdown(t,threshold=float(event.replace("drawdown","")),horizon=horizon,years=years)
            elif event.startswith("dayfall"):
                r = await dayfall(t,threshold=float(event.replace("dayfall","")),horizon=min(horizon,60),years=1)
            elif event.startswith("streak"):
                r = await streak(t,min_days=int(event.replace("streak","")),horizon=horizon,years=years)
            else: continue
            s = r["summary"]
            results.append({"ticker":t,"event_count":s["event_count"],
                            "avg_forward_return":s["avg_forward_return"],
                            "median_forward_return":s["median_forward_return"],
                            "pct_positive":s["pct_positive"],
                            "best_return":s["best_return"],"worst_return":s["worst_return"]})
        except Exception as e:
            results.append({"ticker":t,"error":str(e),"event_count":0,
                           "avg_forward_return":0.0,"median_forward_return":0.0,
                           "pct_positive":0.0,"best_return":0.0,"worst_return":0.0})
    return {"analysis_type":"compare","event":event,"horizon_days":horizon,"results":results}

@app.get("/health")
async def health():
    return {"status":"ok","av_key":"configured" if AV_KEY else "MISSING - add AV_KEY in Render env vars"}
