from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import math, os, pickle, time, httpx, random

app = FastAPI(title="Market Pattern Analyzer API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

AV_KEY = os.environ.get("AV_KEY", "")  # Alpha Vantage API key from env var

# ── DISK CACHE ────────────────────────────────────────────────────
_MEM_CACHE = {}
_CACHE_TTL  = 1800    # 30 min memory
_DISK_TTL   = 86400   # 24h disk
_DISK_DIR   = "/tmp/av_cache"
os.makedirs(_DISK_DIR, exist_ok=True)

def _cache_key(ticker, years):
    return f"{ticker.upper()}_{years}"

def _mem_get(key):
    if key in _MEM_CACHE and time.time() - _MEM_CACHE[key][0] < _CACHE_TTL:
        return _MEM_CACHE[key][1].copy()
    return None

def _disk_get(key):
    path = f"{_DISK_DIR}/{key}.pkl"
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < _DISK_TTL:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return None

def _cache_set(key, df):
    _MEM_CACHE[key] = (time.time(), df)
    try:
        with open(f"{_DISK_DIR}/{key}.pkl", "wb") as f:
            pickle.dump(df, f)
    except Exception:
        pass

# ── DATA FETCH ────────────────────────────────────────────────────
async def fetch_av(ticker: str) -> pd.DataFrame:
    """Fetch full daily history from Alpha Vantage."""
    if not AV_KEY:
        raise HTTPException(status_code=500, detail="Alpha Vantage API key not configured. Add AV_KEY env var in Render.")
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "datatype": "json",
        "apikey": AV_KEY,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params)
    data = r.json()

    if "Error Message" in data:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found on Alpha Vantage.")
    if "Note" in data or "Information" in data:
        msg = data.get("Note") or data.get("Information") or ""
        raise HTTPException(status_code=429, detail=f"Alpha Vantage rate limit: {msg}")
    if "Time Series (Daily)" not in data:
        raise HTTPException(status_code=500, detail=f"Unexpected response for {ticker}: {list(data.keys())}")

    ts = data["Time Series (Daily)"]
    rows = []
    for date_str, vals in ts.items():
        rows.append({
            "date": pd.to_datetime(date_str),
            "open":  float(vals.get("1. open", 0)),
            "high":  float(vals.get("2. high", 0)),
            "low":   float(vals.get("3. low", 0)),
            "close": float(vals.get("5. adjusted close", vals.get("4. close", 0))),
        })
    df = pd.DataFrame(rows).set_index("date").sort_index()
    df = df.dropna()
    if len(df) < 20:
        raise HTTPException(status_code=404, detail=f"Not enough data for {ticker} ({len(df)} rows).")
    return df

async def get_history(ticker: str, years: int) -> pd.DataFrame:
    key = _cache_key(ticker, years)

    # Memory cache
    df = _mem_get(key)
    if df is not None:
        return df

    # Disk cache
    df = _disk_get(key)
    if df is not None:
        _MEM_CACHE[key] = (time.time(), df)
        return df.copy()

    # Fetch full history then slice
    full_key = f"{ticker.upper()}_full"
    df_full = _mem_get(full_key) or _disk_get(full_key)
    if df_full is None:
        df_full = await fetch_av(ticker)
        _cache_set(full_key, df_full)

    cutoff = datetime.today() - timedelta(days=years * 365 + 60)
    df = df_full[df_full.index >= cutoff].copy()
    if len(df) < 20:
        raise HTTPException(status_code=404, detail=f"Not enough data for {ticker} in last {years} years.")
    _cache_set(key, df)
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
    entry_price = df.iloc[idx]["close"]
    if entry_price == 0: return None
    return round((future.iloc[0]["close"] / entry_price - 1) * 100, 2)

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
        params = {"q": q, "quotesCount": 10, "newsCount": 0}
        headers = {"User-Agent": "Mozilla/5.0"}
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(url, params=params, headers=headers)
            data = r.json()
        results = []
        for item in data.get("quotes", []):
            if item.get("quoteType") in {"EQUITY","ETF","INDEX","MUTUALFUND"}:
                results.append({"symbol":item.get("symbol",""),"name":item.get("longname") or item.get("shortname",""),"exchange":item.get("exchDisp",item.get("exchange","")),"type":item.get("quoteType","")})
        return {"results": results}
    except Exception as e:
        return {"results": [], "error": str(e)}

# ── DRAWDOWN ──────────────────────────────────────────────────────
@app.get("/drawdown")
async def drawdown(ticker: str, threshold: float = 30, horizon: int = 365, years: int = 10):
    ticker = ticker.upper().strip()
    df = await get_history(ticker, years)
    closes = df["close"]
    events = []
    peak = closes.iloc[0]
    in_episode = False

    for i in range(1, len(closes)):
        price = closes.iloc[i]
        if not in_episode:
            if price > peak:
                peak = price
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

# ── DAY FALL ──────────────────────────────────────────────────────
@app.get("/dayfall")
async def dayfall(ticker: str, threshold: float = 10, horizon: int = 365, years: int = 10):
    ticker = ticker.upper().strip()
    df = await get_history(ticker, years)
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

# ── STREAK ────────────────────────────────────────────────────────
@app.get("/streak")
async def streak(ticker: str, min_days: int = 5, horizon: int = 90, years: int = 10):
    ticker = ticker.upper().strip()
    df = await get_history(ticker, years)
    df = df.copy(); df["day_ret"] = df["close"].pct_change()
    events = []; streak_len = 0
    for i in range(1, len(df)):
        ret = df.iloc[i]["day_ret"]
        if pd.isna(ret): streak_len = 0; continue
        if ret < 0: streak_len += 1
        else:
            if streak_len >= min_days:
                end_idx = i - 1
                fwd = forward_return(df, end_idx, horizon)
                events.append({"date":df.index[end_idx].strftime("%Y-%m-%d"),"price":safe(df.iloc[end_idx]["close"]),
                               "streak_length":streak_len,"forward_return_pct":safe(fwd) if fwd is not None else None})
            streak_len = 0
    valid = [e["forward_return_pct"] for e in events if e["forward_return_pct"] is not None]
    stats = summary_stats(valid)
    stats["avg_streak_length"] = safe(np.mean([e["streak_length"] for e in events])) if events else 0.0
    return {"ticker":ticker,"analysis_type":"streak","min_days":min_days,
            "horizon_days":horizon,"years":years,"events":events,"summary":stats}

# ── VOLATILITY ────────────────────────────────────────────────────
@app.get("/volatility")
async def volatility(ticker: str, window: int = 20, percentile: int = 85, horizon: int = 90, years: int = 10):
    ticker = ticker.upper().strip()
    df = await get_history(ticker, years)
    df = df.copy(); df["day_ret"] = df["close"].pct_change()
    df["vol"] = df["day_ret"].rolling(window).std() * np.sqrt(252) * 100
    df = df.dropna(subset=["vol"])
    if df.empty: raise HTTPException(status_code=400, detail="Not enough data for volatility.")
    threshold_v = float(np.percentile(df["vol"].values, percentile))
    events = []
    for i in range(len(df)):
        v = df.iloc[i]["vol"]
        if v >= threshold_v:
            fwd = forward_return(df, i, horizon)
            events.append({"date":df.index[i].strftime("%Y-%m-%d"),"price":safe(df.iloc[i]["close"]),
                           "volatility":safe(v),"forward_return_pct":safe(fwd) if fwd is not None else None})
    deduped, last_date = [], None
    for e in events:
        d = datetime.strptime(e["date"], "%Y-%m-%d")
        if last_date is None or (d - last_date).days >= 20:
            deduped.append(e); last_date = d
    valid = [e["forward_return_pct"] for e in deduped if e["forward_return_pct"] is not None]
    stats = summary_stats(valid); stats["vol_threshold"] = safe(threshold_v)
    return {"ticker":ticker,"analysis_type":"volatility","window":window,"percentile":percentile,
            "horizon_days":horizon,"years":years,"events":deduped,"summary":stats}

# ── SEASONALITY ───────────────────────────────────────────────────
@app.get("/seasonality")
async def seasonality(ticker: str, years: int = 10):
    ticker = ticker.upper().strip()
    df = await get_history(ticker, years)
    monthly = df["close"].resample("MS").last().pct_change() * 100
    monthly = monthly.dropna()
    names = ["ENE","FEB","MAR","ABR","MAY","JUN","JUL","AGO","SEP","OCT","NOV","DIC"]
    result = []
    for m in range(1, 13):
        vals = monthly[monthly.index.month == m].tolist()
        result.append({"month":m,"month_name":names[m-1],"avg_return":safe(np.mean(vals)) if vals else 0.0,
                       "pct_positive":safe(np.mean([v>0 for v in vals])*100) if vals else 0.0,"sample_count":len(vals)})
    best = max(result, key=lambda x: x["avg_return"])
    worst = min(result, key=lambda x: x["avg_return"])
    return {"ticker":ticker,"analysis_type":"seasonality","years":years,
            "monthly_data":result,"best_month":best["month_name"],"worst_month":worst["month_name"]}

# ── CANDLES ───────────────────────────────────────────────────────
@app.get("/candles")
async def candles(ticker: str, years: int = 10, events: str = ""):
    ticker = ticker.upper().strip()
    df = await get_history(ticker, years)
    monthly = df.resample("MS").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    candle_list = [{"time":row.Index.strftime("%Y-%m-%d"),"open":safe(row.open),"high":safe(row.high),
                    "low":safe(row.low),"close":safe(row.close)} for row in monthly.itertuples()]
    event_list = []
    if events:
        for ds in events.split(","):
            ds = ds.strip()
            if not ds: continue
            try:
                d = datetime.strptime(ds, "%Y-%m-%d")
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
                r = await drawdown(t, threshold=float(event.replace("drawdown","")), horizon=horizon, years=years)
            elif event.startswith("dayfall"):
                r = await dayfall(t, threshold=float(event.replace("dayfall","")), horizon=horizon, years=years)
            elif event.startswith("streak"):
                r = await streak(t, min_days=int(event.replace("streak","")), horizon=horizon, years=years)
            else: continue
            s = r["summary"]
            results.append({"ticker":t,"event_count":s["event_count"],"avg_forward_return":s["avg_forward_return"],
                            "median_forward_return":s["median_forward_return"],"pct_positive":s["pct_positive"],
                            "best_return":s["best_return"],"worst_return":s["worst_return"]})
        except Exception as e:
            results.append({"ticker":t,"error":str(e),"event_count":0,"avg_forward_return":0.0,
                           "median_forward_return":0.0,"pct_positive":0.0,"best_return":0.0,"worst_return":0.0})
    return {"analysis_type":"compare","event":event,"horizon_days":horizon,"results":results}

@app.get("/health")
async def health():
    key_status = "configured" if AV_KEY else "MISSING - add AV_KEY env var in Render"
    return {"status": "ok", "av_key": key_status}
