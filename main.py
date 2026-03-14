from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import math
import httpx

app = FastAPI(title="Market Pattern Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── DATA FETCH ────────────────────────────────────────────────────
def get_history(ticker: str, years: int) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=years * 365 + 60)
    
    last_err = None
    for attempt in range(3):
        try:
            import time
            if attempt > 0:
                time.sleep(2 * attempt)
            
            # Use session with browser-like headers to avoid rate limiting
            import requests
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            })
            t = yf.Ticker(ticker, session=session)
            df = t.history(start=start, end=end, auto_adjust=True)
            
            if df is not None and not df.empty:
                break
            last_err = f"No data returned for '{ticker}'"
        except Exception as e:
            last_err = str(e)
            if "rate limit" in last_err.lower() or "too many" in last_err.lower():
                continue
            break
    else:
        raise HTTPException(status_code=500, detail=f"Yahoo Finance rate limit. Wait a moment and retry. ({last_err})")
    
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for '{ticker}'. Check the ticker symbol (e.g. GGAL.BA for Buenos Aires).")
    
    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns:
        raise HTTPException(status_code=500, detail=f"Unexpected data format for {ticker}")
    for col in ["open", "high", "low"]:
        if col not in df.columns:
            df[col] = df["close"]
    df = df[["open", "high", "low", "close"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna()
    if len(df) < 20:
        raise HTTPException(status_code=404, detail=f"Not enough data for '{ticker}' ({len(df)} rows).")
    return df

import time as _time
_cache = {}
_CACHE_TTL = 300  # 5 minutes

def _cache_key(ticker, years):
    return f"{ticker}_{years}"

def get_history_cached(ticker: str, years: int) -> pd.DataFrame:
    key = _cache_key(ticker, years)
    now = _time.time()
    if key in _cache and now - _cache[key][0] < _CACHE_TTL:
        return _cache[key][1].copy()
    df = get_history(ticker, years)
    _cache[key] = (now, df)
    return df.copy()


def safe(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return 0.0
    return round(float(v), 4)

def forward_return(df: pd.DataFrame, idx: int, horizon_days: int) -> Optional[float]:
    entry_date = df.index[idx]
    target_date = entry_date + timedelta(days=horizon_days)
    future = df[df.index >= target_date]
    if future.empty:
        return None
    entry_price = df.iloc[idx]["close"]
    if entry_price == 0:
        return None
    return round((future.iloc[0]["close"] / entry_price - 1) * 100, 2)

def summary_stats(returns: list) -> dict:
    if not returns:
        return {"event_count": 0, "avg_forward_return": 0.0, "median_forward_return": 0.0,
                "pct_positive": 0.0, "best_return": 0.0, "worst_return": 0.0, "std_dev": 0.0}
    arr = np.array(returns)
    return {
        "event_count": len(returns),
        "avg_forward_return": safe(np.mean(arr)),
        "median_forward_return": safe(np.median(arr)),
        "pct_positive": safe(np.mean(arr > 0) * 100),
        "best_return": safe(np.max(arr)),
        "worst_return": safe(np.min(arr)),
        "std_dev": safe(np.std(arr)),
    }


# ── TICKER SEARCH ─────────────────────────────────────────────────
@app.get("/search")
async def search_ticker(q: str):
    if len(q) < 1:
        return {"results": []}
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": q, "quotesCount": 10, "newsCount": 0, "enableFuzzyQuery": "false"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(url, params=params, headers=headers)
            data = r.json()
        quotes = data.get("quotes", [])
        results = []
        valid_types = {"EQUITY", "ETF", "INDEX", "MUTUALFUND"}
        for item in quotes:
            symbol = item.get("symbol", "")
            name = item.get("longname") or item.get("shortname") or ""
            exchange = item.get("exchange", "")
            exchDisp = item.get("exchDisp", exchange)
            qtype = item.get("quoteType", "")
            if symbol and qtype in valid_types:
                results.append({"symbol": symbol, "name": name, "exchange": exchDisp, "type": qtype})
        return {"results": results}
    except Exception as e:
        return {"results": [], "error": str(e)}


# ── DRAWDOWN ──────────────────────────────────────────────────────
@app.get("/drawdown")
def drawdown(ticker: str, threshold: float = 30, horizon: int = 365, years: int = 10):
    ticker = ticker.upper().strip()
    df = get_history_cached(ticker, years)
    closes = df["close"]
    events = []
    peak = closes.iloc[0]
    in_drawdown = False

    for i in range(1, len(closes)):
        price = closes.iloc[i]
        if price > peak:
            peak = price
            in_drawdown = False
        dd_pct = (price / peak - 1) * 100
        if dd_pct <= -threshold and not in_drawdown:
            in_drawdown = True
        if in_drawdown:
            is_bottom = (i == len(closes) - 1) or (closes.iloc[i + 1] > price)
            if is_bottom:
                fwd = forward_return(df, i, horizon)
                events.append({
                    "date": df.index[i].strftime("%Y-%m-%d"),
                    "price": safe(price),
                    "drawdown_pct": safe(dd_pct),
                    "forward_return_pct": safe(fwd) if fwd is not None else None,
                })
                in_drawdown = False

    valid_returns = [e["forward_return_pct"] for e in events if e["forward_return_pct"] is not None]
    return {"ticker": ticker, "analysis_type": "drawdown", "threshold": threshold,
            "horizon_days": horizon, "years": years, "events": events,
            "summary": summary_stats(valid_returns)}


# ── DAY FALL ──────────────────────────────────────────────────────
@app.get("/dayfall")
def dayfall(ticker: str, threshold: float = 10, horizon: int = 365, years: int = 10):
    ticker = ticker.upper().strip()
    df = get_history_cached(ticker, years)
    df = df.copy()
    df["day_ret"] = df["close"].pct_change() * 100
    events = []
    for i in range(1, len(df)):
        day_ret = df.iloc[i]["day_ret"]
        if pd.isna(day_ret) or day_ret > -threshold:
            continue
        fwd = forward_return(df, i, horizon)
        events.append({
            "date": df.index[i].strftime("%Y-%m-%d"),
            "price": safe(df.iloc[i]["close"]),
            "day_return_pct": safe(day_ret),
            "forward_return_pct": safe(fwd) if fwd is not None else None,
        })
    valid_returns = [e["forward_return_pct"] for e in events if e["forward_return_pct"] is not None]
    stats = summary_stats(valid_returns)
    stats["avg_day_drop"] = safe(np.mean([e["day_return_pct"] for e in events])) if events else 0.0
    return {"ticker": ticker, "analysis_type": "dayfall", "threshold": threshold,
            "horizon_days": horizon, "years": years, "events": events, "summary": stats}


# ── STREAK ────────────────────────────────────────────────────────
@app.get("/streak")
def streak(ticker: str, min_days: int = 5, horizon: int = 90, years: int = 10):
    ticker = ticker.upper().strip()
    df = get_history_cached(ticker, years)
    df = df.copy()
    df["day_ret"] = df["close"].pct_change()
    events = []
    streak_len = 0
    for i in range(1, len(df)):
        ret = df.iloc[i]["day_ret"]
        if pd.isna(ret):
            streak_len = 0
            continue
        if ret < 0:
            streak_len += 1
        else:
            if streak_len >= min_days:
                end_idx = i - 1
                fwd = forward_return(df, end_idx, horizon)
                events.append({
                    "date": df.index[end_idx].strftime("%Y-%m-%d"),
                    "price": safe(df.iloc[end_idx]["close"]),
                    "streak_length": streak_len,
                    "forward_return_pct": safe(fwd) if fwd is not None else None,
                })
            streak_len = 0
    valid_returns = [e["forward_return_pct"] for e in events if e["forward_return_pct"] is not None]
    stats = summary_stats(valid_returns)
    stats["avg_streak_length"] = safe(np.mean([e["streak_length"] for e in events])) if events else 0.0
    return {"ticker": ticker, "analysis_type": "streak", "min_days": min_days,
            "horizon_days": horizon, "years": years, "events": events, "summary": stats}


# ── VOLATILITY ────────────────────────────────────────────────────
@app.get("/volatility")
def volatility(ticker: str, window: int = 20, percentile: int = 85, horizon: int = 90, years: int = 10):
    ticker = ticker.upper().strip()
    df = get_history_cached(ticker, years)
    df = df.copy()
    df["day_ret"] = df["close"].pct_change()
    df["vol"] = df["day_ret"].rolling(window).std() * np.sqrt(252) * 100
    df = df.dropna(subset=["vol"])
    if df.empty:
        raise HTTPException(status_code=400, detail="Not enough data for volatility calculation")
    threshold_v = float(np.percentile(df["vol"].values, percentile))
    events = []
    for i in range(len(df)):
        v = df.iloc[i]["vol"]
        if v >= threshold_v:
            fwd = forward_return(df, i, horizon)
            events.append({
                "date": df.index[i].strftime("%Y-%m-%d"),
                "price": safe(df.iloc[i]["close"]),
                "volatility": safe(v),
                "forward_return_pct": safe(fwd) if fwd is not None else None,
            })
    # Deduplicate: one event per 20-day window
    deduped, last_date = [], None
    for e in events:
        d = datetime.strptime(e["date"], "%Y-%m-%d")
        if last_date is None or (d - last_date).days >= 20:
            deduped.append(e)
            last_date = d
    valid_returns = [e["forward_return_pct"] for e in deduped if e["forward_return_pct"] is not None]
    stats = summary_stats(valid_returns)
    stats["vol_threshold"] = safe(threshold_v)
    return {"ticker": ticker, "analysis_type": "volatility", "window": window,
            "percentile": percentile, "horizon_days": horizon, "years": years,
            "events": deduped, "summary": stats}


# ── SEASONALITY ───────────────────────────────────────────────────
@app.get("/seasonality")
def seasonality(ticker: str, years: int = 10):
    ticker = ticker.upper().strip()
    df = get_history_cached(ticker, years)
    monthly = df["close"].resample("MS").last().pct_change() * 100
    monthly = monthly.dropna()
    month_names = ["ENE","FEB","MAR","ABR","MAY","JUN","JUL","AGO","SEP","OCT","NOV","DIC"]
    result = []
    for m in range(1, 13):
        vals = monthly[monthly.index.month == m].tolist()
        avg = safe(np.mean(vals)) if vals else 0.0
        pct_pos = safe(np.mean([v > 0 for v in vals]) * 100) if vals else 0.0
        result.append({"month": m, "month_name": month_names[m-1],
                        "avg_return": avg, "pct_positive": pct_pos, "sample_count": len(vals)})
    best = max(result, key=lambda x: x["avg_return"])
    worst = min(result, key=lambda x: x["avg_return"])
    return {"ticker": ticker, "analysis_type": "seasonality", "years": years,
            "monthly_data": result, "best_month": best["month_name"], "worst_month": worst["month_name"]}


# ── CANDLES ───────────────────────────────────────────────────────
@app.get("/candles")
def candles(ticker: str, years: int = 10, events: str = ""):
    ticker = ticker.upper().strip()
    df = get_history_cached(ticker, years)
    monthly = df.resample("MS").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    candle_list = [
        {"time": row.Index.strftime("%Y-%m-%d"),
         "open": safe(row.open), "high": safe(row.high),
         "low": safe(row.low), "close": safe(row.close)}
        for row in monthly.itertuples()
    ]
    event_list = []
    if events:
        for date_str in events.split(","):
            date_str = date_str.strip()
            if not date_str:
                continue
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d")
                month_start = d.replace(day=1).strftime("%Y-%m-%d")
                near = df[df.index >= d]
                price = safe(near.iloc[0]["close"]) if not near.empty else 0.0
                event_list.append({"time": month_start, "label": date_str, "price": price})
            except Exception:
                pass
    return {"ticker": ticker, "candles": candle_list, "events": event_list}


# ── COMPARE ───────────────────────────────────────────────────────
@app.get("/compare")
def compare(tickers: str, event: str = "drawdown30", horizon: int = 365, years: int = 10):
    results = []
    for t in tickers.split(","):
        t = t.strip().upper()
        if not t:
            continue
        try:
            if event.startswith("drawdown"):
                r = drawdown(t, threshold=float(event.replace("drawdown","")), horizon=horizon, years=years)
            elif event.startswith("dayfall"):
                r = dayfall(t, threshold=float(event.replace("dayfall","")), horizon=horizon, years=years)
            elif event.startswith("streak"):
                r = streak(t, min_days=int(event.replace("streak","")), horizon=horizon, years=years)
            else:
                continue
            s = r["summary"]
            results.append({"ticker": t, "event_count": s["event_count"],
                             "avg_forward_return": s["avg_forward_return"],
                             "median_forward_return": s["median_forward_return"],
                             "pct_positive": s["pct_positive"],
                             "best_return": s["best_return"],
                             "worst_return": s["worst_return"]})
        except Exception as e:
            results.append({"ticker": t, "error": str(e), "event_count": 0,
                             "avg_forward_return": 0.0, "median_forward_return": 0.0,
                             "pct_positive": 0.0, "best_return": 0.0, "worst_return": 0.0})
    return {"analysis_type": "compare", "event": event, "horizon_days": horizon, "results": results}


@app.get("/health")
def health():
    return {"status": "ok"}
