from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import math

app = FastAPI(title="Market Pattern Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

def get_history(ticker: str, years: int) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=years * 365 + 30)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    df = df[["Open", "High", "Low", "Close"]].copy()
    df.columns = ["open", "high", "low", "close"]
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    return df

def safe(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0.0
    return round(float(v), 4)

def forward_return(df: pd.DataFrame, idx: int, horizon_days: int) -> Optional[float]:
    """Return % change from df.iloc[idx] to horizon_days calendar days later."""
    entry_date = df.index[idx]
    target_date = entry_date + timedelta(days=horizon_days)
    future = df[df.index >= target_date]
    if future.empty:
        return None
    entry_price = df.iloc[idx]["close"]
    exit_price = future.iloc[0]["close"]
    return round((exit_price / entry_price - 1) * 100, 2)

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


# ── DRAWDOWN ──────────────────────────────────────────────────────
@app.get("/drawdown")
def drawdown(ticker: str, threshold: float = 30, horizon: int = 365, years: int = 10):
    df = get_history(ticker, years)
    closes = df["close"]
    events = []
    peak = closes.iloc[0]
    peak_idx = 0
    in_drawdown = False
    dd_start_idx = 0

    for i in range(1, len(closes)):
        price = closes.iloc[i]
        if price > peak:
            peak = price
            peak_idx = i
            in_drawdown = False
        dd_pct = (price / peak - 1) * 100
        if dd_pct <= -threshold and not in_drawdown:
            in_drawdown = True
            dd_start_idx = i
        if in_drawdown and (i == len(closes) - 1 or closes.iloc[i+1] > price):
            fwd = forward_return(df, i, horizon)
            events.append({
                "date": df.index[i].strftime("%Y-%m-%d"),
                "price": safe(price),
                "drawdown_pct": safe(dd_pct),
                "forward_return_pct": safe(fwd) if fwd is not None else None,
            })
            in_drawdown = False

    valid_returns = [e["forward_return_pct"] for e in events if e["forward_return_pct"] is not None]
    stats = summary_stats(valid_returns)
    return {"ticker": ticker.upper(), "analysis_type": "drawdown", "threshold": threshold,
            "horizon_days": horizon, "events": events, "summary": stats}


# ── DAY FALL ──────────────────────────────────────────────────────
@app.get("/dayfall")
def dayfall(ticker: str, threshold: float = 10, horizon: int = 365, years: int = 10):
    df = get_history(ticker, years)
    df["day_ret"] = df["close"].pct_change() * 100
    events = []
    for i in range(1, len(df)):
        day_ret = df.iloc[i]["day_ret"]
        if day_ret <= -threshold:
            fwd = forward_return(df, i, horizon)
            events.append({
                "date": df.index[i].strftime("%Y-%m-%d"),
                "price": safe(df.iloc[i]["close"]),
                "day_return_pct": safe(day_ret),
                "forward_return_pct": safe(fwd) if fwd is not None else None,
            })
    valid_returns = [e["forward_return_pct"] for e in events if e["forward_return_pct"] is not None]
    stats = summary_stats(valid_returns)
    avg_drop = safe(np.mean([e["day_return_pct"] for e in events])) if events else 0.0
    stats["avg_day_drop"] = avg_drop
    return {"ticker": ticker.upper(), "analysis_type": "dayfall", "threshold": threshold,
            "horizon_days": horizon, "events": events, "summary": stats}


# ── STREAK ────────────────────────────────────────────────────────
@app.get("/streak")
def streak(ticker: str, min_days: int = 5, horizon: int = 90, years: int = 10):
    df = get_history(ticker, years)
    df["day_ret"] = df["close"].pct_change()
    events = []
    streak_len = 0
    for i in range(1, len(df)):
        if df.iloc[i]["day_ret"] < 0:
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
    return {"ticker": ticker.upper(), "analysis_type": "streak", "min_days": min_days,
            "horizon_days": horizon, "events": events, "summary": stats}


# ── VOLATILITY ────────────────────────────────────────────────────
@app.get("/volatility")
def volatility(ticker: str, window: int = 20, percentile: int = 85, horizon: int = 90, years: int = 10):
    df = get_history(ticker, years)
    df["day_ret"] = df["close"].pct_change()
    df["vol"] = df["day_ret"].rolling(window).std() * np.sqrt(252) * 100
    df = df.dropna()
    threshold = np.percentile(df["vol"], percentile)
    events = []
    for i in range(len(df)):
        if df.iloc[i]["vol"] >= threshold:
            fwd = forward_return(df, i, horizon)
            events.append({
                "date": df.index[i].strftime("%Y-%m-%d"),
                "price": safe(df.iloc[i]["close"]),
                "volatility": safe(df.iloc[i]["vol"]),
                "forward_return_pct": safe(fwd) if fwd is not None else None,
            })
    # Deduplicate: keep only one event per 30-day window
    deduped = []
    last_date = None
    for e in events:
        d = datetime.strptime(e["date"], "%Y-%m-%d")
        if last_date is None or (d - last_date).days >= 30:
            deduped.append(e)
            last_date = d
    valid_returns = [e["forward_return_pct"] for e in deduped if e["forward_return_pct"] is not None]
    stats = summary_stats(valid_returns)
    stats["vol_threshold"] = safe(threshold)
    return {"ticker": ticker.upper(), "analysis_type": "volatility", "window": window,
            "percentile": percentile, "horizon_days": horizon, "events": deduped, "summary": stats}


# ── SEASONALITY ───────────────────────────────────────────────────
@app.get("/seasonality")
def seasonality(ticker: str, years: int = 10):
    df = get_history(ticker, years)
    df["month_ret"] = df["close"].pct_change() * 100
    df["month"] = df.index.month
    month_names = ["ENE","FEB","MAR","ABR","MAY","JUN","JUL","AGO","SEP","OCT","NOV","DIC"]
    monthly = []
    for m in range(1, 13):
        vals = df[df["month"] == m]["month_ret"].dropna().tolist()
        avg = safe(np.mean(vals)) if vals else 0.0
        pct_pos = safe(np.mean([v > 0 for v in vals]) * 100) if vals else 0.0
        monthly.append({"month": m, "month_name": month_names[m-1], "avg_return": avg,
                         "pct_positive": pct_pos, "sample_count": len(vals)})
    best = max(monthly, key=lambda x: x["avg_return"])
    worst = min(monthly, key=lambda x: x["avg_return"])
    return {"ticker": ticker.upper(), "analysis_type": "seasonality", "years": years,
            "monthly_data": monthly, "best_month": best["month_name"], "worst_month": worst["month_name"]}


# ── CANDLES (monthly OHLC) ────────────────────────────────────────
@app.get("/candles")
def candles(ticker: str, years: int = 10, events: str = ""):
    df = get_history(ticker, years)
    monthly = df.resample("MS").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
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
    return {"ticker": ticker.upper(), "candles": candle_list, "events": event_list}


# ── COMPARE ───────────────────────────────────────────────────────
@app.get("/compare")
def compare(tickers: str, event: str = "drawdown30", horizon: int = 365, years: int = 10):
    results = []
    for t in tickers.split(","):
        t = t.strip().upper()
        try:
            if event.startswith("drawdown"):
                thresh = float(event.replace("drawdown", ""))
                r = drawdown(t, threshold=thresh, horizon=horizon, years=years)
            elif event.startswith("dayfall"):
                thresh = float(event.replace("dayfall", ""))
                r = dayfall(t, threshold=thresh, horizon=horizon, years=years)
            elif event.startswith("streak"):
                days = int(event.replace("streak", ""))
                r = streak(t, min_days=days, horizon=horizon, years=years)
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
