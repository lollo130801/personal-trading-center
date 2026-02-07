from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import ccxt
import numpy as np
import pandas as pd
import yfinance as yf

Interval = Literal[
    "1s",
    "5s",
    "15s",
    "1m",
    "5m",
    "15m",
    "1h",
    "1d",
    "1wk",
    "1mo",
]


@dataclass
class MarketRequest:
    symbol: str
    interval: Interval
    limit: int
    source: Literal["yfinance", "binance"]


YFINANCE_INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}

CCXT_INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "1d": "1d",
    "1wk": "1w",
    "1mo": "1M",
}

SECONDS_INTERVALS = {
    "1s": 1,
    "5s": 5,
    "15s": 15,
}


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "timestamp"})
    elif "Date" in df.columns:
        df = df.rename(columns={"Date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_ohlcv(request: MarketRequest) -> pd.DataFrame:
    if request.interval in SECONDS_INTERVALS:
        if request.source != "binance":
            raise ValueError("Intervalli in secondi supportati solo con Binance.")
        return _fetch_ohlcv_from_trades(request)

    if request.source == "yfinance":
        return _fetch_yfinance(request)

    return _fetch_ccxt_ohlcv(request)


def _fetch_yfinance(request: MarketRequest) -> pd.DataFrame:
    interval = YFINANCE_INTERVALS.get(request.interval)
    if not interval:
        raise ValueError("Intervallo non supportato da Yahoo Finance.")
    df = yf.download(
        request.symbol,
        interval=interval,
        period=_estimate_period(request.interval, request.limit),
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        return df
    df = df.reset_index()
    return _normalize_df(df)


def _fetch_ccxt_ohlcv(request: MarketRequest) -> pd.DataFrame:
    exchange = ccxt.binance()
    interval = CCXT_INTERVALS[request.interval]
    since = int(
        (datetime.now(timezone.utc) - _interval_delta(request.interval, request.limit))
        .timestamp()
        * 1000
    )
    candles = exchange.fetch_ohlcv(request.symbol, timeframe=interval, since=since, limit=request.limit)
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def _fetch_ohlcv_from_trades(request: MarketRequest) -> pd.DataFrame:
    exchange = ccxt.binance()
    seconds = SECONDS_INTERVALS[request.interval]
    lookback = seconds * request.limit
    since_ms = int((datetime.now(timezone.utc) - timedelta(seconds=lookback)).timestamp() * 1000)
    trades = exchange.fetch_trades(request.symbol, since=since_ms, limit=1000)
    if not trades:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")
    ohlcv = df["price"].resample(f"{seconds}S").ohlc()
    volume = df["amount"].resample(f"{seconds}S").sum()
    ohlcv["volume"] = volume
    ohlcv = ohlcv.dropna().reset_index()
    ohlcv = ohlcv.rename(columns={"timestamp": "timestamp"})
    return ohlcv[["timestamp", "open", "high", "low", "close", "volume"]]


def _estimate_period(interval: str, limit: int) -> str:
    if interval in {"1m", "5m", "15m", "1h"}:
        return "7d"
    if interval == "1d":
        return "1y"
    if interval in {"1wk", "1mo"}:
        return "5y"
    return "1mo"


def _interval_delta(interval: str, limit: int) -> timedelta:
    mapping = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
        "1wk": timedelta(weeks=1),
        "1mo": timedelta(days=30),
    }
    return mapping[interval] * limit
