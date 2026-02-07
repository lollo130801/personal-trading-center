from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from pykalman import KalmanFilter


@dataclass
class IndicatorSpec:
    name: str
    description: str
    params: Dict[str, float]
    placement: str
    compute: Callable[[pd.DataFrame, Dict[str, float]], pd.DataFrame]


def _sma(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    window = int(params["window"])
    series = df["close"].rolling(window=window).mean()
    return pd.DataFrame({"SMA": series})


def _ema(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    span = int(params["span"])
    series = df["close"].ewm(span=span, adjust=False).mean()
    return pd.DataFrame({"EMA": series})


def _wma(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    window = int(params["window"])
    weights = np.arange(1, window + 1)
    series = df["close"].rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return pd.DataFrame({"WMA": series})


def _rsi(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    period = int(params["period"])
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return pd.DataFrame({"RSI": rsi})


def _fourier(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    components = int(params["components"])
    values = df["close"].to_numpy()
    fft = np.fft.fft(values)
    fft[components:-components] = 0
    filtered = np.real(np.fft.ifft(fft))
    return pd.DataFrame({"FourierFiltered": filtered})


def _kalman(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    transition_cov = params["transition_cov"]
    observation_cov = params["observation_cov"]
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=transition_cov,
        observation_covariance=observation_cov,
    )
    state_means, _ = kf.filter(df["close"].values)
    return pd.DataFrame({"Kalman": state_means.flatten()})


INDICATORS: List[IndicatorSpec] = [
    IndicatorSpec(
        name="SMA",
        description="Media mobile semplice",
        params={"window": 20},
        placement="overlay",
        compute=_sma,
    ),
    IndicatorSpec(
        name="EMA",
        description="Media mobile esponenziale",
        params={"span": 20},
        placement="overlay",
        compute=_ema,
    ),
    IndicatorSpec(
        name="WMA",
        description="Media mobile pesata",
        params={"window": 20},
        placement="overlay",
        compute=_wma,
    ),
    IndicatorSpec(
        name="RSI",
        description="Relative Strength Index",
        params={"period": 14},
        placement="below",
        compute=_rsi,
    ),
    IndicatorSpec(
        name="Fourier",
        description="Filtro tramite trasformata di Fourier",
        params={"components": 10},
        placement="overlay",
        compute=_fourier,
    ),
    IndicatorSpec(
        name="Kalman",
        description="Filtro di Kalman",
        params={"transition_cov": 0.01, "observation_cov": 1.0},
        placement="overlay",
        compute=_kalman,
    ),
]
