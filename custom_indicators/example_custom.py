from __future__ import annotations

import pandas as pd

from indicators.defaults import IndicatorSpec


def bollinger(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    window = int(params["window"])
    std = float(params["std"])
    mean = df["close"].rolling(window=window).mean()
    deviation = df["close"].rolling(window=window).std()
    upper = mean + std * deviation
    lower = mean - std * deviation
    return pd.DataFrame({"BB_Mean": mean, "BB_Upper": upper, "BB_Lower": lower})


INDICATORS = [
    IndicatorSpec(
        name="Bollinger",
        description="Bande di Bollinger",
        params={"window": 20, "std": 2.0},
        placement="overlay",
        compute=bollinger,
    )
]
