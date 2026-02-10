from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from indicators.defaults import INDICATORS as DEFAULT_INDICATORS
from indicators.defaults import IndicatorSpec


@dataclass
class SelectedIndicator:
    spec: IndicatorSpec
    params: Dict[str, float]
    placement: str


def load_indicators(custom_dir: Path) -> List[IndicatorSpec]:
    indicators = list(DEFAULT_INDICATORS)
    if not custom_dir.exists():
        return indicators

    for path in custom_dir.glob("*.py"):
        if path.name.startswith("__"):
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "INDICATORS"):
                indicators.extend(module.INDICATORS)
    return indicators


def compute_indicator(indicator: SelectedIndicator, df: pd.DataFrame) -> pd.DataFrame:
    data = indicator.spec.compute(df, indicator.params)
    data = data.copy()
    data["timestamp"] = df["timestamp"].values
    return data
