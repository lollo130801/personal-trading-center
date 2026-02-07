from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from services.indicators import SelectedIndicator, compute_indicator, load_indicators
from services.market_data import MarketRequest, fetch_ohlcv


st.set_page_config(page_title="Personal Trading Center", layout="wide")

st.title("Personal Trading Center")

custom_indicator_dir = Path(__file__).parent / "custom_indicators"

with st.sidebar:
    st.header("Mercati")
    title_override = st.text_input("Titolo dashboard", "Mercati e Indicatori")
    source = st.selectbox("Fonte dati", ["binance", "yfinance"], index=0)
    symbol = st.text_input("Simbolo principale", "BTC/USDT" if source == "binance" else "AAPL")
    compare_symbols = st.text_input(
        "Confronta con (separati da virgola)",
        "ETH/USDT" if source == "binance" else "MSFT, TSLA",
    )
    interval = st.selectbox(
        "Intervallo",
        ["1s", "5s", "15s", "1m", "5m", "15m", "1h", "1d", "1wk", "1mo"],
        index=3,
    )
    limit = st.slider("Numero candele", min_value=50, max_value=500, value=200, step=10)
    st.divider()
    st.header("Grafico")
    chart_type = st.selectbox("Tipo grafico", ["Candlestick", "Linea", "Barre"], index=0)
    st.caption("Zoom verticale/orizzontale con scroll o box-select. Drag per pan.")
    auto_refresh = st.toggle("Aggiornamento automatico", value=False)
    refresh_seconds = st.number_input("Secondi tra refresh", min_value=1, max_value=60, value=5)

    st.divider()
    st.header("Indicatori")
    available_indicators = load_indicators(custom_indicator_dir)
    indicator_lookup = {indicator.name: indicator for indicator in available_indicators}
    selected_indicator_names = st.multiselect(
        "Seleziona indicatori",
        list(indicator_lookup.keys()),
        default=["SMA", "RSI", "Kalman"],
    )

    indicator_selections: List[SelectedIndicator] = []
    for name in selected_indicator_names:
        spec = indicator_lookup[name]
        with st.expander(f"{spec.name} · {spec.description}", expanded=True):
            placement = st.selectbox(
                "Posizione",
                ["overlay", "below", "tab"],
                index=["overlay", "below", "tab"].index(spec.placement),
                key=f"placement-{spec.name}",
            )
            params: Dict[str, float] = {}
            for param_name, default_value in spec.params.items():
                params[param_name] = st.number_input(
                    f"{param_name}",
                    value=float(default_value),
                    key=f"param-{spec.name}-{param_name}",
                )
            indicator_selections.append(
                SelectedIndicator(spec=spec, params=params, placement=placement)
            )

st.subheader(title_override)

if auto_refresh:
    if hasattr(st, "autorefresh"):
        st.autorefresh(interval=refresh_seconds * 1000, key="auto-refresh")
    else:
        st.info("Auto refresh non disponibile in questa versione di Streamlit.")

request = MarketRequest(
    symbol=symbol,
    interval=interval,
    limit=limit,
    source=source,
)

@st.cache_data(ttl=5)
def load_data(request: MarketRequest) -> pd.DataFrame:
    return fetch_ohlcv(request)


@st.cache_data(ttl=5)
def load_compare_data(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for item in symbols:
        compare_request = MarketRequest(
            symbol=item,
            interval=interval,
            limit=limit,
            source=source,
        )
        datasets[item] = fetch_ohlcv(compare_request)
    return datasets

try:
    data = load_data(request)
    compare_list = [item.strip() for item in compare_symbols.split(",") if item.strip()]
    compare_data = load_compare_data(compare_list) if compare_list else {}
except Exception as exc:  # noqa: BLE001
    st.error(f"Errore durante il fetch dei dati: {exc}")
    st.stop()

if data.empty:
    st.warning("Nessun dato disponibile per i parametri selezionati.")
    st.stop()

below_indicators = [ind for ind in indicator_selections if ind.placement == "below"]
overlay_indicators = [ind for ind in indicator_selections if ind.placement == "overlay"]
tab_indicators = [ind for ind in indicator_selections if ind.placement == "tab"]

rows = 1 + len(below_indicators)
row_heights = [0.7] + [0.3 / max(len(below_indicators), 1)] * len(below_indicators)
fig = make_subplots(
    rows=rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=row_heights,
)

if chart_type == "Candlestick":
    fig.add_trace(
        go.Candlestick(
            x=data["timestamp"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name=symbol,
            increasing_line_color="#16a34a",
            increasing_fillcolor="#22c55e",
            decreasing_line_color="#dc2626",
            decreasing_fillcolor="#ef4444",
        ),
        row=1,
        col=1,
    )
elif chart_type == "Linea":
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["close"],
            mode="lines",
            name=symbol,
        ),
        row=1,
        col=1,
    )
else:
    fig.add_trace(
        go.Bar(
            x=data["timestamp"],
            y=data["close"],
            name=symbol,
        ),
        row=1,
        col=1,
    )

if compare_data:
    for compare_symbol, compare_df in compare_data.items():
        if compare_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=compare_df["timestamp"],
                y=compare_df["close"],
                mode="lines",
                name=compare_symbol,
                line=dict(width=1.5, dash="dot"),
            ),
            row=1,
            col=1,
        )

for indicator in overlay_indicators:
    indicator_data = compute_indicator(indicator, data)
    for column in indicator_data.columns:
        if column == "timestamp":
            continue
        fig.add_trace(
            go.Scatter(
                x=indicator_data["timestamp"],
                y=indicator_data[column],
                mode="lines",
                name=column,
            ),
            row=1,
            col=1,
        )

for index, indicator in enumerate(below_indicators, start=2):
    indicator_data = compute_indicator(indicator, data)
    for column in indicator_data.columns:
        if column == "timestamp":
            continue
        fig.add_trace(
            go.Scatter(
                x=indicator_data["timestamp"],
                y=indicator_data[column],
                mode="lines",
                name=column,
            ),
            row=index,
            col=1,
        )

fig.update_layout(
    height=720,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=40, r=40, t=40, b=30),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    hovermode="x unified",
    dragmode="pan",
)
fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(148, 163, 184, 0.2)",
    rangeslider=dict(
        visible=True,
        bgcolor="rgba(148, 163, 184, 0.12)",
        bordercolor="rgba(148, 163, 184, 0.4)",
        borderwidth=1,
        thickness=0.06,
    ),
)
fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(148, 163, 184, 0.2)",
    fixedrange=False,
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config=dict(scrollZoom=True, displaylogo=False, responsive=True),
)

if tab_indicators:
    st.subheader("Indicatori in tab")
    tabs = st.tabs([ind.spec.name for ind in tab_indicators])
    for tab, indicator in zip(tabs, tab_indicators, strict=False):
        with tab:
            indicator_data = compute_indicator(indicator, data)
            tab_fig = go.Figure()
            for column in indicator_data.columns:
                if column == "timestamp":
                    continue
                tab_fig.add_trace(
                    go.Scatter(
                        x=indicator_data["timestamp"],
                        y=indicator_data[column],
                        mode="lines",
                        name=column,
                    )
                )
            tab_fig.update_layout(height=300, margin=dict(l=40, r=40, t=20, b=20))
            st.plotly_chart(tab_fig, use_container_width=True)

st.divider()
with st.expander("Come aggiungere indicatori personalizzati"):
    st.markdown(
        """
        Puoi aggiungere indicatori Python creando un file nella cartella `custom_indicators/`.
        Ogni file deve esportare una lista `INDICATORS` con istanze di `IndicatorSpec`.

        Esempio minimo:
        ```python
        from indicators.defaults import IndicatorSpec
        import pandas as pd

        def my_indicator(df: pd.DataFrame, params: dict) -> pd.DataFrame:
            return pd.DataFrame({"MyIndicator": df["close"].rolling(10).mean()})

        INDICATORS = [
            IndicatorSpec(
                name="MyIndicator",
                description="Esempio personalizzato",
                params={},
                placement="overlay",
                compute=my_indicator,
            )
        ]
        ```
        """
    )
